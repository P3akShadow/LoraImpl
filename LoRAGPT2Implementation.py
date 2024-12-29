# pip install rouge-score

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import evaluate
from datasets import load_dataset
from transformers import get_scheduler
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler 


class CustomLoRAGPT2Attention(GPT2Attention):
    def __init__(self, config, lora_rank=64, lora_alpha=128):
        super().__init__(config)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Initialize LoRA matrices
        embed_dim = config.hidden_size
        self.q_lora_a = nn.Parameter(torch.zeros(embed_dim, self.lora_rank))
        self.q_lora_b = nn.Parameter(torch.zeros(self.lora_rank, embed_dim))
        self.v_lora_a = nn.Parameter(torch.zeros(embed_dim, self.lora_rank))
        self.v_lora_b = nn.Parameter(torch.zeros(self.lora_rank, embed_dim))

        # Better initialization
        for param in [self.q_lora_a, self.q_lora_b, self.v_lora_a, self.v_lora_b]:
            nn.init.xavier_uniform_(param, gain=0.1)

        # Add layer norm
        self.lora_ln = nn.LayerNorm(embed_dim)
        # Dropout
        self.lora_dropout = nn.Dropout(p=0.05)

        # Freeze all original parameters
        for name, param in self.named_parameters():
            if not any(x in name for x in ['lora_a', 'lora_b', 'lora_ln']):
                param.requires_grad_(False)
            else:
                print(f"LoRA trainable param: {name}")

    def lora_query(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(x)
        query, _, _ = qkv.split(self.split_size, dim=2)
        x_ln = self.lora_ln(x)
        lora_output = x_ln @ self.lora_dropout(self.q_lora_a) @ self.q_lora_b
        return query + (self.lora_alpha * lora_output)

    def lora_value(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(x)
        _, _, value = qkv.split(self.split_size, dim=2)
        x_ln = self.lora_ln(x)
        lora_output = x_ln @ self.lora_dropout(self.v_lora_a) @ self.v_lora_b
        return value + (self.lora_alpha * lora_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        qkv = self.c_attn(hidden_states)
        _, key, _ = qkv.split(self.split_size, dim=2)

        query = self.lora_query(hidden_states)
        value = self.lora_value(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_output = torch.matmul(query, key.transpose(-1, -2))
        attn_output = attn_output / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_output = attn_output + attention_mask

        attn_weights = F.softmax(attn_output, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def _split_heads(self, tensor, num_heads, head_dim):
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, head_dim):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)


class LoraWrapperGPT2NLG(nn.Module):
    def __init__(
        self,
        model_id="gpt2",
        lora_rank=32,
        lora_alpha=64,
        train_biases=False,
        train_layer_norms=False
    ):
        super().__init__()

        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.config = self.model.config

        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.train_biases = train_biases
        self.train_layer_norms = train_layer_norms

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.freeze_parameters_except_lora()
        self.replace_attention_layers()

        # Enable gradient checkpointing if desired
        self.model.gradient_checkpointing_disable()

        self.print_trainable_parameters()

    def freeze_parameters_except_lora(self):
        for name, param in self.model.named_parameters():
            should_train = False
            if any(x in name for x in ['lora_a', 'lora_b', 'lora_ln']):
                should_train = True
            elif self.train_biases and "bias" in name:
                should_train = True
            elif self.train_layer_norms and "ln" in name:
                should_train = True
            param.requires_grad_(should_train)

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params} || all params: {all_params} || "
            f"trainable%: {100*trainable_params/all_params:.2f}"
        )

    def replace_attention_layers(self):
        count = 0
        for block in self.model.transformer.h:
            if hasattr(block, 'attn'):
                block.attn = CustomLoRAGPT2Attention(
                    config=self.config,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha
                )
                count += 1
        print(f"Replaced {count} GPT2Attention layers with CustomLoRAGPT2Attention")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        We do NOT forcibly pass `use_cache=False` here, to avoid
        'got multiple values for argument use_cache'.
        We rely on config.use_cache=False or the caller if needed.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        default_params = {
            'max_new_tokens': 64,
            'do_sample': True,
            'top_p': 0.92,
            'temperature': 0.7,
            'num_beams': 5,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        default_params.update(kwargs)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **default_params
        )


class NLGDataset(Dataset):
    """
    Simple dataset for E2E NLG, 
    with a tokenization approach that partially masks the 'input' portion.
    """
    def __init__(self, split='train', max_length=128):
        self.data = load_dataset('GEM/e2e_nlg')
        self.data = self.data[split]
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length

        self.inputs = []
        self.targets = []

        for item in self.data:
            # simpler approach: 
            input_text = item["meaning_representation"]
            target_text = item["target"]
            if not target_text.endswith('.'):
                target_text += '.'

            combined = f"{input_text}\n{target_text}"
            self.inputs.append(input_text)
            self.targets.append(target_text)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        # Full combined
        full_text = input_text + "\n" + target_text

        # tokenize full
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # We'll partially mask the input portion
        labels = encoding["input_ids"].clone()

        # The input portion is everything up to the newline + 1
        # (just a naive approach - adjust if you want more precise masking)
        input_length = len(self.tokenizer.encode(input_text)) + 1  
        labels[0, :input_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def evaluate_nlg(model, eval_loader, tokenizer, device):
    model.eval()

    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # we won't heavily tune generation here
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.85,
                temperature=0.6,
                num_beams=5,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True,
            )

            for generated_seq in outputs:
                decoded = tokenizer.decode(generated_seq, skip_special_tokens=True)
                all_preds.append(decoded.strip())

            for lbl in batch['labels']:
                # Convert the -100 to pad
                lbl = torch.where(lbl == -100, tokenizer.pad_token_id, lbl)
                decoded = tokenizer.decode(lbl, skip_special_tokens=True)
                all_refs.append(decoded.strip())

    # Show an example
    if len(all_preds) > 0:
        print("\nExample generation:")
        print("Pred:", all_preds[0])
        print("Ref: ", all_refs[0])

    # Evaluate
    results = {}
    results["bleu"] = bleu.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs]
    )["bleu"]

    r_scores = rouge.compute(predictions=all_preds, references=all_refs)
    results["rouge1"] = r_scores["rouge1"]
    results["rouge2"] = r_scores["rouge2"]
    results["rougeL"] = r_scores["rougeL"]

    results["meteor"] = meteor.compute(
        predictions=all_preds,
        references=all_refs
    )["meteor"]

    return results


def verify_parameters(model):
    """
    Print which parameters are LoRA-trainable vs non-LoRA, etc.
    """
    stats = {
        'lora_trainable': 0,
        'lora_frozen': 0,
        'non_lora_trainable': 0,
        'non_lora_frozen': 0,
        'lora_param_count': 0,
        'total_param_count': 0,
    }

    print("\nDetailed Parameter Status:")
    print("-" * 50)

    for name, param in model.named_parameters():
        stats['total_param_count'] += param.numel()
        is_lora = any(x in name for x in ['lora_a', 'lora_b', 'lora_ln'])

        if is_lora:
            if param.requires_grad:
                stats['lora_trainable'] += 1
                stats['lora_param_count'] += param.numel()
                print(f"✓ LoRA trainable: {name}")
            else:
                stats['lora_frozen'] += 1
                print(f"✗ LoRA frozen: {name}")
        else:
            if param.requires_grad:
                stats['non_lora_trainable'] += 1
                print(f"! Non-LoRA trainable: {name}")
            else:
                stats['non_lora_frozen'] += 1

    print("\nSummary:")
    print(f"LoRA trainable parameters: {stats['lora_trainable']}")
    print(f"LoRA frozen parameters: {stats['lora_frozen']}")
    print(f"Non-LoRA trainable parameters: {stats['non_lora_trainable']}")
    print(f"Non-LoRA frozen parameters: {stats['non_lora_frozen']}")
    print(f"Total parameters: {stats['total_param_count']:,}")
    print(f"LoRA parameters: {stats['lora_param_count']:,}")
    print(f"LoRA percentage: {100.0 * stats['lora_param_count']/stats['total_param_count']:.2f}%")

    return stats


def verify_gradients(model, loss):
    """
    Print out gradient stats after a backward pass,
    to confirm LoRA grads flow and others are zero.
    """
    print("\nGradient verification after backward pass:")
    grad_stats = {
        'lora_with_grad': 0,
        'non_lora_with_grad': 0,
        'frozen_with_grad': 0
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm_val = param.grad.norm().item()
            is_lora = any(x in name for x in ['lora_a', 'lora_b', 'lora_ln'])

            if param.requires_grad:
                if is_lora:
                    grad_stats['lora_with_grad'] += 1
                    print(f"✓ LoRA gradient: {name}, norm={norm_val:.4f}")
                else:
                    grad_stats['non_lora_with_grad'] += 1
                    print(f"! Non-LoRA gradient: {name}, norm={norm_val:.4f}")
            else:
                grad_stats['frozen_with_grad'] += 1
                print(f"✗ Frozen param has grad: {name}, norm={norm_val:.4f}")

    print("\nGradient Summary:")
    for k,v in grad_stats.items():
        print(f"{k} = {v}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize model with LoRA only training (no biases or layer norms)
    model = LoraWrapperGPT2NLG(
        model_id='gpt2',
        lora_rank=32,
        lora_alpha=64,
        train_biases=False,
        train_layer_norms=False
    )
    model.to(device)
    model.train()  # Ensure model starts in training mode

    # Verify parameters before training
    print("\nVerifying parameters before training starts:")
    initial_stats = verify_parameters(model)

    train_dataset = NLGDataset(split='train', max_length=128)
    val_dataset = NLGDataset(split='validation', max_length=128)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
        
    optimizer = optim.AdamW(
        trainable_params,
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    num_training_steps = len(train_loader) * 10
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    num_epochs = 10
    best_bleu = 0
    patience = 3
    no_improve = 0

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Verify parameters at start of first epoch
        if epoch == 0:
            print(f"\nVerifying parameters at start of epoch {epoch + 1}")
            verify_parameters(model)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Verify gradients on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"\nVerifying gradients on first batch:")
                model.train()
                # Verification forward pass
                with torch.set_grad_enabled(True):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    verification_loss = outputs.loss
                    
                    print(f"Loss requires grad: {verification_loss.requires_grad}")
                    print(f"Loss grad_fn: {verification_loss.grad_fn}")
                    
                    if verification_loss.requires_grad:
                        verification_loss.backward()
                        verify_gradients(model, verification_loss)
                    else:
                        raise ValueError("Verification loss does not require gradients!")
                        
                optimizer.zero_grad(set_to_none=True)

            # Normal training step
            model.train()
            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                # Check if loss requires gradients
                if not loss.requires_grad:
                    raise ValueError("Training loss does not require gradients!")
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} average loss: {avg_loss:.4f}")

        print("\nEvaluating...")
        model.eval()
        metrics = evaluate_nlg(model, val_loader, model.tokenizer, device)

        print("\nValidation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        if metrics['bleu'] > best_bleu:
            best_bleu = metrics['bleu']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_bleu': best_bleu,
            }, 'best_model.pt')
            print(f"\nSaved new best model with BLEU score: {best_bleu:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("\nEarly stopping triggered!")
                break

    print("\nTraining completed!")

    # Final parameter verification
    print("\nVerifying parameters after training:")
    final_stats = verify_parameters(model)

    print("\nComparing initial vs final stats:")
    print("-" * 50)
    for key in initial_stats:
        if initial_stats[key] != final_stats[key]:
            print(f"WARNING: {key} changed during training!")
            print(f"  Initial: {initial_stats[key]}")
            print(f"  Final:   {final_stats[key]}")
        else:
            print(f"✓ {key} remained constant: {initial_stats[key]}")

if __name__ == "__main__":
    main()
