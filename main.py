"""The main file with the training and evaluation loop."""
import torch
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from transformers import RobertaTokenizer, get_scheduler

from data.glue import GLUEDataset
from data.nlg import NLGDataset
from models.lora_gpt2 import LoraWrapperGPT2NLG, verify_parameters, verify_gradients
from models.lora_roberta import LoraWrapperRoberta


def compute_metrics(task_name, preds, labels):
    """Compute metrics for GLUE tasks"""
    if task_name == 'cola':
        return {'mcc': matthews_corrcoef(labels, preds)}
    elif task_name in ['sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {'accuracy': acc, 'f1': f1}
    elif task_name == 'stsb':
        return {'pearson': np.corrcoef(preds, labels)[0, 1],
                'spearman': scipy.stats.spearmanr(preds, labels)[0]}
    elif task_name == 'mnli':
        return {'accuracy': accuracy_score(labels, preds)}


def train_epoch(model, train_loader, optimizer, criterion, device, task_name):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        if task_name == 'stsb':
            loss = criterion(outputs.squeeze(), labels)
            preds = outputs.squeeze().detach().cpu().numpy()
        else:
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    metrics = compute_metrics(task_name, np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(train_loader)
    return metrics


def evaluate(model, eval_loader, criterion, device, task_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            if task_name == 'stsb':
                loss = criterion(outputs.squeeze(), labels)
                preds = outputs.squeeze().cpu().numpy()
            else:
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(task_name, np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(eval_loader)
    return metrics


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


def main_gpt2():
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(
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

    scaler = torch.cuda.amp.GradScaler()
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
                with torch.cuda.amp.autocast():
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

def main_roberta():
    # Configuration
    task_name = 'sst2'  # Choose from GLUE tasks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-4
    max_length = 128

    print(f"Running GLUE task: {task_name}")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load datasets
    train_dataset = GLUEDataset(task_name, 'train', tokenizer, max_length)
    eval_dataset = GLUEDataset(task_name, 'validation', tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    # Initialize model
    num_labels = train_dataset.num_labels
    model = LoraWrapperRoberta(
        task_type='glue',
        num_classes=num_labels,
        lora_rank=8,
        train_biases=True,
        train_embedding=False,
        train_layer_norms=True
    )
    model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss() if task_name == 'stsb' else torch.nn.CrossEntropyLoss()

    # Training loop
    best_metric = float('-inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, task_name)
        print("\nTraining metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")

        # Evaluation
        eval_metrics = evaluate(model, eval_loader, criterion, device, task_name)
        print("\nValidation metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")

        # Save best model
        current_metric = eval_metrics['accuracy'] if 'accuracy' in eval_metrics else eval_metrics['mcc']
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_lora_state_dict(f'lora_weights_{task_name}_best.pt')

    print("\nTraining completed!")
    print(f"Best validation metric: {best_metric:.4f}")


if __name__ == '__main__':
    # main_roberta()
    main_gpt2()