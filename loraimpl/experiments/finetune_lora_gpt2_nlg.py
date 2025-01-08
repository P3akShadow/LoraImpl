import torch
import transformers
from tqdm import tqdm
from transformers import get_scheduler, GPT2TokenizerFast
import wandb

from loraimpl.data.nlg import NLGDataset
from loraimpl.models.lora_gpt2 import GPT2LMHeadModelLora
from loraimpl.utils.helper import evaluate_nlg, summarize_model


def main():
    # Configuration
    num_epochs = 10
    model_name = 'gpt2-large'
    model_config = {
        'lora_rank': 8,
        'lora_alpha': 16,
    }
    train_dataset_config = {
        'split': 'train',
        'max_length': 128
    }
    val_dataset_config = {
        'split': 'validation',
        'max_length': 128
    }
    train_loader_config = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    val_loader_config = {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True
    }
    optimizer_config = {
        'lr': 2e-5,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    seed = 42

    # Log configuration to Weights & Biases and run experiment
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'model_config': model_config,
        'train_dataset_config': train_dataset_config,
        'val_dataset_config': val_dataset_config,
        'train_loader_config': train_loader_config,
        'val_loader_config': val_loader_config,
        'optimizer_config': optimizer_config,
        'seed': seed,
    }
    wandb.init(project="lora", config=config)
    run_experiment(**config)

def run_experiment(model_name, model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config, seed=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize model with LoRA only training (no biases or layer norms)
    model = GPT2LMHeadModelLora.from_pretrained(model_name, **model_config)
    model.to(device)
    model.train()  # Ensure model starts in training mode
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    train_dataset = NLGDataset(**train_dataset_config)
    val_dataset = NLGDataset(**val_dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)

    num_training_steps = len(train_loader) * 10
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    scaler = torch.cuda.amp.GradScaler()
    best_bleu = 0
    patience = 3
    no_improve = 0

    summarize_model(model, dataloader=train_loader, device=device)
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Normal training step
            model.train()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

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
        metrics = evaluate_nlg(model, val_loader, tokenizer, device)

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
                # print("\nEarly stopping triggered!")
                # break
                print()

        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'validation_metrics': metrics
        })

    print("\nTraining completed!")


if __name__ == '__main__':
    main()