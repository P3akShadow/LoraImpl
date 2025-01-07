import torch
from tqdm import tqdm
from transformers import get_scheduler
import wandb

from loraimpl.data.nlg import NLGDataset
from loraimpl.models.lora_gpt2 import LoraWrapperGPT2NLG, verify_parameters, verify_gradients
from loraimpl.utils.helper import evaluate_nlg


def run_experiment(model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize model with LoRA only training (no biases or layer norms)
    model = LoraWrapperGPT2NLG(**model_config)
    model.to(device)
    model.train()  # Ensure model starts in training mode

    # Verify parameters before training
    print("\nVerifying parameters before training starts:")
    initial_stats = verify_parameters(model)

    train_dataset = NLGDataset(**train_dataset_config)
    val_dataset = NLGDataset(**val_dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(trainable_params, **optimizer_config)

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


def main():
    model_config = {
        'model_id': 'gpt2',
        'lora_rank': 32,
        'lora_alpha': 64,
        'train_biases': False,
        'train_layer_norms': False
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
    train_dataset_config = {
        'split': 'train',
        'max_length': 128
    }
    val_dataset_config = {
        'split': 'validation',
        'max_length': 128
    }
    optimizer_config = {
        'lr': 2e-5,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    num_epochs = 10

    run_experiment(model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config)


if __name__ == '__main__':
    main()