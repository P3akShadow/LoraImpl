import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast
import sys
import wandb

from loraimpl.data.nlg import CollateFunction
from loraimpl.models.lora_gpt2 import GPT2LMHeadModelLora
from loraimpl.utils.helper import evaluate_nlg, summarize_model


def main():
    # Configuration
    config = {
        'num_epochs': 10,
        'model_cfg': {
            'name': 'gpt2',
            'kwargs': {  # LoRA hyperparameters from the paper
                'lora_rank': 4,
                'lora_alpha': 32,
            }
        },
        'dataset_cfg': {
            'name': 'GEM/e2e_nlg',
            'max_length': 128
        },
        'loader_cfg': {
            'batch_size': 8,
            'num_workers': 4,
        },
        'optimizer_cfg': {
            'lr': 2e-5,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'tokenizer_cfg': {
            'padding': True,
            'padding_side': 'left',
            'truncation': True,
            'return_tensors': 'pt'
        },
        'seed': 42
    }

    wandb_kwargs = {
        'project': 'lora-team',
        'config': config,
    }
    if len(sys.argv) > 1:  # Continue existing run?
        wandb_kwargs['id'] = sys.argv[1]
    with wandb.init(**wandb_kwargs) as run:  # Log configuration to Weights & Biases and run experiment
        run_experiment(**config, run=run, cont='id' in wandb_kwargs)
    # run_experiment(**config)

def run_experiment(num_epochs, model_cfg, dataset_cfg, loader_cfg, optimizer_cfg, tokenizer_cfg, seed=None, run=None, cont=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if cont:  # Load model from checkpoint
        if run is not None:
            run.restore('checkpoint')
        model = GPT2LMHeadModelLora.from_pretrained("checkpoint", local_files_only=True)
    else:  # Initialize model with LoRA only training (no biases or layer norms)
        model = GPT2LMHeadModelLora.from_pretrained(model_cfg['name'], **model_cfg['kwargs'])

    model.to(device)
    model.train()  # Ensure model starts in training mode

    train_dataset = load_dataset(dataset_cfg['name'], split='train')
    val_dataset = load_dataset(dataset_cfg['name'], split='validation')

    tokenizer = GPT2TokenizerFast.from_pretrained(model_cfg['name'], **tokenizer_cfg)

    collate_fn = CollateFunction(tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, **loader_cfg)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn.validation, shuffle=False, **loader_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)

    summarize_model(model, dataloader=train_loader, device=device)

    print('\nEvaluating before training...')
    metrics = evaluate_nlg(model, val_loader, tokenizer, device)

    print(f'\nTraining for {num_epochs} epochs...')

    if run is not None:
        wandb.log({
            'epoch': 0,
            'validation_metrics': metrics
        })
        run.save('checkpoint', policy='live')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Normal training step
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1} average loss: {avg_loss:.4f}')

        print('\nEvaluating...')
        metrics = evaluate_nlg(model, val_loader, tokenizer, device)

        print('\nValidation metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')

        model.save_pretrained('checkpoint')

        if run is not None:
            run.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'validation_metrics': metrics
            })
            run.save('checkpoint')

    print('\nTraining completed!')


if __name__ == '__main__':
    main()