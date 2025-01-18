import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast
import sys
import wandb

from loraimpl.data.nlg import CollateFunction
from loraimpl.models.bareFinetuning_gpt2 import GPT2LMHeadModelFinetuning
from loraimpl.utils.helper import evaluate_nlg, summarize_model


def main():
    # Configuration
    config = {
        'num_epochs': 10,
        'model_cfg': {
            'name': 'gpt2',
            'kwargs': {  # LoRA hyperparameters from the paper

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
        'entity': 'maximiliansuess',
        'project': 'lora',
        'config': config,
    }
    if len(sys.argv) > 1:  # Continue existing run?
        wandb_kwargs['id'] = sys.argv[1]
        wandb_kwargs['resume'] = 'must'
        print(f'Trying to resume run {sys.argv[1]}...')
    with wandb.init(**wandb_kwargs) as run:  # Log configuration to Weights & Biases and run experiment
        run_experiment(**config, run=run, cont=run.resumed)

def run_experiment(num_epochs, model_cfg, dataset_cfg, loader_cfg, optimizer_cfg, tokenizer_cfg, seed=None, run=None, cont=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if cont:  # Load model from checkpoint
        if run is not None:
            print('Restoring checkpoint...')
            run.restore('checkpoint/config.json', replace=True)
            run.restore('checkpoint/generation_config.json', replace=True)
            run.restore('checkpoint/model.safetensors', replace=True)
            run.restore('checkpoint/optimizer.pt', replace=True)
        print('Loading model from checkpoint...')
        model = GPT2LMHeadModelFinetuning.from_pretrained("checkpoint", local_files_only=True, **model_cfg['kwargs'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
        optimizer.load_state_dict(torch.load("checkpoint/optimizer.pt", map_location=device, weights_only=False))
        start_epoch = run.summary['epoch']
    else:  # Initialize model with LoRA only training (no biases or layer norms)
        model = GPT2LMHeadModelFinetuning.from_pretrained(model_cfg['name'], **model_cfg['kwargs'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
        start_epoch = 0


    tokenizer = GPT2TokenizerFast.from_pretrained(model_cfg['name'], **tokenizer_cfg)

    collate_fn = CollateFunction(tokenizer)

    train_dataset = load_dataset(dataset_cfg['name'], split='train')
    val_dataset = load_dataset(dataset_cfg['name'], split='validation')
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, **loader_cfg)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn.validation, shuffle=False, **loader_cfg)

    summarize_model(model, dataloader=train_loader, device=device)

    if not cont and run is not None:
        print('\nEvaluating before training...')
        metrics = evaluate_nlg(model, val_loader, tokenizer, device)
        wandb.log({
            'epoch': 0,
            'validation_metrics': metrics
        })

    print(f'\nTraining for {num_epochs} epochs...')

    for epoch in range(start_epoch, num_epochs):
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
        torch.save(optimizer.state_dict(), 'checkpoint/optimizer.pt')

        if run is not None:
            run.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'validation_metrics': metrics
            })
            run.save('checkpoint/*', policy='now')

    print('\nTraining completed!')


if __name__ == '__main__':
    main()
