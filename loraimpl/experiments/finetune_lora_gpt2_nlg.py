import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
import sys
import wandb

from loraimpl.data.nlg import CollateFunction
from loraimpl.models import gpt2_modifications
from loraimpl.utils.helper import evaluate_nlg, summarize_model


def main():
    # Configuration
    config = {
        'num_epochs': 10,
        'model_cfg': {
            'name': 'gpt2',
            'modification': 'lora',  # lora, adapter or none
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
            'padding_side': 'right',
            'truncation': True,
            'return_tensors': 'pt'
        },
        'inference_cfg': {
            'num_beams': 10,
            'no_repeat_ngram_size': 4,
            'length_penalty': 0.9,
            'max_length': 300,
            'early_stopping': True,
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
    if wandb_kwargs.get('id', '') == '--no-wandb':
        run_experiment(**config)
        return
    with wandb.init(**wandb_kwargs) as run:  # Log configuration to Weights & Biases and run experiment
        run_experiment(**config, run=run)

def run_experiment(num_epochs, model_cfg, dataset_cfg, loader_cfg, optimizer_cfg, tokenizer_cfg, inference_cfg, seed=None, run=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = gpt2_modifications[model_cfg['modification']]  # GPT2LMHeadModelLora, GPT2LMHeadModelFinetuning or GPT2LMHeadModel (original)

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if run is not None and run.resumed:  # Load model from checkpoint
        print('Restoring checkpoint...')
        run.restore('checkpoint/config.json', replace=True)
        run.restore('checkpoint/generation_config.json', replace=True)
        run.restore('checkpoint/model.safetensors', replace=True)
        run.restore('checkpoint/optimizer.pt', replace=True)
        run.restore('checkpoint/scheduler.pt', replace=True)
        print('Loading model from checkpoint...')
        model = model_cls.from_pretrained("checkpoint", local_files_only=True, **model_cfg['kwargs'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
        optimizer.load_state_dict(torch.load("checkpoint/optimizer.pt", map_location=device, weights_only=False))
        start_epoch = run.summary['epoch']
    else:  # Initialize model with LoRA only training (no biases or layer norms)
        model = model_cls.from_pretrained(model_cfg['name'], **model_cfg['kwargs'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
        start_epoch = 0


    tokenizer = GPT2TokenizerFast.from_pretrained(model_cfg['name'], **tokenizer_cfg)

    collate_fn = CollateFunction(tokenizer)

    train_dataset = load_dataset(dataset_cfg['name'], split='train')
    val_dataset = load_dataset(dataset_cfg['name'], split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, **loader_cfg)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn.evaluation, shuffle=False, **(loader_cfg | {'batch_size': 1}))
    # TODO: Note: huggingface warns if padding is not left for generating,
    #   but that confuses the model with batch evaluation...
    #   Also the model documentation says it should be right because of absolute position embeddings.
    #   We'll keep batch size 1 for evaluating, which is less efficient but shouldn't use padding at all.

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=(num_epochs - start_epoch) * len(train_loader))
    if run is not None and run.resumed:
        scheduler.load_state_dict(torch.load("checkpoint/scheduler.pt", map_location=device), weights_only=False)

    summarize_model(model, dataloader=train_loader, device=device)

    metrics = evaluate_nlg(model, val_loader, tokenizer, device, inference_cfg)
    if run is not None and not run.resumed:
        print('\nEvaluating before training...')
        wandb.log({
            'epoch': 0,
            'validation_metrics': metrics
        })

    print(f'\nTraining for {num_epochs} epochs...')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
            batch = {k: v.to(device) for k, v in batch.items()}  # noqa

            # Normal training step
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1} average loss: {avg_loss:.4f}')

        print('\nEvaluating...')
        metrics = evaluate_nlg(model, val_loader, tokenizer, device, inference_cfg)

        print('\nValidation metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')

        model.save_pretrained('checkpoint')
        torch.save(optimizer.state_dict(), 'checkpoint/optimizer.pt')
        torch.save(scheduler.state_dict(), 'checkpoint/scheduler.pt')

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