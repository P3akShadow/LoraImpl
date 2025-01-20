import torch
import transformers
from transformers import RobertaTokenizer
import wandb

from loraimpl.data.glue import GLUEDataset
from loraimpl.models.lora_roberta import LoraWrapperRoberta
from loraimpl.utils.helper import train_epoch, evaluate_glue, summarize_model

import random as rnd

def main():
    # Configuration
    num_epochs = 20
    model_name = 'roberta-base'
    model_config = {
        'task_type': 'glue',
        'lora_rank': 8,
        'train_biases': True,
        'train_embedding': False,
        'train_layer_norms': True
    }
    train_dataset_config = {
        'task_name': 'sst2',
        'split': 'train',
        'max_length': 128,
    }
    val_dataset_config = {
        'task_name': 'sst2',
        'split': 'validation',
        'max_length': 128,
    }
    train_loader_config = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    val_loader_config = {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True
    }
    optimizer_config = {
        'lr': 1e-4,
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

    #run experiments in loop
    for task in ["stsb", "cola", "sst2"]:
        for lora_rank in [1,2,4,8,16]:
            config["model_config"]["lora_rank"] = lora_rank
            config["train_dataset_config"]["task_name"] = task
            config["val_dataset_config"]["task_name"] = task

            wandb_name = f"finetune_lora_roberta_{lora_rank}_{task}_{rnd.randrange(1000)}"
            wandb.init(project="lora", config=config, name=wandb_name)
            run_experiment(**config)
            wandb.finish()


    #wandb.init(project="lora", config=config)
    #run_experiment(**config)




def run_experiment(model_name, model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config, seed=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    task_name = train_dataset_config['task_name']

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Load datasets
    train_dataset = GLUEDataset(tokenizer=tokenizer, **train_dataset_config)
    eval_dataset = GLUEDataset(tokenizer=tokenizer, **val_dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, **val_loader_config)

    # Initialize model
    model = LoraWrapperRoberta(num_classes=train_dataset.num_labels, **model_config)
    model.to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    criterion = torch.nn.MSELoss() if task_name == 'stsb' else torch.nn.CrossEntropyLoss()

    summarize_model(model, dataloader=train_loader, device=device)
    print(f"\nTraining for {num_epochs} epochs...")

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
        eval_metrics = evaluate_glue(model, eval_loader, criterion, device, task_name)
        print("\nValidation metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")

        # Save best model
        current_metric = {}
        if 'accuracy' in eval_metrics:
            current_metric = eval_metrics['accuracy']
        elif 'mcc' in eval_metrics:
            current_metric = eval_metrics['mcc']
        elif 'pearson' in eval_metrics:
            current_metric = eval_metrics['pearson']
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_lora_state_dict(f'lora_weights_{task_name}_best.pt')

        # Log metrics to Weights & Biases
        wandb.log({
            'epoch': epoch + 1,
            'loss': train_metrics['loss'],
            'validation_metrics': eval_metrics,
        })


    print("\nTraining completed!")
    print(f"Best validation metric: {best_metric:.4f}")


if __name__ == '__main__':
    main()
