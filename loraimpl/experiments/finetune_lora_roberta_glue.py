import torch
from transformers import RobertaTokenizer

from loraimpl.data.glue import GLUEDataset
from loraimpl.models.lora_roberta import LoraWrapperRoberta
from loraimpl.utils.helper import train_epoch, evaluate_glue


def main():
    # Configuration
    num_epochs = 3
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
    model_config = {
        'task_type': 'glue',
        'lora_rank': 8,
        'train_biases': True,
        'train_embedding': False,
        'train_layer_norms': True
    }
    optimizer_config = {
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }

    run_experiment(model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config)


def run_experiment(model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_name = train_dataset_config['task_name']

    print(f"Running GLUE task: {task_name}")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load datasets
    train_dataset = GLUEDataset(tokenizer=tokenizer, **train_dataset_config)
    eval_dataset = GLUEDataset(tokenizer=tokenizer, **val_dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, **val_loader_config)

    # Initialize model
    model = LoraWrapperRoberta(num_classes=train_dataset.num_labels, **model_config)
    model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
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
        eval_metrics = evaluate_glue(model, eval_loader, criterion, device, task_name)
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
    main()
