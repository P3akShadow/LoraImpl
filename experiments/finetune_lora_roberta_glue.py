import torch
from transformers import RobertaTokenizer

from data.glue import GLUEDataset
from models.lora_roberta import LoraWrapperRoberta
from utils.helper import train_epoch, evaluate


def main():
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
    main()
