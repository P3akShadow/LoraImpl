from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch
from datasets import load_dataset

# Load RoBERTa tokenizer and model
def load_roberta(model_name="roberta-base", num_labels=2):
    """
    Loads the RoBERTa model and tokenizer for sequence classification.

    Args:
        model_name (str): Name of the pre-trained RoBERTa model.
        num_labels (int): Number of labels for classification.

    Returns:
        model, tokenizer: Pre-trained RoBERTa model and tokenizer.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

# Prepare the dataset
def preprocess_data(dataset, tokenizer, max_length=128):
    """
    Tokenize and preprocess the dataset.

    Args:
        dataset: The dataset to preprocess.
        tokenizer: Tokenizer instance.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        tokenized_dataset: Preprocessed dataset.
    """
    def tokenize_fn(example):
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=max_length)

    return dataset.map(tokenize_fn, batched=True)

# Training loop
def train_model(model, train_loader, optimizer, device):
    """
    Train the RoBERTa model.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for training.
        device: Device to train on (CPU or GPU).
    """
    model.train()
    loss_fn = CrossEntropyLoss()

    for epoch in range(3):  # Training for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Move data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    # Load pre-trained RoBERTa and tokenizer
    model_name = "roberta-base"
    model, tokenizer = load_roberta(model_name=model_name, num_labels=2)

    # Load and preprocess the dataset
    dataset = load_dataset("imdb", split="train[:10%]")  # Example dataset (IMDb)
    tokenized_dataset = preprocess_data(dataset, tokenizer)

    # Prepare DataLoader
    train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train the model
    train_model(model, train_loader, optimizer, device)

