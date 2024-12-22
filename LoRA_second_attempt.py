import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    AdamW
)
from datasets import load_dataset

#######################################
# 1) LoRALinear Implementation
#######################################
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        # Freeze base weight
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.weight.requires_grad_(False)

        # Optional bias
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
            self.bias.requires_grad_(False)
        else:
            self.bias = None

        # Trainable low-rank factors
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # base_output: x @ W^T
        base_output = x @ self.weight.T
        # LoRA update: alpha * (x @ A @ B)
        lora_output = self.alpha * (x @ self.A @ self.B)
        out = base_output + lora_output
        if self.use_bias:
            out += self.bias
        return out


#######################################
# 2) Helper: Replace nn.Linear => LoRALinear
#######################################
def replace_linear_with_lora(module, rank=4, alpha=8.0):
    """
    Recursively replace all nn.Linear submodules in `module` with LoRALinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            in_dim = child.in_features
            out_dim = child.out_features
            bias = child.bias is not None

            # Construct LoRALinear
            lora_linear = LoRALinear(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank,
                alpha=alpha,
                bias=bias,
            )

            # Copy pretrained weights
            with torch.no_grad():
                lora_linear.weight.copy_(child.weight)
                if bias:
                    lora_linear.bias.copy_(child.bias)

            # Replace in the parent module
            setattr(module, name, lora_linear)
        else:
            replace_linear_with_lora(child, rank, alpha)


#######################################
# 3) Main script
#######################################
def main():
    # -------------------------------------------------
    # A) Load roberta-base
    # -------------------------------------------------
    model = RobertaModel.from_pretrained("roberta-base")

    # -------------------------------------------------
    # B) Replace Linear => LoRALinear
    # -------------------------------------------------
    replace_linear_with_lora(model, rank=4, alpha=8.0)

    # -------------------------------------------------
    # C) Freeze all except LoRA parameters
    # -------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    # -------------------------------------------------
    # D) Prepare SST-2 data
    # -------------------------------------------------
    dataset = load_dataset("glue", "sst2")

    # check the top of the dataset
    print(dataset)

    # The Roberta tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=64
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # -------------------------------------------------
    # E) Classification head (simple)
    # -------------------------------------------------
    # We'll define a small classifier on top of RobertaModel's outputs:
    class RobertaWithClassificationHead(nn.Module):
        def __init__(self, base_model, hidden_size=768, num_labels=2):
            super().__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # For roberta, [CLS] embedding is often the first token hidden state
            hidden_states = outputs.last_hidden_state
            cls_token_emb = hidden_states[:, 0, :]
            logits = self.classifier(cls_token_emb)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

            return logits, loss

    # Wrap the LoRA-augmented roberta in a classification head
    model_with_head = RobertaWithClassificationHead(model, hidden_size=768, num_labels=2)

    # -------------------------------------------------
    # F) Optimizer: only LoRA + classification head
    # -------------------------------------------------
    # We do want to train the classification head weights, so let's unfreeze them:
    for param in model_with_head.classifier.parameters():
        param.requires_grad = True

    # Now build an optimizer for everything that is trainable
    optimizer = AdamW(filter(lambda p: p.requires_grad, model_with_head.parameters()), lr=1e-4)

    # -------------------------------------------------
    # G) Training Loop (simplified)
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_head.to(device)

    epochs = 5
    for epoch in range(epochs):
        print("We are currently on epoch ", epoch)
        model_with_head.train()
        total_loss = 0
        for batch_id,batch in enumerate(train_loader):
            if(batch_id % 100 == 0):
                print(f"Batch {batch_id}, {len(train_loader)}")
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, loss = model_with_head(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("We got out!!! \o/")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

        # Optional: Evaluate on val set
        model_with_head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits, loss = model_with_head(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
