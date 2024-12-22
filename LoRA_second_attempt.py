import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, AdamW

# -------------------------------
# 1) Your LoRALinear Implementation
# -------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.weight.requires_grad_(False)  # Freeze base weight

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
            self.bias.requires_grad_(False)  # Freeze bias
        else:
            self.bias = None

        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        base_output = x @ self.weight.T
        lora_output = self.alpha * (x @ self.A @ self.B)
        out = base_output + lora_output
        if self.use_bias:
            out += self.bias
        return out


# -------------------------------
# 2) Replace function
# -------------------------------
def replace_linear_with_lora(module, rank=4, alpha=8.0):
    """
    Recursively replaces all nn.Linear submodules with LoRALinear in a given module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            in_dim = child.in_features
            out_dim = child.out_features
            bias = child.bias is not None

            lora_linear = LoRALinear(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank,
                alpha=alpha,
                bias=bias,
            )
            # Copy the pretrained weights over
            with torch.no_grad():
                lora_linear.weight.copy_(child.weight)
                if bias:
                    lora_linear.bias.copy_(child.bias)

            setattr(module, name, lora_linear)
        else:
            replace_linear_with_lora(child, rank, alpha)


# -------------------------------
# 3) Main
# -------------------------------
def main():
    # 3.1) Load pretrained RoBERTa
    model = RobertaModel.from_pretrained("roberta-base")

    # 3.2) Replace all nn.Linear with LoRALinear
    replace_linear_with_lora(model, rank=4, alpha=8.0)

    # 3.3) Freeze all parameters except LoRA (A and B)
    #      One approach: set everything to requires_grad=False,
    #      then unfreeze only A & B.
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        # If parameter name includes ".A" or ".B", unfreeze it
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    # 3.4) Create a basic optimizer (only sees LoRA params)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 3.5) Dummy data + tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Some toy sentences
    texts = ["Hello world!", "How are you today?", "LoRA is awesome!", "I love PyTorch!"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Fake labels for demonstration (e.g., 4 text samples, 2 possible classes)
    labels = torch.tensor([0, 1, 0, 1])

    # 3.6) Simple training loop
    model.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # RobertaModel returns a dict with outputs, typically:
        #   last_hidden_state, pooler_output, hidden_states, attentions, etc.
        outputs = model(**inputs)
        # Let's pick the [CLS] token representation (for demonstration).
        # For RoBERTa, the "CLS" representation is often the first token embedding:
        hidden_state = outputs.last_hidden_state  # shape [batch_size, seq_len, hidden_dim]
        cls_rep = hidden_state[:, 0, :]           # shape [batch_size, hidden_dim]

        # A toy classification "logits" using a random linear head for demonstration
        # In a real scenario, you'd wrap your RobertaModel with a classification head
        # or use RobertaForSequenceClassification. But let's just do something quick:
        #   For example, let's assume the last dimension is 768; we make a quick linear:
        logits = nn.Linear(cls_rep.shape[1], 2)(cls_rep)  # shape [batch_size, 2]

        # Use cross-entropy
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch+1}] loss = {loss.item():.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
