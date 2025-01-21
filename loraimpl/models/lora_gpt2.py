import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from loraimpl.data.nlg import CollateFunction


class GPT2LMHeadModelLora(GPT2LMHeadModel):
    """Extend GPT2LMHeadModel to support LoRA."""

    def __init__(self, config, lora_rank=64, lora_alpha=16):
        super().__init__(config)

        # replace all attention layers with LoRA-extended versions
        for i, block in enumerate(self.transformer.h):
            block.attn = GPT2AttentionLora(self.config, lora_rank=lora_rank, lora_alpha=lora_alpha)

        self.freeze_non_lora()

    def freeze_non_lora(self):
        for name, param in self.named_parameters():
            if "lora" not in name:
                param.requires_grad = False


class GPT2AttentionLora(GPT2Attention):
    """Extend GPT2Attention to support LoRA."""
    def __init__(self, config, lora_rank=64, lora_alpha=16):
        super().__init__(config)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Initialize Conv1D layers for LoRA
        if self.is_cross_attention:
            self.c_attn = Conv1DLora(2 * self.embed_dim, self.embed_dim, lora_rank, lora_alpha)
            self.q_attn = Conv1DLora(self.embed_dim, self.embed_dim, lora_rank, lora_alpha)
        else:
            self.c_attn = Conv1DLora(3 * self.embed_dim, self.embed_dim, lora_rank, lora_alpha)


class Conv1DLora(Conv1D):
    """Extend nn.Conv1D to support LoRA."""
    def __init__(self, nx, nf, lora_rank=64, lora_alpha=16):
        super().__init__(nx, nf)
        # hyperparameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank
        # initialize A random and B to zeros
        self.lora_a = nn.Parameter(torch.rand(nf, lora_rank))
        self.lora_b = nn.Parameter(torch.zeros(lora_rank, nx))

    def __repr__(self) -> str:
        return f"Conv1DLora (nf={self.nf}, nx={self.nx}, rank={self.lora_rank}, alpha={self.lora_alpha})"

    def forward(self, x):
        lora_weight = self.weight.addmm(self.lora_a, self.lora_b, alpha=self.scaling)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), lora_weight)
        return x.view(size_out)


if __name__ == "__main__":
    # Example of how to load the models
    # model = GPT2LMHeadModelLora.from_pretrained("gpt2", lora_rank=8, lora_alpha=16)
    from torchinfo import summary
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset

    model = GPT2LMHeadModelLora.from_pretrained("gpt2")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.padding_side = "left"

    # TorchInfo Summaries
    summary(model, depth=7)

    train_dataset = load_dataset('GEM/e2e_nlg', split='validation')
    collate_fn = CollateFunction(tokenizer)
    data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn.evaluation, batch_size=2)

    model.eval()
    test_input, _ = next(iter(data_loader))
    test_output = model.generate(**test_input)
    decoded = tokenizer.decode(test_output[0], skip_special_tokens=False)
    print(decoded)







