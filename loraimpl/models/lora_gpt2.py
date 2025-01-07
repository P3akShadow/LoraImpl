import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class GPT2LMHeadModelLora(GPT2LMHeadModel):
    """Extend GPT2LMHeadModel to support LoRA."""

    def __init__(self, config, lora_rank=64, lora_alpha=128):
        super().__init__(config)

        # replace all attention layers with LoRA-extended versions
        for i, block in enumerate(self.transformer.h):
            block.attn = GPT2AttentionLora(self.config, lora_rank=lora_rank, lora_alpha=lora_alpha)

        self.freeze_non_lora()

    def freeze_non_lora(self):
        for name, param in self.named_parameters():
            print(name)
            if "lora" not in name:
                param.requires_grad = False


class GPT2AttentionLora(GPT2Attention):
    """Extend GPT2Attention to support LoRA."""
    def __init__(self, config, lora_rank=64, lora_alpha=128):
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
    def __init__(self, nx, nf, lora_rank=64, lora_alpha=128):
        super().__init__(nx, nf)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_a = nn.Parameter(torch.zeros(lora_rank, nx))
        self.lora_b = nn.Parameter(torch.zeros(nf, lora_rank))

    def __repr__(self) -> str:
        return f"Conv1DLora (nf={self.nf}, nx={self.nx}, rank={self.lora_rank}, alpha={self.lora_alpha})"

    def forward(self, x):
        # call super class forward to get output
        x = super().forward(x)
        # apply LoRA
        x = torch.addmm(x, self.lora_b, self.lora_a)
        return x



if __name__ == "__main__":
    # Example of how to load the models
    # model_vanilla = GPT2LMHeadModel.from_pretrained("gpt2")
    # model_lora = GPT2LMHeadModelLora.from_pretrained("gpt2")
    model_lora_custom = GPT2LMHeadModelLora.from_pretrained("gpt2", lora_rank=8, lora_alpha=16)

    # describe models
    # print(model_vanilla)
    # print(model_lora)
    print(model_lora_custom)

    # Freeze non-LoRA parameters
    model_lora_custom.freeze_non_lora()

    # TorchInfo Summaries
    from torchinfo import summary
    # summary(model_vanilla, depth=7)
    # summary(model_lora, depth=7)
    summary(model_lora_custom, depth=7)


