import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Optional, Tuple, Union


class GPT2LMHeadModelLora(GPT2LMHeadModel):
    def __init__(self, config, lora_rank=64, lora_alpha=128):
        super().__init__(config)
        # replace all attention layers with LoRA-extended versions
        for i, block in enumerate(self.transformer.h):
            block.attn = GPT2AttentionLora(self.config, lora_rank=lora_rank, lora_alpha=lora_alpha)



class GPT2AttentionLora(GPT2Attention):
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
    """Extend nn.Conv1D to support LoRA.
    """
    def __init__(self, nx, nf, lora_rank=64, lora_alpha=128):
        super().__init__(nx, nf)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.weight.requires_grad = False
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
    model_vanilla = GPT2LMHeadModel.from_pretrained("gpt2")
    model_lora = GPT2LMHeadModelLora.from_pretrained("gpt2")
    model_lora_custom = GPT2LMHeadModelLora.from_pretrained("gpt2", lora_rank=8, lora_alpha=16)

    # describe models
    print(model_vanilla)
    print(model_lora)
    print(model_lora_custom)

    # TorchInfo Summaries
    from torchinfo import summary
    summary(model_vanilla)
    summary(model_lora)
    summary(model_lora_custom)

