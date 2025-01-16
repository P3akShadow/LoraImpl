import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from loraimpl.data.nlg import CollateFunction


class GPT2LMHeadModelFinetuning(GPT2LMHeadModel):
    """Extend GPT2LMHeadModel to support finetuning."""

    def __init__(self, config, lora_rank=64, lora_alpha=16):
        super().__init__(config)

        # replace all attention layers with LoRA-extended versions
        for i, block in enumerate(self.transformer.h):
            block.attn = GPT2AttentionFinetuning(self.config)

        self.freeze_non_ft()

    def freeze_non_ft(self):
        for name, param in self.named_parameters():
            if "finetuning" not in name:
                param.requires_grad = False


class GPT2AttentionFinetuning(GPT2Attention):
    """Extend GPT2Attention to support Finetuing."""
    def __init__(self, config):
        super().__init__(config)

        # Initialize Conv1D layers for LoRA
        if self.is_cross_attention:
            self.c_attn = Conv1DFinetuning(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1DFinetuning(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1DFinetuning(3 * self.embed_dim, self.embed_dim)


class Conv1DFinetuning(Conv1D):
    """Extend nn.Conv1D to support Finetuning."""
    def __init__(self, nx, nf):
        super().__init__(nx, nf)
        # hyperparameters
        self.finetuning_matrix = nn.Parameter(torch.zeros(nf, nx))

    def __repr__(self) -> str:
        return f"Conv1DFinetuning (nf={self.nf}, nx={self.nx})"

    def forward(self, x):
        finetuning_weight = self.weight.add(self.finetuning_matrix)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), finetuning_weight)
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
    data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn.validation, batch_size=2)

    model.eval()
    test_input, _ = next(iter(data_loader))
    test_output = model.generate(**test_input)
    decoded = tokenizer.decode(test_output[0], skip_special_tokens=False)
    print(decoded)







