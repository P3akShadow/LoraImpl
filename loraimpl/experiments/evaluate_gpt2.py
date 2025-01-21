"""Load and evaluate a local GPT2 model on a dataset."""
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from loraimpl.data.nlg import CollateFunction
from loraimpl.models import gpt2_modifications
from loraimpl.utils.helper import evaluate_nlg


def main(name='lora', rank=4, alpha=32):
    # Load the model
    model_cls = gpt2_modifications[name]
    model = model_cls.from_pretrained("checkpoint", local_files_only=True, lora_rank=rank, lora_alpha=alpha)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    collate_fn = CollateFunction(tokenizer, split='test')
    dataset = load_dataset("GEM/e2e_nlg", split='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    evaluate_nlg(model, data_loader, tokenizer, device=torch.device('cpu'), inference_cfg={})


if __name__ == '__main__':
    main()