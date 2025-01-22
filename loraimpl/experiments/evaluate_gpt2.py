"""Load and evaluate a local GPT2 model on a dataset."""
import torch
from datasets import load_dataset
from torch.xpu import device
from transformers import GPT2TokenizerFast

from loraimpl.data.nlg import CollateFunction
from loraimpl.models import gpt2_modifications
from loraimpl.utils.helper import evaluate_nlg


def main(name='lora', rank=4, alpha=32):
    # Load the model
    inference_config = {
        'num_beams': 10,
        'no_repeat_ngram_size': 4,
        'length_penalty': 0.9,
        'max_length': 500,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = gpt2_modifications[name]
    model = model_cls.from_pretrained("checkpoint", local_files_only=True, lora_rank=rank, lora_alpha=alpha)
    model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    collate_fn = CollateFunction(tokenizer, split='test')
    dataset = load_dataset("GEM/e2e_nlg", split='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    evaluate_nlg(model, data_loader, tokenizer, device=device, inference_cfg=inference_config)


if __name__ == '__main__':
    main()