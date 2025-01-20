from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class NLGDataset(Dataset):
    """
    Simple dataset for E2E NLG,
    with a tokenization approach that partially masks the 'input' portion.
    """
    def __init__(self, split='train', max_length=128, name='GEM/e2e_nlg'):
        self.data = load_dataset(name, trust_remote_code=True)
        self.data = self.data[split]
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length

        self.inputs = []
        self.targets = []

        for item in self.data:
            # simpler approach:
            input_text = item["meaning_representation"]
            target_text = item["target"]
            if not target_text.endswith('.'):
                target_text += '.'

            combined = f"{input_text}\n{target_text}"
            self.inputs.append(input_text)
            self.targets.append(target_text)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        # Full combined
        full_text = input_text + "\n" + target_text

        # tokenize full
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # We'll partially mask the input portion
        labels = encoding["input_ids"].clone()

        # The input portion is everything up to the newline + 1
        # (just a naive approach - adjust if you want more precise masking)
        input_length = len(self.tokenizer.encode(input_text)) + 1
        labels[0, :input_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


