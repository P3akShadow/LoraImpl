from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from yaml import full_load


class NLGDataset(Dataset):
    """
    Simple dataset for E2E NLG,
    with a tokenization approach that partially masks the 'input' portion.
    """
    def __init__(self, model_name, split='train', max_length=128):
        self.data = load_dataset('GEM/e2e_nlg', trust_remote_code=True)
        self.data = self.data[split]
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

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


class CollateFunction:
    def __init__(self, tokenizer, split='train', label_mask_id=-100):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.split = split
        self.label_mask_id = label_mask_id

    def __call__(self, batch):
        if self.split == 'train':
            return self.train(batch)
        else:
            return self.evaluation(batch)

    def train(self, batch):
        full_text = [entry['meaning_representation'] + entry['target'] for entry in batch]
        input_text = [entry['meaning_representation'] for entry in batch]

        # Tokenize inputs and targets
        inputs_encoded = self.tokenizer(
            text=full_text,
            text_target=full_text,  # causal language modeling -> labels will be shifted internally
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        # Mask labels of input portion
        input_length = [len(self.tokenizer.tokenize(entry)) for entry in input_text]
        for i, length in enumerate(input_length):
            inputs_encoded['labels'][i, :length] = self.label_mask_id


        return inputs_encoded

    def evaluation(self, batch):
        inputs = [entry['meaning_representation'] for entry in batch]
        targets = [entry['target'] for entry in batch]

        # Tokenize inputs and targets
        inputs_encoded = self.tokenizer(
            text=inputs,
            text_target=targets,
            return_tensors='pt',
            padding=True,
            truncation=True,
            padding_side='left',
        )
        # get the absolute position_ids
        inputs_encoded['position_ids'] = (inputs_encoded['attention_mask'].cumsum(dim=1) - 1).clamp(min=0)
        references = [entry['references'] for entry in batch if 'references' in entry]
        return inputs_encoded, references


# debugging and testing
if __name__ == '__main__':
    import torch
    s = 'train'
    ds1 = load_dataset('GEM/e2e_nlg', split=s)
    ds2 = NLGDataset("gpt2", split=s)
    cfn = CollateFunction(GPT2TokenizerFast.from_pretrained("gpt2"))
    dl1 = torch.utils.data.DataLoader(ds1, batch_size=5, collate_fn=cfn)
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=5)
    for b1, b2 in zip(dl1, dl2):
        print(b1)
        print(b2)

