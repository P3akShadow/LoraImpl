import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import RobertaTokenizer


class GLUEDataset(Dataset):
    def __init__(self, task_name, split, tokenizer, max_length=128):
        """
        Initialize GLUE dataset

        Args:
            task_name: Name of the GLUE task (e.g., 'cola', 'sst2', 'mrpc', etc.)
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.dataset = load_dataset('glue', task_name)[split]
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

        # Map task names to their corresponding label columns and number of labels
        self.task_to_labels = {
            'cola': ('label', 2),
            'sst2': ('label', 2),
            'mrpc': ('label', 2),
            'qqp': ('label', 2),
            'stsb': ('label', 1),  # Regression task
            'mnli': ('label', 3),
            'qnli': ('label', 2),
            'rte': ('label', 2),
            'wnli': ('label', 2)
        }

        self.label_col, self.num_labels = self.task_to_labels[task_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Handle different task formats
        if self.task_name in ['cola', 'sst2']:
            text = example['sentence']
            encoding = self.tokenizer(text,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')
        elif self.task_name in ['mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
            text = example['sentence1']
            text_pair = example['sentence2']
            encoding = self.tokenizer(text, text_pair,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')
        elif self.task_name == 'stsb':
            text = example['sentence1']
            text_pair = example['sentence2']
            encoding = self.tokenizer(text, text_pair,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')

        # Convert label to tensor
        label = torch.tensor(example[self.label_col])
        if self.task_name == 'stsb':
            label = label.float()

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }


class CollateFunction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        """Creates and encodes a batch for a list of examples depending on the task.
        The tasks provice data in different formats (we focus on sst2, mnli, cola and stsb).

        sst2 example:
        {
            "sentence": "hide new secretions from the parental units",
            "label": 0,
            "idx": 0
        }
        mnli example:
        {
          "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
          "hypothesis": "Product and geography are what make cream skimming work.",
          "label": 1,
          "idx": 0
        }
        cola example:
        {
          "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
          "label": 1,
          "idx": 0
        }
        stsb example:
        {
          "sentence1": "A plane is taking off.",
          "sentence2": "An air plane is taking off.",
          "label": 5.0,
          "idx": 0
        }
        """
        text = []
        text_pair = []
        for example in batch:
            if 'sentence' in example:
                text.append(example['sentence'])
            elif 'sentence1' in example:
                text.append(example['sentence1'])
            elif 'premise' in example:
                text.append(example['premise'])
            if 'sentence2' in example:
                text_pair.append(example['sentence2'])
            elif 'hypothesis' in example:
                text_pair.append(example['hypothesis'])
        if not text_pair:
            text_pair = None

        # Encode text and text pair
        encoding = self.tokenizer(text, text_pair, truncation=True, padding='longest', return_tensors='pt')
        encoding['label'] = torch.tensor([example['label'] for example in batch])
        return encoding


if __name__ == '__main__':
    # The tasks provide data in different formats (we focus on sst2, mnli, cola and stsb).
    tk = RobertaTokenizer.from_pretrained('roberta-base')
    cfn = CollateFunction(tk)

    print('Loading SST-2 dataset example...')
    ds = load_dataset('glue', 'sst2', split='train')
    dl = DataLoader(ds, batch_size=32, collate_fn=cfn)
    print(next(iter(dl)))

    print('Loading MNLI dataset example...')
    ds = load_dataset('glue', 'mnli', split='train')
    dl = DataLoader(ds, batch_size=32, collate_fn=cfn)
    print(next(iter(dl)))

    print('Loading CoLA dataset example...')
    ds = load_dataset('glue', 'cola', split='train')
    dl = DataLoader(ds, batch_size=32, collate_fn=cfn)
    print(next(iter(dl)))

    print('Loading STS-B dataset example...')
    ds = load_dataset('glue', 'stsb', split='train')
    dl = DataLoader(ds, batch_size=32, collate_fn=cfn)
    print(next(iter(dl)))



