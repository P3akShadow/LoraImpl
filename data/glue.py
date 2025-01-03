from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


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
