# PyTorch Dataset class for Legal Text Decoder
# Handles loading and tokenizing Hungarian legal text data.

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config


class LegalTextDataset(Dataset):
    """Dataset for legal text understandability classification."""

    def __init__(
        self,
        split: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = config.MAX_LENGTH,
        data_dir: str = config.PROCESSED_DATA_DIR
    ):
        """
        Initialize the dataset.

        Args:
            split: One of 'train', 'val', or 'test'
            tokenizer: HuggingFace tokenizer (if None, will load default)
            max_length: Maximum token length for truncation
            data_dir: Directory containing processed JSON files
        """
        self.split = split
        self.max_length = max_length

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        # Load data
        filepath = Path(data_dir) / f"{split}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.texts = [item['text'] for item in data]
        # Convert 1-5 labels to 0-4 for PyTorch (CrossEntropyLoss expects 0-indexed)
        self.labels = [item['label'] - 1 for item in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load tokenizer once and share
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create datasets
    train_dataset = LegalTextDataset('train', tokenizer=tokenizer)
    val_dataset = LegalTextDataset('val', tokenizer=tokenizer)
    test_dataset = LegalTextDataset('test', tokenizer=tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing LegalTextDataset...")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    dataset = LegalTextDataset('train', tokenizer=tokenizer)

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label (0-indexed): {sample['label'].item()}")

    # Test DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['label'].shape}")
