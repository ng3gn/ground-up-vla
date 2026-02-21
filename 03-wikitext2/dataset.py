"""
Dataset loader for NL2Bash.

This module loads and prepares the NL2Bash dataset for training.
The base dataset class is framework-agnostic and can be used with both
PyTorch and TorchLite implementations.
"""

import random
from typing import List, Tuple, Dict, Optional


class NL2BashDataset:
    """
    Dataset for NL2Bash parallel data.

    Loads (NL, command) pairs and formats them for decoder-only transformer:
    Format: [nl_tokens] + <START> + [cmd_tokens] + <END>
    """

    def __init__(self, nl_file: str, cm_file: str, tokenizer,
                 max_examples: Optional[int] = None):
        """
        Initialize dataset.

        Args:
            nl_file: Path to natural language file (.nl)
            cm_file: Path to command file (.cm)
            tokenizer: NL2BashTokenizer instance
            max_examples: Maximum number of examples to load (None = all)
        """
        self.tokenizer = tokenizer
        self.examples = []

        # Load parallel data
        print(f"Loading dataset from {nl_file} and {cm_file}...")
        with open(nl_file, 'r', encoding='utf-8') as f_nl, \
             open(cm_file, 'r', encoding='utf-8') as f_cm:

            for i, (nl_line, cm_line) in enumerate(zip(f_nl, f_cm)):
                if max_examples and i >= max_examples:
                    break

                nl_text = nl_line.strip()
                cm_text = cm_line.strip()

                if nl_text and cm_text:  # Skip empty lines
                    # Encode as combined sequence
                    combined_ids, nl_length = tokenizer.encode_combined(
                        nl_text, cm_text
                    )

                    self.examples.append({
                        'nl_text': nl_text,
                        'cm_text': cm_text,
                        'combined_ids': combined_ids,
                        'nl_length': nl_length,
                        'total_length': len(combined_ids)
                    })

        print(f"  Loaded {len(self.examples)} examples")

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get example by index.

        Returns:
            Dictionary with keys:
                - combined_ids: [nl_tokens] + [START] + [cmd_tokens] + [END]
                - nl_length: Number of NL tokens (for loss masking)
                - total_length: Total sequence length
                - nl_text: Original NL text (for reference)
                - cm_text: Original command text (for reference)
        """
        return self.examples[idx]

    def split(self, train_ratio: int = 10, dev_ratio: int = 1,
              test_ratio: int = 1, seed: int = 42) -> Tuple['NL2BashDataset',
                                                              'NL2BashDataset',
                                                              'NL2BashDataset']:
        """
        Split dataset into train, dev, and test sets.

        Args:
            train_ratio: Ratio for training set (default: 10)
            dev_ratio: Ratio for dev set (default: 1)
            test_ratio: Ratio for test set (default: 1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, dev_dataset, test_dataset)
        """
        # Set random seed
        random.seed(seed)

        # Shuffle indices
        indices = list(range(len(self.examples)))
        random.shuffle(indices)

        # Calculate split sizes
        total_ratio = train_ratio + dev_ratio + test_ratio
        n_train = int(len(self.examples) * train_ratio / total_ratio)
        n_dev = int(len(self.examples) * dev_ratio / total_ratio)

        # Split indices
        train_indices = indices[:n_train]
        dev_indices = indices[n_train:n_train + n_dev]
        test_indices = indices[n_train + n_dev:]

        # Create sub-datasets
        train_dataset = NL2BashDataset.__new__(NL2BashDataset)
        train_dataset.tokenizer = self.tokenizer
        train_dataset.examples = [self.examples[i] for i in train_indices]

        dev_dataset = NL2BashDataset.__new__(NL2BashDataset)
        dev_dataset.tokenizer = self.tokenizer
        dev_dataset.examples = [self.examples[i] for i in dev_indices]

        test_dataset = NL2BashDataset.__new__(NL2BashDataset)
        test_dataset.tokenizer = self.tokenizer
        test_dataset.examples = [self.examples[i] for i in test_indices]

        print(f"\nDataset split:")
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Dev:   {len(dev_dataset)} examples")
        print(f"  Test:  {len(test_dataset)} examples")

        return train_dataset, dev_dataset, test_dataset


def collate_fn_decoder(batch: List[dict], pad_id: int) -> dict:
    """
    Collate function for batching decoder-only sequences.

    Pads sequences to the same length and creates masks.

    Args:
        batch: List of examples from dataset
        pad_id: ID of padding token

    Returns:
        Dictionary with:
            - input_ids: [batch_size, max_len] - Input token IDs (padded)
            - target_ids: [batch_size, max_len] - Target token IDs (shifted by 1)
            - attention_mask: [batch_size, max_len] - 1 for real tokens, 0 for padding
            - loss_mask: [batch_size, max_len] - 1 for command tokens, 0 for NL/padding
            - nl_lengths: [batch_size] - NL lengths for each example
            - lengths: [batch_size] - Total lengths before padding
    """
    batch_size = len(batch)

    # Get max length in batch
    max_len = max(ex['total_length'] for ex in batch)

    # Initialize arrays
    input_ids = []
    target_ids = []
    attention_mask = []
    loss_mask = []
    nl_lengths = []
    lengths = []

    for ex in batch:
        seq = ex['combined_ids']
        nl_len = ex['nl_length']
        seq_len = len(seq)

        # Input: full sequence
        input_seq = seq[:]

        # Target: sequence shifted by 1
        # Input:  [nl] [START] [cmd] [END] [PAD]
        # Target: [nl] [cmd]   [END] [PAD] [PAD]
        target_seq = seq[1:] + [pad_id]

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = [1] * seq_len

        # Loss mask: 0 for NL tokens, 1 for command tokens (including END)
        # We only want to compute loss on the command generation
        loss_m = [0] * nl_len + [0] + [1] * (seq_len - nl_len - 1)
        # Pattern: [0...0] + [0] + [1...1]
        #          \_NL_/    START  \CMD+END/

        # Pad sequences
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_seq = input_seq + [pad_id] * pad_len
            target_seq = target_seq + [pad_id] * pad_len
            attn_mask = attn_mask + [0] * pad_len
            loss_m = loss_m + [0] * pad_len

        input_ids.append(input_seq)
        target_ids.append(target_seq)
        attention_mask.append(attn_mask)
        loss_mask.append(loss_m)
        nl_lengths.append(nl_len)
        lengths.append(seq_len)

    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'nl_lengths': nl_lengths,
        'lengths': lengths,
        'batch_size': batch_size,
        'max_len': max_len
    }


# PyTorch-specific implementation
try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class PyTorchNL2BashDataset(Dataset):
        """PyTorch wrapper for NL2BashDataset."""

        def __init__(self, base_dataset: NL2BashDataset):
            """
            Initialize PyTorch dataset.

            Args:
                base_dataset: NL2BashDataset instance
            """
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            return self.base_dataset[idx]

    def create_pytorch_dataloader(dataset: NL2BashDataset,
                                   batch_size: int,
                                   shuffle: bool = True,
                                   pad_id: int = 0) -> DataLoader:
        """
        Create PyTorch DataLoader for NL2Bash dataset.

        Args:
            dataset: NL2BashDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            pad_id: Padding token ID

        Returns:
            PyTorch DataLoader
        """
        pytorch_dataset = PyTorchNL2BashDataset(dataset)

        def collate(batch):
            batch_dict = collate_fn_decoder(batch, pad_id)
            # Convert to PyTorch tensors
            return {
                'input_ids': torch.LongTensor(batch_dict['input_ids']),
                'target_ids': torch.LongTensor(batch_dict['target_ids']),
                'attention_mask': torch.FloatTensor(batch_dict['attention_mask']),
                'loss_mask': torch.FloatTensor(batch_dict['loss_mask']),
                'nl_lengths': batch_dict['nl_lengths'],
                'lengths': batch_dict['lengths']
            }

        return DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate
        )

    PYTORCH_AVAILABLE = True

except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. PyTorch dataloader will not work.")


# TorchLite-specific implementation
def create_torchlite_dataloader(dataset: NL2BashDataset,
                                 batch_size: int,
                                 shuffle: bool = True,
                                 pad_id: int = 0):
    """
    Create TorchLite-compatible batch iterator.

    Args:
        dataset: NL2BashDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        pad_id: Padding token ID

    Yields:
        Dictionary with batch data (as numpy arrays or lists)
    """
    import numpy as np

    indices = list(range(len(dataset)))

    while True:  # Infinite iterator (like PyTorch DataLoader)
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            batch_dict = collate_fn_decoder(batch, pad_id)

            # Convert to numpy arrays for TorchLite
            yield {
                'input_ids': np.array(batch_dict['input_ids'], dtype=np.int64),
                'target_ids': np.array(batch_dict['target_ids'], dtype=np.int64),
                'attention_mask': np.array(batch_dict['attention_mask'], dtype=np.float32),
                'loss_mask': np.array(batch_dict['loss_mask'], dtype=np.float32),
                'nl_lengths': batch_dict['nl_lengths'],
                'lengths': batch_dict['lengths']
            }


if __name__ == '__main__':
    # Example usage
    import os
    from vocab import Vocabulary
    from tokenizer import NL2BashTokenizer

    print("=" * 60)
    print("Testing NL2Bash Dataset Loader")
    print("=" * 60)

    # Load shared vocabulary
    data_dir = 'data'
    vocab = Vocabulary.load(os.path.join(data_dir, 'shared_vocab.txt'))
    tokenizer = NL2BashTokenizer(vocab)

    # Load dataset
    nl_file = os.path.join(data_dir, 'all.nl')
    cm_file = os.path.join(data_dir, 'all.cm')

    dataset = NL2BashDataset(nl_file, cm_file, tokenizer)

    print(f"\nTotal examples: {len(dataset)}")

    # Show first example
    print("\nFirst example:")
    ex = dataset[0]
    print(f"  NL: {ex['nl_text']}")
    print(f"  CM: {ex['cm_text']}")
    print(f"  Combined IDs: {ex['combined_ids']}")
    print(f"  NL length: {ex['nl_length']}")
    print(f"  Total length: {ex['total_length']}")

    # Split dataset
    train_dataset, dev_dataset, test_dataset = dataset.split(
        train_ratio=10, dev_ratio=1, test_ratio=1
    )

    # Test collation
    print("\n" + "=" * 60)
    print("Testing Batch Collation")
    print("=" * 60)

    batch = [train_dataset[i] for i in range(3)]
    batch_dict = collate_fn_decoder(batch, vocab.pad_id)

    print(f"\nBatch size: {batch_dict['batch_size']}")
    print(f"Max length: {batch_dict['max_len']}")
    print(f"NL lengths: {batch_dict['nl_lengths']}")
    print(f"Sequence lengths: {batch_dict['lengths']}")

    print("\nFirst sequence in batch:")
    print(f"  Input IDs:  {batch_dict['input_ids'][0]}")
    print(f"  Target IDs: {batch_dict['target_ids'][0]}")
    print(f"  Attn mask:  {batch_dict['attention_mask'][0]}")
    print(f"  Loss mask:  {batch_dict['loss_mask'][0]}")

    # Test PyTorch dataloader if available
    if PYTORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing PyTorch DataLoader")
        print("=" * 60)

        dataloader = create_pytorch_dataloader(
            train_dataset, batch_size=4, shuffle=False, pad_id=vocab.pad_id
        )

        batch = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  target_ids: {batch['target_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  loss_mask: {batch['loss_mask'].shape}")
        print(f"\nFirst example loss mask:")
        print(f"  {batch['loss_mask'][0].tolist()}")
        print(f"  (0s for NL + START, 1s for CMD + END)")

    print("\n" + "=" * 60)
    print("Dataset loader ready!")
    print("=" * 60)
