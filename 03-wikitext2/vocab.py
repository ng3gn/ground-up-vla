"""
Vocabulary builder for NL2Bash.

This module builds vocabulary from the NL2Bash dataset. It is framework-agnostic
and can be used with both PyTorch and TorchLite implementations.
"""

from collections import Counter
from typing import List, Dict, Set
import os


class Vocabulary:
    """
    Vocabulary for mapping tokens to IDs and vice versa.

    Special tokens:
        <PAD>   - Padding token (ID: 0)
        <UNK>   - Unknown token (ID: 1)
        <START> - Start of sequence (ID: 2)
        <END>   - End of sequence (ID: 3)
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'

    def __init__(self):
        """Initialize empty vocabulary."""
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.token_counts: Counter = Counter()

        # Add special tokens first (so they get consistent IDs)
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary with fixed IDs."""
        special_tokens = [
            self.PAD_TOKEN,   # ID: 0
            self.UNK_TOKEN,   # ID: 1
            self.START_TOKEN, # ID: 2
            self.END_TOKEN,   # ID: 3
        ]
        for token in special_tokens:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def add_token(self, token: str):
        """
        Add a token to the vocabulary.

        Args:
            token: Token string to add
        """
        self.token_counts[token] += 1
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def add_tokens(self, tokens: List[str]):
        """
        Add multiple tokens to the vocabulary.

        Args:
            tokens: List of token strings to add
        """
        for token in tokens:
            self.add_token(token)

    def get_id(self, token: str) -> int:
        """
        Get the ID for a token.

        Args:
            token: Token string

        Returns:
            Token ID (returns UNK_TOKEN ID if not found)
        """
        return self.token2id.get(token, self.token2id[self.UNK_TOKEN])

    def get_token(self, idx: int) -> str:
        """
        Get the token for an ID.

        Args:
            idx: Token ID

        Returns:
            Token string (returns UNK_TOKEN if not found)
        """
        return self.id2token.get(idx, self.UNK_TOKEN)

    def get_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert list of tokens to list of IDs.

        Args:
            tokens: List of token strings

        Returns:
            List of token IDs
        """
        return [self.get_id(token) for token in tokens]

    def get_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert list of IDs to list of tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        return [self.get_token(idx) for idx in ids]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        """Get PAD token ID."""
        return self.token2id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """Get UNK token ID."""
        return self.token2id[self.UNK_TOKEN]

    @property
    def start_id(self) -> int:
        """Get START token ID."""
        return self.token2id[self.START_TOKEN]

    @property
    def end_id(self) -> int:
        """Get END token ID."""
        return self.token2id[self.END_TOKEN]

    def save(self, filepath: str):
        """
        Save vocabulary to file.

        Args:
            filepath: Path to save vocabulary
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx in sorted(self.id2token.keys()):
                token = self.id2token[idx]
                count = self.token_counts.get(token, 0)
                f.write(f"{idx}\t{token}\t{count}\n")

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        Load vocabulary from file.

        Args:
            filepath: Path to vocabulary file

        Returns:
            Loaded Vocabulary object
        """
        vocab = cls()
        # Clear the auto-added special tokens, we'll load them from file
        vocab.token2id.clear()
        vocab.id2token.clear()
        vocab.token_counts.clear()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    token = parts[1]
                    count = int(parts[2]) if len(parts) >= 3 else 0

                    vocab.token2id[token] = idx
                    vocab.id2token[idx] = token
                    if count > 0:
                        vocab.token_counts[token] = count

        return vocab


def build_vocabulary_from_file(filepath: str, vocab: Vocabulary,
                                min_freq: int = 1) -> Vocabulary:
    """
    Build vocabulary from a text file.

    Args:
        filepath: Path to text file (one example per line)
        vocab: Vocabulary object to add tokens to
        min_freq: Minimum frequency for a token to be included

    Returns:
        Updated Vocabulary object
    """
    # First pass: count all tokens
    token_counts = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            token_counts.update(tokens)

    # Second pass: add tokens that meet minimum frequency
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab.add_token(token)
            # Update the count to reflect actual corpus frequency
            vocab.token_counts[token] = count

    return vocab


def build_shared_vocabulary(nl_file: str, cm_file: str,
                            min_freq: int = 1,
                            output_dir: str = None) -> Vocabulary:
    """
    Build a single shared vocabulary from both NL and command files.
    This is used for decoder-only transformer architectures.

    Args:
        nl_file: Path to natural language file (.nl)
        cm_file: Path to command file (.cm)
        min_freq: Minimum frequency for a token to be included
        output_dir: Directory to save vocabulary (optional)

    Returns:
        Single Vocabulary object containing tokens from both files
    """
    print("Building shared vocabulary...")

    # Create single vocabulary
    vocab = Vocabulary()

    # Add tokens from NL file
    print(f"  Reading {nl_file}...")
    vocab = build_vocabulary_from_file(nl_file, vocab, min_freq)
    nl_tokens = len(vocab) - 4  # Subtract special tokens

    # Add tokens from command file (adds to existing vocab)
    print(f"  Reading {cm_file}...")
    vocab = build_vocabulary_from_file(cm_file, vocab, min_freq)
    cm_tokens = len(vocab) - 4 - nl_tokens  # New tokens from commands

    print(f"  Total vocabulary size: {len(vocab)} tokens")
    print(f"    Special tokens: 4")
    print(f"    NL tokens: {nl_tokens}")
    print(f"    Command tokens: {cm_tokens}")
    print(f"    Shared tokens: {nl_tokens + cm_tokens - (len(vocab) - 4)}")

    # Save vocabulary if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vocab_path = os.path.join(output_dir, 'shared_vocab.txt')
        vocab.save(vocab_path)
        print(f"  Saved vocabulary to {vocab_path}")

    return vocab


def build_vocabularies(nl_file: str, cm_file: str,
                       min_freq: int = 1,
                       output_dir: str = None) -> tuple:
    """
    Build separate vocabularies for natural language and commands.

    NOTE: This function is kept for backwards compatibility, but
    decoder-only architectures should use build_shared_vocabulary() instead.

    Args:
        nl_file: Path to natural language file (.nl)
        cm_file: Path to command file (.cm)
        min_freq: Minimum frequency for a token to be included
        output_dir: Directory to save vocabularies (optional)

    Returns:
        Tuple of (nl_vocab, cm_vocab)
    """
    print("Building separate vocabularies...")
    print("NOTE: Decoder-only models should use build_shared_vocabulary() instead")

    # Build NL vocabulary
    print(f"  Reading {nl_file}...")
    nl_vocab = Vocabulary()
    nl_vocab = build_vocabulary_from_file(nl_file, nl_vocab, min_freq)
    print(f"  NL vocabulary size: {len(nl_vocab)} tokens")

    # Build command vocabulary
    print(f"  Reading {cm_file}...")
    cm_vocab = Vocabulary()
    cm_vocab = build_vocabulary_from_file(cm_file, cm_vocab, min_freq)
    print(f"  Command vocabulary size: {len(cm_vocab)} tokens")

    # Save vocabularies if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nl_vocab_path = os.path.join(output_dir, 'nl_vocab.txt')
        cm_vocab_path = os.path.join(output_dir, 'cm_vocab.txt')

        nl_vocab.save(nl_vocab_path)
        cm_vocab.save(cm_vocab_path)
        print(f"  Saved vocabularies to {output_dir}/")

    return nl_vocab, cm_vocab


if __name__ == '__main__':
    # Example usage: build shared vocabulary for decoder-only model
    data_dir = 'data'
    nl_file = os.path.join(data_dir, 'all.nl')
    cm_file = os.path.join(data_dir, 'all.cm')

    # Build shared vocabulary with minimum frequency of 2 (to reduce vocab size)
    print("=" * 60)
    print("Building Shared Vocabulary for Decoder-Only Transformer")
    print("=" * 60)

    vocab = build_shared_vocabulary(
        nl_file, cm_file,
        min_freq=2,
        output_dir=data_dir
    )

    # Print some statistics
    print("\n" + "=" * 60)
    print("Vocabulary Statistics:")
    print("=" * 60)
    print(f"  Total vocab size: {len(vocab)} tokens")
    print(f"\nSpecial token IDs:")
    print(f"  PAD: {vocab.pad_id}")
    print(f"  UNK: {vocab.unk_id}")
    print(f"  START: {vocab.start_id}")
    print(f"  END: {vocab.end_id}")

    # Show some example tokens
    print(f"\nExample tokens (IDs 4-23):")
    for i in range(4, min(24, len(vocab))):
        token = vocab.get_token(i)
        count = vocab.token_counts.get(token, 0)
        print(f"  {i}: {token:20s} (count: {count})")

    print("\n" + "=" * 60)
    print("Note: This shared vocabulary is used for decoder-only")
    print("transformer models where NL and commands share token space.")
    print("=" * 60)
