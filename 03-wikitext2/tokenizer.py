"""
Tokenizer for NL2Bash.

This module provides tokenization functionality. It is framework-agnostic
and can be used with both PyTorch and TorchLite implementations.

Dataset-specific constants:
    Based on NL2Bash dataset analysis:
    - NL max length: 56 tokens (99th percentile: 31)
    - CM max length: 50 tokens (99th percentile: 24)
    - Safe max lengths set to accommodate all sequences
"""

from typing import List, Tuple, Optional
from vocab import Vocabulary

# Safe max lengths for NL2Bash dataset (covers all 12,607 examples)
MAX_NL_LENGTH = 64   # Actual max is 56, use 64 (power of 2)
MAX_CM_LENGTH = 64   # Actual max is 50, use 64 for consistency


class Tokenizer:
    """
    Tokenizer that converts text to token IDs and back.

    This is a simple word-level tokenizer that splits on whitespace.
    """

    def __init__(self, vocab: Vocabulary):
        """
        Initialize tokenizer with a vocabulary.

        Args:
            vocab: Vocabulary object for token<->ID mapping
        """
        self.vocab = vocab

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        # Simple word-level tokenization (split on whitespace)
        return text.strip().split()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add START and END tokens

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = self.vocab.get_ids(tokens)

        if add_special_tokens:
            token_ids = [self.vocab.start_id] + token_ids + [self.vocab.end_id]

        return token_ids

    def decode(self, token_ids: List[int],
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens (PAD, START, END)

        Returns:
            Decoded text string
        """
        tokens = self.vocab.get_tokens(token_ids)

        if skip_special_tokens:
            special_tokens = {
                self.vocab.PAD_TOKEN,
                self.vocab.START_TOKEN,
                self.vocab.END_TOKEN
            }
            tokens = [t for t in tokens if t not in special_tokens]

        return ' '.join(tokens)

    def encode_batch(self, texts: List[str],
                     add_special_tokens: bool = True,
                     max_length: Optional[int] = None,
                     padding: bool = True) -> Tuple[List[List[int]], List[int]]:
        """
        Encode a batch of texts with optional padding.

        Args:
            texts: List of input text strings
            add_special_tokens: Whether to add START and END tokens
            max_length: Maximum sequence length (None = use longest in batch)
            padding: Whether to pad sequences to max_length

        Returns:
            Tuple of (encoded_ids, lengths) where:
                - encoded_ids: List of token ID lists (padded if padding=True)
                - lengths: List of original sequence lengths (before padding)

        Note:
            For NL2Bash dataset, all sequences fit within MAX_NL_LENGTH (64) and
            MAX_CM_LENGTH (64), so truncation will not occur if using safe defaults.
        """
        # Encode all texts
        encoded = [self.encode(text, add_special_tokens) for text in texts]
        lengths = [len(seq) for seq in encoded]

        if padding:
            # Determine max length
            if max_length is None:
                max_length = max(lengths)

            # Pad sequences
            padded = []
            for seq in encoded:
                if len(seq) < max_length:
                    # Pad with PAD token
                    seq = seq + [self.vocab.pad_id] * (max_length - len(seq))
                elif len(seq) > max_length:
                    # Truncate if too long (won't happen with safe max_length)
                    seq = seq[:max_length]
                padded.append(seq)

            return padded, lengths
        else:
            return encoded, lengths

    def decode_batch(self, batch_ids: List[List[int]],
                     skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            batch_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]


class NL2BashTokenizer:
    """
    Tokenizer for NL2Bash that handles both NL and command sequences.

    Supports two modes:
    1. Shared vocabulary (decoder-only): One vocabulary for both NL and commands
    2. Separate vocabularies (encoder-decoder): Different vocabularies for NL and commands
    """

    def __init__(self, nl_vocab: Vocabulary, cm_vocab: Vocabulary = None):
        """
        Initialize with vocabulary/vocabularies.

        Args:
            nl_vocab: Vocabulary for natural language (or shared vocabulary if cm_vocab is None)
            cm_vocab: Vocabulary for bash commands (optional, for encoder-decoder models)
        """
        if cm_vocab is None:
            # Shared vocabulary mode (decoder-only)
            self.shared_mode = True
            self.nl_tokenizer = Tokenizer(nl_vocab)
            self.cm_tokenizer = self.nl_tokenizer  # Same tokenizer for both
            self.vocab = nl_vocab
            self.nl_vocab = nl_vocab
            self.cm_vocab = nl_vocab
        else:
            # Separate vocabulary mode (encoder-decoder)
            self.shared_mode = False
            self.nl_tokenizer = Tokenizer(nl_vocab)
            self.cm_tokenizer = Tokenizer(cm_vocab)
            self.nl_vocab = nl_vocab
            self.cm_vocab = cm_vocab
            self.vocab = None  # No single shared vocab in this mode

    def encode_nl(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode natural language text."""
        return self.nl_tokenizer.encode(text, add_special_tokens)

    def encode_cm(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode bash command text."""
        return self.cm_tokenizer.encode(text, add_special_tokens)

    def decode_nl(self, token_ids: List[int],
                  skip_special_tokens: bool = True) -> str:
        """Decode natural language token IDs."""
        return self.nl_tokenizer.decode(token_ids, skip_special_tokens)

    def decode_cm(self, token_ids: List[int],
                  skip_special_tokens: bool = True) -> str:
        """Decode bash command token IDs."""
        return self.cm_tokenizer.decode(token_ids, skip_special_tokens)

    def encode_pair(self, nl_text: str, cm_text: str,
                    add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
        """
        Encode a (NL, command) pair.

        Args:
            nl_text: Natural language text
            cm_text: Bash command text
            add_special_tokens: Whether to add START and END tokens

        Returns:
            Tuple of (nl_ids, cm_ids)
        """
        nl_ids = self.encode_nl(nl_text, add_special_tokens)
        cm_ids = self.encode_cm(cm_text, add_special_tokens)
        return nl_ids, cm_ids

    def encode_combined(self, nl_text: str, cm_text: str) -> Tuple[List[int], int]:
        """
        Encode NL and command as a single combined sequence for decoder-only models.

        Format: [nl_tokens] + [START] + [cm_tokens] + [END]

        Args:
            nl_text: Natural language text
            cm_text: Bash command text

        Returns:
            Tuple of (combined_ids, nl_length) where:
                - combined_ids: [nl_tokens] + [START] + [cm_tokens] + [END]
                - nl_length: Number of NL tokens (for creating loss mask)
        """
        # Encode NL without special tokens
        nl_ids = self.encode_nl(nl_text, add_special_tokens=False)

        # Encode command without special tokens
        cm_ids = self.encode_cm(cm_text, add_special_tokens=False)

        # Combine: [nl_tokens] + [START] + [cm_tokens] + [END]
        combined_ids = nl_ids + [self.vocab.start_id] + cm_ids + [self.vocab.end_id]

        return combined_ids, len(nl_ids)

    def decode_combined(self, combined_ids: List[int],
                       nl_length: int,
                       skip_special_tokens: bool = True) -> Tuple[str, str]:
        """
        Decode a combined sequence back into NL and command.

        Args:
            combined_ids: Combined token IDs
            nl_length: Number of NL tokens at the beginning
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Tuple of (nl_text, cm_text)
        """
        # Split back into NL and command portions
        nl_ids = combined_ids[:nl_length]
        # Skip START token, take until END or end of sequence
        cm_start = nl_length + 1

        # Find END token
        try:
            end_idx = combined_ids[cm_start:].index(self.vocab.end_id)
            cm_ids = combined_ids[cm_start:cm_start + end_idx]
        except ValueError:
            # No END token found, take rest of sequence
            cm_ids = combined_ids[cm_start:]

        nl_text = self.decode_nl(nl_ids, skip_special_tokens)
        cm_text = self.decode_cm(cm_ids, skip_special_tokens)

        return nl_text, cm_text

    def encode_batch_pairs(self, nl_texts: List[str], cm_texts: List[str],
                          add_special_tokens: bool = True,
                          max_nl_length: Optional[int] = MAX_NL_LENGTH,
                          max_cm_length: Optional[int] = MAX_CM_LENGTH,
                          padding: bool = True) -> dict:
        """
        Encode a batch of (NL, command) pairs.

        Args:
            nl_texts: List of natural language texts
            cm_texts: List of bash command texts
            add_special_tokens: Whether to add START and END tokens
            max_nl_length: Maximum NL sequence length (default: MAX_NL_LENGTH=64)
            max_cm_length: Maximum command sequence length (default: MAX_CM_LENGTH=64)
            padding: Whether to pad sequences

        Returns:
            Dictionary with keys:
                - 'nl_ids': Encoded NL sequences
                - 'nl_lengths': Original NL sequence lengths
                - 'cm_ids': Encoded command sequences
                - 'cm_lengths': Original command sequence lengths

        Note:
            Default max lengths are set to safely accommodate all sequences in
            the NL2Bash dataset without truncation.
        """
        nl_ids, nl_lengths = self.nl_tokenizer.encode_batch(
            nl_texts, add_special_tokens, max_nl_length, padding
        )
        cm_ids, cm_lengths = self.cm_tokenizer.encode_batch(
            cm_texts, add_special_tokens, max_cm_length, padding
        )

        return {
            'nl_ids': nl_ids,
            'nl_lengths': nl_lengths,
            'cm_ids': cm_ids,
            'cm_lengths': cm_lengths
        }


if __name__ == '__main__':
    # Example usage
    import os
    from vocab import Vocabulary

    data_dir = 'data'

    # Load shared vocabulary (for decoder-only model)
    print("=" * 60)
    print("Testing Shared Vocabulary Tokenizer (Decoder-Only)")
    print("=" * 60)
    shared_vocab = Vocabulary.load(os.path.join(data_dir, 'shared_vocab.txt'))
    tokenizer = NL2BashTokenizer(shared_vocab)

    # Example NL and command
    nl_text = "find all files in current directory"
    cm_text = "find . -type f"

    print(f"\nOriginal:")
    print(f"  NL: {nl_text}")
    print(f"  CM: {cm_text}")
    print()

    # Test combined encoding (decoder-only format)
    combined_ids, nl_length = tokenizer.encode_combined(nl_text, cm_text)
    print(f"Combined Encoding (Decoder-Only Format):")
    print(f"  Format: [NL tokens] + <START> + [CMD tokens] + <END>")
    print(f"  Combined IDs: {combined_ids}")
    print(f"  NL length: {nl_length}")
    print(f"  Total length: {len(combined_ids)}")
    print()

    # Decode combined
    decoded_nl, decoded_cm = tokenizer.decode_combined(combined_ids, nl_length)
    print(f"Decoded from Combined:")
    print(f"  NL: {decoded_nl}")
    print(f"  CM: {decoded_cm}")
    print()

    # Show token breakdown
    print(f"Token Breakdown:")
    print(f"  NL tokens:    {combined_ids[:nl_length]}")
    print(f"  <START>:      [{combined_ids[nl_length]}]")
    print(f"  CMD tokens:   {combined_ids[nl_length+1:-1]}")
    print(f"  <END>:        [{combined_ids[-1]}]")
    print()

    # Batch encoding example with combined sequences
    nl_batch = [
        "find all files in current directory",
        "list all python files",
        "count lines in file"
    ]
    cm_batch = [
        "find . -type f",
        "ls *.py",
        "wc -l filename"
    ]

    print("=" * 60)
    print("Batch Combined Encoding Example:")
    print("=" * 60)

    # Encode each pair as combined sequence
    combined_batch = []
    nl_lengths = []
    for nl, cm in zip(nl_batch, cm_batch):
        combined_ids, nl_len = tokenizer.encode_combined(nl, cm)
        combined_batch.append(combined_ids)
        nl_lengths.append(nl_len)

    print(f"  Batch size: {len(combined_batch)}")
    print(f"  Sequence lengths: {[len(seq) for seq in combined_batch]}")
    print(f"  NL lengths: {nl_lengths}")
    print()

    # Show first example breakdown
    print(f"Example 1 breakdown:")
    print(f"  Total tokens: {len(combined_batch[0])}")
    print(f"  NL portion: {combined_batch[0][:nl_lengths[0]]}")
    print(f"  START: [{combined_batch[0][nl_lengths[0]]}]")
    print(f"  CMD portion: {combined_batch[0][nl_lengths[0]+1:]}")
    print()
    print("Note: In training, loss is computed only on CMD portion + END token")
