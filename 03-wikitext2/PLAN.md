# NL2Bash Transformer Implementation Plan

## Overview
Build a **decoder-only transformer** model to translate natural language descriptions into bash commands, following the NL2Bash dataset. This architecture matches modern LLMs (GPT, LLaMA) and is simpler than encoder-decoder designs. The implementation will support both PyTorch and TorchLite frameworks.

## Completed ✓

### 1. Vocabulary Builder (`vocab.py`)
- [x] `Vocabulary` class with token<->ID mappings
- [x] Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
- [x] Build vocabularies from `all.nl` and `all.cm`
- [x] Save/load vocabulary to disk
- [x] Support for minimum frequency filtering
- [x] **NEW**: `build_shared_vocabulary()` for decoder-only architecture
  - Creates single unified vocabulary with 9,078 tokens (min_freq=2)
  - Combines tokens from both NL and command files
  - Saves to `shared_vocab.txt`

### 2. Tokenizer (`tokenizer.py`)
- [x] `Tokenizer` class for single vocabulary
- [x] `NL2BashTokenizer` for dual vocabularies (NL + command)
- [x] Encode/decode with special tokens
- [x] Batch encoding with padding
- [x] Framework-agnostic (works with numpy/lists)
- [x] **NEW**: `encode_combined()` for decoder-only format
  - Combines NL and command: `[nl_tokens] + <START> + [cmd_tokens] + <END>`
  - Returns combined IDs and NL length (for loss masking)
- [x] **NEW**: `decode_combined()` to split back into NL and command
- [x] Supports both shared vocabulary (decoder-only) and separate vocabularies (encoder-decoder)

### 3. Dataset Loader (`dataset.py`)
- [x] `NL2BashDataset` class - Framework-agnostic dataset
  - Loads parallel NL and command data
  - Uses `encode_combined()` for decoder-only format
  - Returns combined sequences with NL lengths
- [x] `split()` method - Split into train/dev/test (10:1:1 ratio)
  - Train: 10,505 examples
  - Dev: 1,050 examples
  - Test: 1,052 examples
- [x] `collate_fn_decoder()` - Batch collation with padding
  - Creates input_ids, target_ids (shifted by 1)
  - Creates attention_mask (1 for real tokens, 0 for padding)
  - Creates loss_mask (1 for command tokens only, 0 for NL/padding)
- [x] `create_pytorch_dataloader()` - PyTorch DataLoader wrapper
  - Converts to torch tensors
  - Handles batching and shuffling
- [x] `create_torchlite_dataloader()` - TorchLite batch iterator
  - Returns numpy arrays
  - Manual batching implementation

### 4. Transformer Model - PyTorch (`model/transformer_full.py`)
- [x] Positional encoding (sine/cosine)
- [x] `create_causal_mask()` - Lower-triangular mask for autoregressive generation
- [x] `MultiHeadAttention` - Multi-head self-attention with masking
- [x] `FeedForward` - Position-wise feed-forward network
- [x] `DecoderLayer` - Single decoder layer (attention + FFN + residuals + layer norm)
- [x] `TransformerDecoder` - Full decoder-only transformer
  - Token embedding + positional encoding
  - Stack of N decoder layers
  - Output projection to vocabulary
- [x] `generate()` method - Auto-regressive generation with temperature and top-k sampling

**Model Statistics (default config):**
- Parameters: ~9.7M (d_model=256, n_heads=4, n_layers=6)
- Input: Combined sequences [batch_size, seq_len]
- Output: Logits [batch_size, seq_len, vocab_size]

### 5. Transformer Model - TorchLite (`model/transformer_lite.py`)
- [x] Helper functions (softmax, layer_norm, positional encoding, causal mask)
- [x] `Embedding` - Token embedding layer (lookup table)
- [x] `MultiHeadAttention` - Simplified multi-head attention
  - Explicit numpy operations for clarity
  - Causal masking support
- [x] `FeedForward` - Position-wise feed-forward network
- [x] `DecoderLayer` - Single decoder layer (attention + FFN + residuals)
- [x] `TransformerDecoderLite` - Full decoder-only transformer
  - Simpler than PyTorch version for education
  - ~3.4M parameters (d_model=128, n_heads=4, n_layers=3)
- [x] `generate()` - Greedy auto-regressive generation

**TorchLite Model Statistics (default config):**
- Parameters: ~3.4M (d_model=128, n_heads=4, n_layers=3)
- Smaller and simpler than PyTorch version
- Shows all operations explicitly (no hidden magic)
- Educational focus: understand what's happening

## All Components Complete! ✅

**Goal**: Implement decoder-only transformer architecture (GPT-style).

**Architecture**:
```
Input: [NL tokens] + <START> + [Command tokens] + <END>
       "find all files" + <START> + "find . -type f" + <END>
         ↓
      Embedding + Positional Encoding
         ↓
      Decoder Layers (causal self-attention)
         ↓
      Linear → Softmax → Next Token Prediction

Loss computed ONLY on command tokens (after <START>)
```

**Key Differences from Encoder-Decoder:**
- ✅ Single transformer stack (simpler!)
- ✅ Only self-attention (no cross-attention)
- ✅ Causal masking on entire sequence
- ✅ NL and command share same token space
- ⚠️ Loss computed only on command portion

**Input Format:**
```python
# Training example:
# NL: "find all files in current directory"
# CMD: "find . -type f"

# Combined sequence:
input_ids = [nl_tokens] + [START] + [cmd_tokens] + [END]
          = [615, 28, 183, 44, 123, 122] + [2] + [83, 106, 131, 186] + [3]

# Target (shifted by 1):
target_ids = [nl_tokens] + [cmd_tokens] + [END] + [PAD]
           = [615, 28, 183, 44, 123, 122] + [83, 106, 131, 186] + [3] + [0]

# Loss mask (only compute loss on command tokens):
loss_mask = [0, 0, 0, 0, 0, 0] + [1, 1, 1, 1] + [1] + [0]
            \_____NL (ignore)_____/   \____CMD (learn)____/
```

**Components needed**:

#### 4.1 Shared Components (framework-agnostic logic)
```python
def positional_encoding(max_len, d_model):
    """
    Create positional encoding matrix using sin/cos functions.
    Can be implemented with numpy, then converted to framework-specific tensors.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pass

def create_causal_mask(seq_len):
    """
    Create causal (lower-triangular) mask for decoder self-attention.
    Prevents attending to future tokens.

    Returns:
        mask: [seq_len, seq_len] where mask[i,j] = 0 if i >= j else -inf
    """
    pass

def create_padding_mask(sequences, pad_id):
    """
    Create mask to ignore padding tokens in attention.

    Returns:
        mask: [batch_size, seq_len] where 1 = real token, 0 = padding
    """
    pass

def create_loss_mask(sequences, nl_lengths, start_id):
    """
    Create mask for computing loss only on command tokens.

    Args:
        sequences: [batch_size, seq_len]
        nl_lengths: List of NL lengths (before <START> token)
        start_id: ID of <START> token

    Returns:
        mask: [batch_size, seq_len] where 1 = command token, 0 = NL token
    """
    pass
```

#### 4.2 PyTorch Model (`model/transformer_full.py`)
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Supports causal masking for decoder-only architecture.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        pass

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] attention mask
        Returns:
            [batch_size, seq_len, d_model]
        """
        pass

class FeedForward(nn.Module):
    """Position-wise feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        pass

class DecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Multi-head self-attention (with causal mask)
    2. Feed-forward network
    3. Layer normalization and residual connections
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        pass

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] causal + padding mask
        Returns:
            [batch_size, seq_len, d_model]
        """
        pass

class TransformerDecoder(nn.Module):
    """
    Decoder-only transformer (GPT-style).
    Combines NL and command in single sequence.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, dropout=0.1, max_len=128):
        """
        Args:
            vocab_size: Combined vocabulary size (nl_vocab + cm_vocab tokens)
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding (fixed or learned)
        self.pos_encoding = # ... positional encoding

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len] - Combined NL + command token IDs
            mask: [batch_size, seq_len, seq_len] - Causal + padding mask

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.embed(x) + self.pos_encoding[:x.size(1)]

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary
        logits = self.output(x)
        return logits

    def generate(self, nl_tokens, max_len=64, strategy='greedy'):
        """
        Auto-regressive generation for inference.

        Args:
            nl_tokens: [batch_size, nl_len] - NL input token IDs
            max_len: Maximum generation length
            strategy: 'greedy' or 'beam_search'

        Returns:
            generated: [batch_size, generated_len] - Generated command IDs
        """
        batch_size = nl_tokens.size(0)

        # Start with NL tokens + <START>
        seq = torch.cat([nl_tokens,
                        torch.full((batch_size, 1), start_id)], dim=1)

        # Generate tokens one by one
        for _ in range(max_len):
            # Forward pass
            logits = self.forward(seq, mask=create_causal_mask(seq.size(1)))

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to sequence
            seq = torch.cat([seq, next_token], dim=1)

            # Stop if all sequences generated <END>
            if (next_token == end_id).all():
                break

        # Return only the command portion (after <START>)
        return seq[:, nl_tokens.size(1)+1:]  # Skip NL + <START>
```

#### 4.3 TorchLite Model (`model/transformer_lite.py`)
```python
# Similar structure to transformer_full.py but using TorchLite operations
# Key differences:
# - Use torchlite.Parameter instead of nn.Parameter
# - Implement forward pass with manual matrix operations
# - Simplified attention mechanism (may skip some optimizations)
# - Manual gradient computation if needed

class TransformerDecoderLite:
    """TorchLite version of decoder-only transformer."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, dropout=0.1, max_len=128):
        pass

    def forward(self, x, mask=None):
        """Similar to PyTorch version but using TorchLite ops."""
        pass

    def generate(self, nl_tokens, max_len=64):
        """Auto-regressive generation."""
        pass
```

**Key considerations**:
- Start with small model: `d_model=256`, `n_heads=4`, `n_layers=6`
- Use dropout for regularization (PyTorch has `nn.Dropout`, TorchLite may need manual)
- Causal masking is critical (prevent attending to future)
- Greedy decoding for inference (can add beam search later)
- Single vocabulary for both NL and commands (simpler than dual vocab)

---

## Training Pipeline (`main.py`)

Will need to update `main.py` to:
1. Build/load vocabularies
   - **Important**: Use a single shared vocabulary for both NL and commands
   - This simplifies the decoder-only architecture
2. Create tokenizer
3. Load dataset and split into train/dev/test
4. Create dataloaders (PyTorch and TorchLite)
   - Format sequences as: `[nl_tokens] + [START] + [cmd_tokens] + [END]`
   - Create loss masks to ignore NL portion
5. Initialize model (both versions)
6. Training loop:
   - Prepare input: combined NL + command sequence
   - Create causal mask (lower triangular)
   - Forward pass with teacher forcing
   - Calculate cross-entropy loss **only on command tokens** (using loss mask)
   - Backward pass and optimizer step
   - Log loss and perplexity
7. Evaluation:
   - Give model NL + `<START>`, generate until `<END>`
   - Calculate BLEU score (may need external library)
   - Calculate exact match accuracy
8. Save model checkpoints

**Key Difference from Encoder-Decoder:**
```python
# Training step:
# Input:  [NL tokens] + [START] + [CMD tokens (shifted)]
# Target: [NL tokens (ignore)] + [CMD tokens] + [END]
# Mask:   [0, 0, ..., 0] + [1, 1, ..., 1]

loss = criterion(logits, targets)
masked_loss = loss * loss_mask  # Only learn from command portion
loss = masked_loss.sum() / loss_mask.sum()
```

## Model Hyperparameters (Starting Point)

```python
# Model architecture
d_model = 256          # Embedding dimension
n_heads = 4            # Number of attention heads
n_encoder_layers = 3   # Number of encoder layers
n_decoder_layers = 3   # Number of decoder layers
d_ff = 1024           # Feed-forward hidden dimension
dropout = 0.1         # Dropout rate

# Training
batch_size = 32       # Batch size
lr = 0.0001           # Learning rate (Adam optimizer)
n_epochs = 50         # Number of training epochs
warmup_steps = 4000   # Learning rate warmup steps

# Sequence lengths (based on dataset analysis)
max_nl_length = 64    # NL max: 56 tokens (99th: 31) → use 64 (power of 2)
max_cm_length = 64    # CM max: 50 tokens (99th: 24) → use 64
# Note: These lengths ensure NO truncation occurs (all 12,607 examples fit)

# Vocabulary
min_freq = 2          # Minimum token frequency
```

## Dataset Statistics

**Natural Language:**
- Total examples: 12,607
- Max length: 56 tokens
- Mean: 13.2 tokens, Median: 12 tokens
- 95th percentile: 24 tokens, 99th percentile: 31 tokens

**Commands:**
- Total examples: 12,607
- Max length: 50 tokens
- Mean: 8.1 tokens, Median: 7 tokens
- 95th percentile: 17 tokens, 99th percentile: 24 tokens

**Key insight**: All sequences are relatively short. Setting max_length=64 accommodates all sequences without any truncation.

## Expected File Structure

```
03-nl2bash/
├── data/
│   ├── all.nl              # Natural language descriptions
│   ├── all.cm              # Bash commands
│   ├── nl_vocab.txt        # Generated NL vocabulary
│   └── cm_vocab.txt        # Generated command vocabulary
├── model/
│   ├── __init__.py
│   ├── transformer_full.py # PyTorch transformer
│   └── transformer_lite.py # TorchLite transformer
├── vocab.py                # ✓ Vocabulary builder
├── tokenizer.py            # ✓ Tokenizer
├── dataset.py              # TODO: Dataset loader
├── main.py                 # TODO: Training/evaluation script
├── PLAN.md                 # This file
└── README.md
```

### 6. Training Script (`main.py`)
- [x] Command-line argument parsing
- [x] Data loading and splitting
- [x] Model initialization
- [x] `compute_masked_loss()` - Loss computation with masking (command tokens only)
- [x] `train_epoch()` - Training loop with gradient clipping
- [x] `evaluate()` - Evaluation on dev/test sets
- [x] `generate_samples()` - Auto-regressive generation
- [x] `calculate_exact_match()` - Exact match accuracy metric
- [x] Checkpoint saving and loading
- [x] Best model tracking (by dev loss)
- [x] Final test evaluation with sample generation

**Training Features:**
- Masked loss (only learns from command tokens)
- Gradient clipping (max_norm=1.0)
- Adam optimizer with configurable learning rate
- Periodic sample generation during training
- Checkpoint saving every N epochs
- Best model tracking
- Comprehensive logging (loss, perplexity, tokens/sec)

## Ready to Train!

### PyTorch Version (Recommended for Training)
```bash
pip install torch numpy
python main.py --n_epochs 50 --batch_size 32
```

### TorchLite Version (Educational)
```bash
pip install numpy
python model/transformer_lite.py  # Test model
```

**Note**: Both implementations are complete but require dependencies:
- PyTorch version: Requires `torch`
- TorchLite version: Requires `numpy`

The TorchLite version is simpler and shows all operations explicitly, making it perfect for understanding how transformers work. However, for actual training, use the PyTorch version as it's much faster.

## Training the Model

Once PyTorch is installed, train with:
```bash
python main.py --n_epochs 50 --batch_size 32 --lr 0.0001
```

Or with custom hyperparameters:
```bash
python main.py \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --batch_size 64 \
  --n_epochs 100 \
  --lr 0.0001 \
  --output_dir outputs/run1
```

## Why Decoder-Only?

**Advantages:**
- ✅ Simpler architecture (one stack vs. two)
- ✅ Matches modern LLMs (GPT, LLaMA, etc.)
- ✅ Easier to understand and implement
- ✅ Fewer components (no cross-attention)
- ✅ More extensible (can add prompting, few-shot learning, etc.)

**Tradeoffs:**
- NL input uses causal masking (can only attend left-to-right)
- Slightly less intuitive than separate encoder/decoder
- Need to carefully mask loss computation

**For educational purposes**, decoder-only is the better choice as it:
1. Teaches the architecture behind modern LLMs
2. Is simpler to implement and debug
3. Still achieves good results on seq2seq tasks
