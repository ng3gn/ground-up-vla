# NL2Bash Transformer

The goal of this module is to build a transformer-based seq2seq model that
translates natural language descriptions into bash commands, using the NL2Bash
dataset.

This is an educational implementation that prioritizes simplicity and
understanding over performance. The pipeline supports both **PyTorch** and
**TorchLite** frameworks.

## Dataset

The [NL2Bash dataset](https://github.com/TellinaTool/nl2bash) contains 12k
one-line Linux shell commands and their natural language descriptions:
- `data/all.nl` - Natural language descriptions
- `data/all.cm` - Corresponding bash commands

## Code

### Components

1. **Vocabulary Builder** (`vocab.py`)
   - Builds shared word-level vocabulary (9,078 tokens)
   - Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
   - Minimum frequency filtering (min_freq=2)
   - Save/load vocabulary files

2. **Tokenizer** (`tokenizer.py`)
   - Framework-agnostic tokenization
   - Encode/decode text to token IDs
   - Combined sequence encoding for decoder-only
   - Batch processing with padding (max_len=64)

3. **Dataset Loader** (`dataset.py`)
   - Loads 12,607 NL-command pairs
   - Splits into train/dev/test (10:1:1)
   - Batch collation with attention and loss masks
   - PyTorch DataLoader wrapper

4. **Transformer Model - PyTorch** (`model/transformer_full.py`)
   - Decoder-only architecture (~9.7M parameters)
   - Multi-head self-attention with causal masking
   - 6 decoder layers with residual connections
   - Auto-regressive generation with temperature/top-k

5. **Transformer Model - TorchLite** (`model/transformer_lite.py`)
   - Simplified decoder-only (~3.4M parameters)
   - Educational implementation showing all operations
   - Explicit numpy operations (no hidden magic)
   - Greedy auto-regressive generation

6. **Training Script** (`main.py`)
   - Complete training loop with masked loss
   - Evaluation and sample generation
   - Checkpoint management
   - Exact match accuracy metric

### Training

Choose your framework:
- **PyTorch** (recommended): Fast, full-featured
- **TorchLite** (educational): Simple, explicit

See `PLAN.md` for complete details.

## Quick Start

### 1. Build Vocabulary

```bash
python vocab.py
```

This will:
- Read `data/all.nl` and `data/all.cm`
- Build shared vocabulary with min_freq=2
- Save to `data/shared_vocab.txt`

### 2. Test Components

```bash
# Test tokenizer
python tokenizer.py

# Test dataset loader
python dataset.py
```

### 3. Install PyTorch (if not already installed)

```bash
pip install torch
```

### 4. Train the Model

```bash
# Default training (50 epochs, batch_size=32)
python main.py

# Custom hyperparameters
python main.py \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --batch_size 64 \
  --n_epochs 100 \
  --lr 0.0001
```

### 5. Monitor Training

The script will:
- Log training progress every 100 batches
- Generate samples every 5 epochs
- Save checkpoints every 5 epochs
- Track best model by dev loss
- Evaluate on test set at the end

Output files in `outputs/`:
- `best_model.pt` - Best model checkpoint
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `results.json` - Final test results

## Architecture

We're building a **decoder-only transformer** model (GPT-style):
- **Single transformer stack**: Simpler than encoder-decoder
- **Causal self-attention**: Attends only to previous tokens
- **Combined input**: NL and command tokens in one sequence
- **Modern approach**: Matches current LLM architectures (GPT, LLaMA)
- **Positional Encoding**: Sine/cosine positional embeddings

**Input format**: `[NL tokens] + <START> + [Command tokens] + <END>`

**Why decoder-only?**
- Simpler to implement (one stack vs. two)
- Matches state-of-the-art LLMs
- Better for educational purposes
- Still effective for seq2seq tasks

Both PyTorch and TorchLite implementations will share the same vocabulary, tokenizer, and dataset components.
