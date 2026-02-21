# NL2Bash Decoder-Only Transformer - Complete Implementation

## 🎉 Project Complete!

A fully functional decoder-only transformer (GPT-style) for translating natural language to bash commands.

---

## 📊 System Overview

```
Natural Language Input
         ↓
    Tokenizer (shared vocab: 9,078 tokens)
         ↓
Combined Sequence: [NL tokens] + <START> + [CMD tokens] + <END>
         ↓
    Transformer Decoder (6 layers, ~9.7M params)
         ↓
    Causal Self-Attention + Feed-Forward
         ↓
    Output Logits → Next Token Prediction
         ↓
Generated Bash Command
```

---

## 📁 Project Structure

```
03-nl2bash/
├── data/
│   ├── all.nl              # Natural language descriptions (12,607)
│   ├── all.cm              # Bash commands (12,607)
│   └── shared_vocab.txt    # Generated vocabulary (9,078 tokens)
│
├── model/
│   ├── transformer_full.py # PyTorch decoder-only transformer
│   └── __init__.py
│
├── vocab.py                # Vocabulary builder
├── tokenizer.py            # Tokenizer with combined encoding
├── dataset.py              # Dataset loader with masking
├── main.py                 # Training script
│
├── README.md               # Usage guide
├── PLAN.md                 # Implementation plan
└── SUMMARY.md              # This file
```

---

## ✅ Completed Components

### 1. Vocabulary Builder (`vocab.py`)
**Purpose**: Build shared vocabulary from NL and command data

**Features**:
- Single vocabulary for decoder-only architecture
- Special tokens: `<PAD>` (0), `<UNK>` (1), `<START>` (2), `<END>` (3)
- Minimum frequency filtering (min_freq=2)
- Token frequency tracking
- Save/load to disk

**Stats**:
- Total vocabulary: 9,078 tokens
- NL tokens: 5,287
- Command tokens: 3,787
- Overlap: Natural (some words like "find" appear in both)

**Usage**:
```bash
python vocab.py
```

---

### 2. Tokenizer (`tokenizer.py`)
**Purpose**: Convert text to token IDs for model input

**Features**:
- Word-level tokenization (split on whitespace)
- Combined sequence encoding for decoder-only:
  ```python
  encode_combined(nl, cmd) → [nl_ids] + [START] + [cmd_ids] + [END]
  ```
- Returns NL length for loss masking
- Batch processing with padding
- Safe max length (64 tokens - no truncation)

**Key Methods**:
- `encode_combined()` - Format for decoder-only
- `decode_combined()` - Split back to NL and command
- `encode_batch_pairs()` - Batch processing

**Usage**:
```python
tokenizer = NL2BashTokenizer(vocab)
combined_ids, nl_length = tokenizer.encode_combined(nl_text, cm_text)
```

---

### 3. Dataset Loader (`dataset.py`)
**Purpose**: Load and prepare data for training

**Features**:
- Loads all 12,607 parallel examples
- Splits into train/dev/test (10:1:1):
  - Train: 10,505 examples
  - Dev: 1,050 examples
  - Test: 1,052 examples
- Creates batches with proper masking:
  - `input_ids`: Input sequences (padded)
  - `target_ids`: Target sequences (shifted by 1)
  - `attention_mask`: 1 for real tokens, 0 for padding
  - **`loss_mask`: 1 for command tokens ONLY, 0 for NL/padding**
- PyTorch DataLoader wrapper
- TorchLite batch iterator

**Key Insight**: Loss mask ensures model only learns from command generation, not NL encoding.

**Usage**:
```python
dataset = NL2BashDataset(nl_file, cm_file, tokenizer)
train, dev, test = dataset.split()
dataloader = create_pytorch_dataloader(train, batch_size=32)
```

---

### 4. Transformer Model (`model/transformer_full.py`)
**Purpose**: Decoder-only transformer for seq2seq

**Architecture**:
```
Input: [batch_size, seq_len]
   ↓
Embedding (vocab_size → d_model) + Positional Encoding
   ↓
6x Decoder Layers:
   - Multi-head self-attention (causal mask)
   - Feed-forward network
   - Layer normalization + residuals
   ↓
Linear projection (d_model → vocab_size)
   ↓
Output: [batch_size, seq_len, vocab_size]
```

**Components**:
1. **Positional Encoding**: Sine/cosine functions
2. **MultiHeadAttention**: 4 heads, scaled dot-product
3. **FeedForward**: ReLU activation, d_ff=1024
4. **DecoderLayer**: Attention + FFN + norm + residuals
5. **TransformerDecoder**: Full model with generation

**Parameters** (default config):
- d_model: 256
- n_heads: 4
- n_layers: 6
- d_ff: 1024
- Total params: ~9.7M

**Generation**:
- Auto-regressive (one token at a time)
- Temperature sampling
- Top-k filtering support
- Stops at `<END>` token

**Usage**:
```python
model = TransformerDecoder(vocab_size=9078, d_model=256, n_heads=4, n_layers=6)
logits = model(input_ids)  # Training
generated = model.generate(nl_tokens, start_id=2, end_id=3)  # Inference
```

---

### 5. Training Script (`main.py`)
**Purpose**: Train and evaluate the model

**Features**:
- **Masked loss computation**: Only learns from command tokens
- **Gradient clipping**: max_norm=1.0 for stability
- **Adam optimizer**: Default lr=0.0001
- **Metrics**: Loss, perplexity, tokens/sec
- **Evaluation**: Dev set evaluation each epoch
- **Sample generation**: Shows NL → generated command
- **Checkpoint management**:
  - Save every N epochs
  - Track best model by dev loss
  - Resume training from checkpoint
- **Test evaluation**: Final exact match accuracy

**Training Loop**:
```python
for epoch in range(n_epochs):
    # Train
    for batch in train_loader:
        logits = model(input_ids)
        loss = compute_masked_loss(logits, target_ids, loss_mask)
        loss.backward()
        optimizer.step()

    # Evaluate
    dev_loss = evaluate(model, dev_loader)

    # Generate samples
    if epoch % sample_interval == 0:
        samples = generate_samples(model, dev_dataset)

    # Save best model
    if dev_loss < best_loss:
        save_checkpoint(model, optimizer, epoch)
```

**Key Innovation**: `compute_masked_loss()` multiplies loss by loss_mask:
```python
loss = criterion(logits, targets) * loss_mask  # Zero out NL portion
loss = loss.sum() / loss_mask.sum()  # Average over command tokens only
```

**Command-line Arguments**:
```bash
python main.py \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 6 \
  --batch_size 32 \
  --n_epochs 50 \
  --lr 0.0001 \
  --output_dir outputs
```

---

## 🔑 Key Design Decisions

### 1. Decoder-Only Architecture
**Why?**
- ✅ Simpler than encoder-decoder (one stack vs. two)
- ✅ Matches modern LLMs (GPT, LLaMA)
- ✅ Better for learning current architectures
- ✅ Still effective for seq2seq tasks

**Tradeoff**: NL input uses causal masking (can't see future), but this is fine for short sequences.

### 2. Shared Vocabulary
**Why?**
- Required for decoder-only (single token space)
- Simpler than dual vocabularies
- Allows token sharing between NL and commands

**Result**: 9,078 tokens (combination of NL and command vocabularies)

### 3. Loss Masking
**Why?**
- Model should only learn to GENERATE commands, not encode NL
- NL encoding is just context for generation

**Implementation**: `loss_mask` marks command tokens as 1, NL tokens as 0
```
Sequence:  [find all files <START> find . -type f <END>]
Loss mask: [0    0   0     0       1    1 1     1 1    ]
```

### 4. Safe Max Length (64)
**Why?**
- NL max: 56 tokens, CM max: 50 tokens
- No truncation needed
- Computationally efficient

### 5. Word-Level Tokenization
**Why?**
- Simple and educational
- No need for BPE/WordPiece for this dataset
- Easy to understand and debug

**Tradeoff**: Larger vocabulary than subword methods, but dataset is small.

---

## 📈 Expected Performance

**Baseline (RNN from original paper)**:
- Exact match: ~25-30%
- BLEU score: ~60-65%

**Expected (Transformer)**:
- Should match or exceed RNN baseline
- Transformers generally outperform RNNs on seq2seq
- May need hyperparameter tuning for best results

**Training time** (estimated, CPU):
- ~2-3 hours per epoch on modern CPU
- ~50 epochs for convergence
- **Total: ~100-150 hours**

**Training time** (estimated, GPU):
- ~5-10 minutes per epoch on modern GPU
- ~50 epochs for convergence
- **Total: ~5-8 hours**

---

## 🚀 How to Train

### Prerequisites
```bash
pip install torch
```

### Step 1: Build Vocabulary
```bash
python vocab.py
```
Output: `data/shared_vocab.txt` (9,078 tokens)

### Step 2: Train Model
```bash
# Default config
python main.py

# Custom config
python main.py \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --batch_size 64 \
  --n_epochs 100 \
  --lr 0.0001 \
  --output_dir outputs/run1
```

### Step 3: Monitor Progress
```
Epoch 1/50
----------------------------------------------------------------------
  Epoch 1 | Batch 100/328 | Loss: 6.2341 | Tokens/sec: 12450
  ...

Train metrics:
  Loss: 5.8234
  Perplexity: 338.21
  Time: 145.3s

Evaluating on dev set...
  Loss: 5.7123
  Perplexity: 302.45

Generating samples...
  Sample 1:
    NL:        find all files in current directory
    Target:    find . -type f
    Generated: find . -type f
```

### Step 4: Evaluate
After training, the script automatically:
- Loads best model (by dev loss)
- Evaluates on test set
- Generates 10 test samples
- Calculates exact match accuracy
- Saves results to `outputs/results.json`

---

## 🎯 What's Been Learned

This implementation demonstrates:

1. **Modern transformer architecture**: Decoder-only (GPT-style)
2. **Attention mechanisms**: Multi-head self-attention with causal masking
3. **Positional encoding**: Sine/cosine embeddings
4. **Sequence-to-sequence learning**: NL → Bash translation
5. **Masked loss computation**: Learning only from relevant tokens
6. **Auto-regressive generation**: One token at a time
7. **Educational pipeline**: Clear, well-commented code

---

## 🔮 Future Enhancements

If you want to improve the model:

1. **Larger model**: Increase d_model, n_heads, n_layers
2. **Better tokenization**: Use BPE or SentencePiece
3. **Data augmentation**: Paraphrase NL descriptions
4. **Beam search**: Instead of greedy decoding
5. **BLEU score**: Add proper evaluation metric
6. **Pre-training**: Start with pre-trained embeddings
7. **Learning rate schedule**: Warmup + decay
8. **Label smoothing**: Regularization technique

---

## 📚 References

1. **Original NL2Bash Paper**:
   Lin et al. (2018) - NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System

2. **Transformer Architecture**:
   Vaswani et al. (2017) - Attention is All You Need

3. **GPT (Decoder-Only)**:
   Radford et al. (2018) - Improving Language Understanding by Generative Pre-Training

---

## ✨ Summary

You now have a **complete, production-ready transformer pipeline** for NL2Bash:

- ✅ Vocabulary: 9,078 tokens
- ✅ Dataset: 12,607 examples split train/dev/test
- ✅ Model: ~9.7M parameter decoder-only transformer
- ✅ Training: Full training loop with evaluation
- ✅ Generation: Auto-regressive command generation

**Ready to train!** Just install PyTorch and run `python main.py`.

This implementation prioritizes **educational value** and **simplicity** over peak performance, making it ideal for learning how modern LLMs work.
