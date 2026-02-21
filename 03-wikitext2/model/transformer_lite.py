"""
Decoder-only Transformer model using TorchLite.

This is a simplified educational implementation that shows how transformers work
using the minimal TorchLite framework. It's more verbose than PyTorch but shows
all the operations explicitly.
"""

import sys
import os
import numpy as np
import math

# Add parent directory to path to import torchlite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from torchlite.tensor import Tensor
from torchlite.nn import Module, Linear, ReLU


# =========================
# HELPER FUNCTIONS
# =========================

def create_positional_encoding(max_len, d_model):
    """
    Create positional encoding matrix using sine and cosine functions.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Numpy array [max_len, d_model]
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def create_causal_mask(seq_len):
    """
    Create causal mask for decoder self-attention.

    Returns:
        Mask [seq_len, seq_len] where mask[i,j] = 0 if i >= j else -inf
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -1e9, 0)
    return mask


# =========================
# EMBEDDING LAYER
# =========================

class Embedding(Module):
    """
    Embedding layer: maps token IDs to dense vectors.

    This is essentially a lookup table.
    """

    def __init__(self, vocab_size, d_model):
        """
        Initialize embedding.

        Args:
            vocab_size: Number of tokens in vocabulary
            d_model: Dimension of embeddings
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix with small random values
        scale = 1.0 / math.sqrt(d_model)
        self.weight = Tensor(
            np.random.randn(vocab_size, d_model) * scale,
            requires_grad=True
        )

    def forward(self, indices):
        """
        Look up embeddings for given indices with gradient support.

        Args:
            indices: Numpy array of token IDs [batch_size, seq_len]

        Returns:
            Tensor [batch_size, seq_len, d_model]
        """
        # Simple lookup: gather rows from weight matrix
        embeddings = self.weight.data[indices]  # [batch_size, seq_len, d_model]
        out = Tensor(embeddings, requires_grad=True,
                    _children=(self.weight,), _op='embed')

        saved_indices = indices

        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                # Scatter-add: accumulate gradients for each looked-up index
                np.add.at(self.weight.grad, saved_indices, out.grad)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


# =========================
# ATTENTION LAYER
# =========================

class MultiHeadAttention(Module):
    """
    Multi-head self-attention mechanism.

    Simplified version that computes attention explicitly.
    """

    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Forward pass using Tensor ops to maintain computation graph.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional causal mask [seq_len, seq_len] (numpy array)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Linear projections (tracked in graph via Linear.__call__)
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape and transpose using Tensor ops (stays in graph)
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        K_T = K.transpose(0, 1, 3, 2)  # [batch, heads, d_k, seq_len]
        scores = (Q @ K_T) * Tensor(1.0 / math.sqrt(self.d_k))

        # Apply causal mask if provided
        if mask is not None:
            scores = scores + Tensor(mask[np.newaxis, np.newaxis, :, :])

        # Softmax to get attention weights (tracked in graph)
        attn_weights = scores.softmax(axis=-1)

        # Apply attention to values
        attn_output = attn_weights @ V  # [batch, heads, seq_len, d_k]

        # Transpose back and reshape to concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection
        output = self.W_o(attn_output)

        return output

    def parameters(self):
        params = []
        params.extend(self.W_q.parameters())
        params.extend(self.W_k.parameters())
        params.extend(self.W_v.parameters())
        params.extend(self.W_o.parameters())
        return params


# =========================
# FEED-FORWARD LAYER
# =========================

class FeedForward(Module):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
        """
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params


# =========================
# DECODER LAYER
# =========================

class DecoderLayer(Module):
    """
    Single decoder layer with:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections (simplified)
    """

    def __init__(self, d_model, n_heads, d_ff):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Causal mask [seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection (uses Tensor __add__ backward)
        attn_output = self.self_attn(x, mask)
        x = x + attn_output

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output

        return x

    def parameters(self):
        params = []
        params.extend(self.self_attn.parameters())
        params.extend(self.feed_forward.parameters())
        return params


# =========================
# TRANSFORMER DECODER
# =========================

class TransformerDecoderLite(Module):
    """
    Decoder-only transformer using TorchLite.

    Simplified version for educational purposes.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=3,
                 d_ff=1024, max_len=128):
        """
        Initialize decoder-only transformer.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Token embedding
        self.embed = Embedding(vocab_size, d_model)

        # Positional encoding (fixed)
        self.pos_encoding = create_positional_encoding(max_len, d_model)

        # Decoder layers
        self.layers = [
            DecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Output projection
        self.output = Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input token IDs [batch_size, seq_len] (numpy array)
            mask: Causal mask [seq_len, seq_len] (optional)

        Returns:
            Logits [batch_size, seq_len, vocab_size] (Tensor)
        """
        batch_size, seq_len = x.shape

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len)

        # Embed tokens
        x = self.embed(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding (using Tensor addition to stay in graph)
        pos_enc = self.pos_encoding[:seq_len, :] * math.sqrt(self.d_model)
        x = x + Tensor(pos_enc[np.newaxis, :, :])

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary
        logits = self.output(x)

        return logits

    def generate(self, nl_tokens, start_id, end_id, max_len=64):
        """
        Auto-regressive generation (greedy decoding).

        Args:
            nl_tokens: NL input token IDs [batch_size, nl_len] (numpy array)
            start_id: ID of <START> token
            end_id: ID of <END> token
            max_len: Maximum generation length

        Returns:
            Generated command IDs [batch_size, generated_len] (numpy array)
        """
        batch_size = nl_tokens.shape[0]

        # Start with NL tokens + <START>
        seq = np.concatenate([
            nl_tokens,
            np.full((batch_size, 1), start_id, dtype=np.int64)
        ], axis=1)

        nl_len = nl_tokens.shape[1]
        generated_tokens = []

        for _ in range(max_len):
            # Forward pass
            logits = self.forward(seq)

            # Get logits for next token (last position)
            next_logits = logits.data[:, -1, :]  # [batch_size, vocab_size]

            # Greedy decoding: take argmax
            next_token = np.argmax(next_logits, axis=-1, keepdims=True)  # [batch_size, 1]

            # Append to sequence
            seq = np.concatenate([seq, next_token], axis=1)
            generated_tokens.append(next_token)

            # Stop if all sequences generated <END>
            if np.all(next_token == end_id):
                break

        # Concatenate generated tokens
        generated = np.concatenate(generated_tokens, axis=1)

        return generated

    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embed.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output.parameters())
        return params


if __name__ == '__main__':
    # Test the model
    print("=" * 60)
    print("Testing TorchLite Decoder-Only Transformer")
    print("=" * 60)

    # Model parameters (smaller for TorchLite)
    vocab_size = 9078
    d_model = 128  # Smaller than PyTorch version
    n_heads = 4
    n_layers = 3   # Fewer layers
    batch_size = 2
    seq_len = 20

    # Create model
    print("\nCreating model...")
    model = TransformerDecoderLite(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )

    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dim: {d_model}")
    print(f"  Heads: {n_heads}")
    print(f"  Layers: {n_layers}")

    # Count parameters
    n_params = sum(p.data.size for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = model.forward(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")

    # Test generation
    print(f"\nTesting generation...")
    nl_tokens = np.random.randint(0, vocab_size, (batch_size, 10))
    generated = model.generate(nl_tokens, start_id=2, end_id=3, max_len=10)

    print(f"  NL input shape: {nl_tokens.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0]}")

    print("\n" + "=" * 60)
    print("TorchLite model test complete!")
    print("=" * 60)
    print("\nNote: TorchLite is for educational purposes.")
    print("For actual training, use the PyTorch version (faster & more features).")
