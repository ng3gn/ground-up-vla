"""
Decoder-only Transformer model using PyTorch.

This implements a GPT-style decoder-only transformer for the NL2Bash task.
The model processes combined sequences: [NL tokens] + <START> + [CMD tokens] + <END>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Create positional encoding matrix using sine and cosine functions.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower-triangular) mask for decoder self-attention.

    Prevents attending to future tokens.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Mask [seq_len, seq_len] where mask[i,j] = 0 if i >= j else -inf
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Supports causal masking for decoder-only architecture.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len, d_model]
            mask: [seq_len, seq_len] attention mask (optional)

        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)  # Broadcast for batch and heads

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch_size, n_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        # [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # Final linear projection
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Multi-head self-attention (with causal mask)
    2. Feed-forward network
    3. Layer normalization and residual connections
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len, d_model]
            mask: [seq_len, seq_len] causal mask

        Returns:
            [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """
    Decoder-only transformer (GPT-style) for NL2Bash.

    Processes combined sequences: [NL tokens] + <START> + [CMD tokens] + <END>
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1,
                 max_len: int = 128):
        """
        Initialize decoder-only transformer.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding (fixed)
        self.register_buffer(
            'pos_encoding',
            create_positional_encoding(max_len, d_model)
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len] - Input token IDs
            mask: [seq_len, seq_len] - Causal mask (optional, will be created if None)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len, x.device)

        # Embed tokens and add positional encoding
        # [batch_size, seq_len, d_model]
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary
        logits = self.output(x)

        return logits

    def generate(self, nl_tokens: torch.Tensor, start_id: int, end_id: int,
                 max_len: int = 64, temperature: float = 1.0,
                 top_k: int = None) -> torch.Tensor:
        """
        Auto-regressive generation for inference.

        Args:
            nl_tokens: [batch_size, nl_len] - NL input token IDs
            start_id: ID of <START> token
            end_id: ID of <END> token
            max_len: Maximum generation length
            temperature: Sampling temperature (1.0 = no change, <1 = more confident)
            top_k: If set, only sample from top-k tokens

        Returns:
            generated: [batch_size, generated_len] - Generated command IDs (without NL prefix)
        """
        self.eval()
        batch_size = nl_tokens.size(0)
        device = nl_tokens.device

        # Start with NL tokens + <START>
        seq = torch.cat([
            nl_tokens,
            torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        ], dim=1)

        nl_len = nl_tokens.size(1)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_len):
                # Forward pass
                logits = self.forward(seq)

                # Get logits for next token (last position)
                next_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(1, top_k_indices, top_k_logits)

                # Sample next token (greedy if temperature=1.0 and no top_k)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                seq = torch.cat([seq, next_token], dim=1)
                generated_tokens.append(next_token)

                # Stop if all sequences generated <END>
                if (next_token == end_id).all():
                    break

        # Concatenate generated tokens
        generated = torch.cat(generated_tokens, dim=1)

        return generated


if __name__ == '__main__':
    # Test the model
    print("=" * 60)
    print("Testing Decoder-Only Transformer")
    print("=" * 60)

    # Model parameters
    vocab_size = 9078
    d_model = 256
    n_heads = 4
    n_layers = 6
    batch_size = 2
    seq_len = 20

    # Create model
    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )

    print(f"\nModel created:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dim: {d_model}")
    print(f"  Heads: {n_heads}")
    print(f"  Layers: {n_layers}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {vocab_size}]")

    # Test generation
    print(f"\nTesting generation...")
    nl_tokens = torch.randint(0, vocab_size, (batch_size, 10))
    generated = model.generate(nl_tokens, start_id=2, end_id=3, max_len=10)

    print(f"  NL input shape: {nl_tokens.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 60)
    print("Model test complete!")
    print("=" * 60)
