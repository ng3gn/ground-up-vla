"""
Training script for NL2Bash decoder-only transformer.

This script trains and evaluates the model on the NL2Bash dataset.
Supports both PyTorch and TorchLite implementations.
"""

import os
import argparse
import json
import time
from typing import Dict, List

from vocab import Vocabulary
from tokenizer import NL2BashTokenizer
from dataset import NL2BashDataset, PYTORCH_AVAILABLE

# Try to import PyTorch
if PYTORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from dataset import create_pytorch_dataloader
    from model.transformer_full import TransformerDecoder
    print("Using PyTorch implementation")
else:
    print("PyTorch not available. Install PyTorch to train the model.")
    print("pip install torch")
    exit(1)


def compute_masked_loss(logits: torch.Tensor, targets: torch.Tensor,
                        loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with masking.

    Only computes loss on command tokens (loss_mask=1), ignoring NL tokens.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        loss_mask: [batch_size, seq_len] - 1 for command tokens, 0 for NL/padding

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross-entropy: [batch_size * seq_len, vocab_size]
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    loss_mask_flat = loss_mask.view(-1)

    # Compute cross-entropy loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(logits_flat, targets_flat)

    # Apply mask (only compute loss on command tokens)
    masked_loss = loss * loss_mask_flat

    # Average over non-masked tokens
    total_loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)

    return total_loss


def train_epoch(model, dataloader, optimizer, device, epoch: int,
                log_interval: int = 100) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: TransformerDecoder model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        log_interval: How often to log progress

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss (only on command tokens)
        loss = compute_masked_loss(logits, target_ids, loss_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * loss_mask.sum().item()
        total_tokens += loss_mask.sum().item()

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tokens/sec: {total_tokens / elapsed:.0f}")

    # Epoch metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens,
        'time': time.time() - start_time
    }


def evaluate(model, dataloader, device) -> Dict[str, float]:
    """
    Evaluate model on dev/test set.

    Args:
        model: TransformerDecoder model
        dataloader: Evaluation data loader
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = compute_masked_loss(logits, target_ids, loss_mask)

            # Accumulate metrics
            total_loss += loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens
    }


def generate_samples(model, dataset, tokenizer, device,
                     num_samples: int = 5) -> List[Dict[str, str]]:
    """
    Generate command samples from the model.

    Args:
        model: TransformerDecoder model
        dataset: Dataset to sample from
        tokenizer: Tokenizer
        device: Device
        num_samples: Number of samples to generate

    Returns:
        List of dictionaries with NL, target command, and generated command
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            example = dataset[i]

            # Get NL tokens (without START/END)
            nl_length = example['nl_length']
            nl_ids = example['combined_ids'][:nl_length]
            nl_tokens = torch.LongTensor([nl_ids]).to(device)

            # Generate command
            generated = model.generate(
                nl_tokens,
                start_id=tokenizer.vocab.start_id,
                end_id=tokenizer.vocab.end_id,
                max_len=64
            )

            # Decode
            generated_ids = generated[0].cpu().tolist()
            generated_text = tokenizer.decode_cm(generated_ids, skip_special_tokens=True)

            samples.append({
                'nl': example['nl_text'],
                'target': example['cm_text'],
                'generated': generated_text
            })

    return samples


def calculate_exact_match(samples: List[Dict[str, str]]) -> float:
    """
    Calculate exact match accuracy.

    Args:
        samples: List of generated samples

    Returns:
        Exact match accuracy (0-1)
    """
    correct = sum(1 for s in samples if s['generated'].strip() == s['target'].strip())
    return correct / len(samples) if samples else 0.0


def save_checkpoint(model, optimizer, epoch, metrics, filepath: str):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"  Saved checkpoint to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Epoch number and metrics
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"  Loaded checkpoint from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")

    return checkpoint['epoch'], checkpoint.get('metrics', {})


def main(args):
    """Main training function."""
    print("=" * 70)
    print("NL2Bash Decoder-Only Transformer Training")
    print("=" * 70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab_path = os.path.join(args.data_dir, 'shared_vocab.txt')
    vocab = Vocabulary.load(vocab_path)
    print(f"  Vocabulary size: {len(vocab)}")

    # Create tokenizer
    tokenizer = NL2BashTokenizer(vocab)

    # Load dataset
    print("\nLoading dataset...")
    nl_file = os.path.join(args.data_dir, 'all.nl')
    cm_file = os.path.join(args.data_dir, 'all.cm')
    dataset = NL2BashDataset(nl_file, cm_file, tokenizer)

    # Split into train/dev/test
    train_dataset, dev_dataset, test_dataset = dataset.split(
        train_ratio=10, dev_ratio=1, test_ratio=1, seed=42
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_pytorch_dataloader(
        train_dataset, args.batch_size, shuffle=True, pad_id=vocab.pad_id
    )
    dev_loader = create_pytorch_dataloader(
        dev_dataset, args.batch_size, shuffle=False, pad_id=vocab.pad_id
    )
    test_loader = create_pytorch_dataloader(
        test_dataset, args.batch_size, shuffle=False, pad_id=vocab.pad_id
    )

    # Create model
    print("\nCreating model...")
    model = TransformerDecoder(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_dev_loss = float('inf')

    for epoch in range(start_epoch, args.n_epochs + 1):
        print(f"\nEpoch {epoch}/{args.n_epochs}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            log_interval=args.log_interval
        )
        print(f"\nTrain metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
        print(f"  Time: {train_metrics['time']:.1f}s")

        # Evaluate on dev set
        print(f"\nEvaluating on dev set...")
        dev_metrics = evaluate(model, dev_loader, device)
        print(f"  Loss: {dev_metrics['loss']:.4f}")
        print(f"  Perplexity: {dev_metrics['perplexity']:.2f}")

        # Generate samples
        if epoch % args.sample_interval == 0:
            print(f"\nGenerating samples...")
            samples = generate_samples(
                model, dev_dataset, tokenizer, device, num_samples=5
            )
            for i, sample in enumerate(samples):
                print(f"\n  Sample {i + 1}:")
                print(f"    NL:        {sample['nl']}")
                print(f"    Target:    {sample['target']}")
                print(f"    Generated: {sample['generated']}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, dev_metrics, checkpoint_path)

        # Save best model
        if dev_metrics['loss'] < best_dev_loss:
            best_dev_loss = dev_metrics['loss']
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, dev_metrics, best_path)
            print(f"  New best model! Dev loss: {best_dev_loss:.4f}")

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final evaluation on test set...")
    print("=" * 70)

    # Load best model
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Perplexity: {test_metrics['perplexity']:.2f}")

    # Generate test samples
    print(f"\nGenerating test samples...")
    test_samples = generate_samples(
        model, test_dataset, tokenizer, device, num_samples=10
    )

    # Calculate exact match
    exact_match = calculate_exact_match(test_samples)
    print(f"\nExact match accuracy: {exact_match:.2%}")

    # Print test samples
    print("\nTest samples:")
    for i, sample in enumerate(test_samples):
        match = "✓" if sample['generated'].strip() == sample['target'].strip() else "✗"
        print(f"\n  {match} Sample {i + 1}:")
        print(f"    NL:        {sample['nl']}")
        print(f"    Target:    {sample['target']}")
        print(f"    Generated: {sample['generated']}")

    # Save final results
    results = {
        'test_loss': test_metrics['loss'],
        'test_perplexity': test_metrics['perplexity'],
        'exact_match': exact_match,
        'samples': test_samples
    }
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NL2Bash decoder-only transformer'
    )

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Generate samples every N epochs')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)
