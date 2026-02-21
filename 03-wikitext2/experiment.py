"""
Depth Experiment: Compare decoder depths for NL2Bash.

Trains models at various n_layers depths and plots loss curves.
- PyTorch: depths [1, 2, 4, 8, 16], 20 epochs each, GPU
- TorchLite: depth [1], 3 epochs, CPU (educational autograd)
"""

import os
import sys
import json
import time
import math
import numpy as np

# Add parent directory for torchlite imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vocab import Vocabulary
from tokenizer import NL2BashTokenizer
from dataset import NL2BashDataset, create_torchlite_dataloader

# =========================
# CONFIGURATION
# =========================

PYTORCH_CONFIG = {
    'd_model': 128,
    'n_heads': 1,
    'd_ff': 512,
    'batch_size': 16,
    'lr': 0.0001,
    'n_epochs': 20,
    'depths': [1, 2, 4, 8, 16],
    'dropout': 0.0, # TODO: Was 0.1, made 0 because haven't explained it yet
    'max_len': 128,
}

TORCHLITE_CONFIG = {
    'd_model': 128,
    'n_heads': 1,
    'd_ff': 512,
    'batch_size': 16,
    'lr': 0.001,
    'n_epochs': 3,
    'depths': [1],
    'max_len': 128,
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'experiment_outputs')


# =========================
# DATA LOADING
# =========================

def load_data(data_dir='data'):
    """Load vocabulary, tokenizer, and dataset. Returns (vocab, tokenizer, train, dev, test)."""
    vocab_path = os.path.join(data_dir, 'shared_vocab.txt')
    vocab = Vocabulary.load(vocab_path)
    tokenizer = NL2BashTokenizer(vocab)

    nl_file = os.path.join(data_dir, 'all.nl')
    cm_file = os.path.join(data_dir, 'all.cm')
    dataset = NL2BashDataset(nl_file, cm_file, tokenizer)

    train_dataset, dev_dataset, test_dataset = dataset.split(train_ratio=10, dev_ratio=1, test_ratio=1, seed=42)
    return vocab, tokenizer, train_dataset, dev_dataset, test_dataset


# =========================
# PYTORCH EXPERIMENT
# =========================

def compute_exact_match_pytorch(model, dataset, tokenizer, device, num_samples=100):
    """Generate commands and compute exact match accuracy (PyTorch)."""
    import torch
    model.eval()
    correct = 0
    total = min(num_samples, len(dataset))

    with torch.no_grad():
        for i in range(total):
            ex = dataset[i]
            nl_length = ex['nl_length']
            nl_ids = ex['combined_ids'][:nl_length]
            nl_tokens = torch.LongTensor([nl_ids]).to(device)

            generated = model.generate(
                nl_tokens,
                start_id=tokenizer.vocab.start_id,
                end_id=tokenizer.vocab.end_id,
                max_len=64
            )

            generated_ids = generated[0].cpu().tolist()
            generated_text = tokenizer.decode_cm(generated_ids, skip_special_tokens=True)

            if generated_text.strip() == ex['cm_text'].strip():
                correct += 1

    return correct / total if total > 0 else 0.0


def compute_exact_match_lite(model, dataset, tokenizer, num_samples=20):
    """Generate commands and compute exact match accuracy (TorchLite)."""
    correct = 0
    total = min(num_samples, len(dataset))

    for i in range(total):
        ex = dataset[i]
        nl_length = ex['nl_length']
        nl_ids = ex['combined_ids'][:nl_length]
        nl_tokens = np.array([nl_ids], dtype=np.int64)

        generated = model.generate(
            nl_tokens,
            start_id=tokenizer.vocab.start_id,
            end_id=tokenizer.vocab.end_id,
            max_len=64
        )

        generated_ids = generated[0].tolist()
        generated_text = tokenizer.decode_cm(generated_ids, skip_special_tokens=True)

        if generated_text.strip() == ex['cm_text'].strip():
            correct += 1

    return correct / total if total > 0 else 0.0


def run_pytorch_experiment(vocab, tokenizer, train_dataset, dev_dataset, test_dataset, config):
    """Train PyTorch models at each depth and return loss histories + final metrics."""
    import torch
    import torch.optim as optim
    from dataset import create_pytorch_dataloader
    from model.transformer_full import TransformerDecoder
    from main import compute_masked_loss, train_epoch, evaluate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nPyTorch device: {device}")

    results = {}

    for depth in config['depths']:
        print(f"\n{'='*60}")
        print(f"PyTorch: Training depth={depth}")
        print(f"{'='*60}")

        model = TransformerDecoder(
            vocab_size=len(vocab),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=depth,
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_len=config['max_len'],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        train_loader = create_pytorch_dataloader(
            train_dataset, config['batch_size'], shuffle=True, pad_id=vocab.pad_id
        )
        dev_loader = create_pytorch_dataloader(
            dev_dataset, config['batch_size'], shuffle=False, pad_id=vocab.pad_id
        )
        test_loader = create_pytorch_dataloader(
            test_dataset, config['batch_size'], shuffle=False, pad_id=vocab.pad_id
        )

        train_losses = []
        dev_losses = []

        for epoch in range(1, config['n_epochs'] + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, log_interval=9999)
            dev_metrics = evaluate(model, dev_loader, device)

            train_losses.append(train_metrics['loss'])
            dev_losses.append(dev_metrics['loss'])

            print(f"  Epoch {epoch:2d}/{config['n_epochs']} | "
                  f"Train loss: {train_metrics['loss']:.4f} | "
                  f"Dev loss: {dev_metrics['loss']:.4f} | "
                  f"Time: {train_metrics['time']:.1f}s")

        # Final evaluation on test set
        test_metrics = evaluate(model, test_loader, device)
        print(f"\n  Test loss: {test_metrics['loss']:.4f}")

        # Exact match accuracy on test set
        print(f"  Computing exact match on test set...")
        exact_match = compute_exact_match_pytorch(model, test_dataset, tokenizer, device, num_samples=100)
        print(f"  Exact match accuracy: {exact_match:.2%}")

        results[depth] = {
            'train_losses': train_losses,
            'dev_losses': dev_losses,
            'final_train_loss': train_losses[-1],
            'final_dev_loss': dev_losses[-1],
            'final_test_loss': test_metrics['loss'],
            'exact_match': exact_match,
            'n_params': n_params,
        }

        print(f"\n  >> depth={depth}: "
              f"train={train_losses[-1]:.4f} | "
              f"dev={dev_losses[-1]:.4f} | "
              f"test={test_metrics['loss']:.4f} | "
              f"exact_match={exact_match:.1%}")

    return results


# =========================
# TORCHLITE EXPERIMENT
# =========================

def compute_masked_loss_lite(logits, targets, loss_mask):
    """
    Compute cross-entropy loss for TorchLite tensors with masking.

    Args:
        logits: Tensor [batch_size, seq_len, vocab_size]
        targets: numpy array [batch_size, seq_len]
        loss_mask: numpy array [batch_size, seq_len]

    Returns:
        Scalar Tensor loss
    """
    from torchlite.tensor import Tensor

    batch_size, seq_len, vocab_size = logits.shape

    # Softmax over vocab dimension
    probs = logits.softmax(axis=-1)

    # Gather probabilities at target indices: one-hot style
    # Create one-hot targets
    targets_flat = targets.reshape(-1)
    one_hot = np.zeros((batch_size * seq_len, vocab_size), dtype=np.float64)
    one_hot[np.arange(batch_size * seq_len), targets_flat] = 1.0

    # Reshape probs to [batch*seq, vocab]
    probs_flat = probs.reshape(batch_size * seq_len, vocab_size)

    # Gather: sum(probs * one_hot, axis=-1) gives prob at target index
    # Then take -log
    target_probs = (probs_flat * Tensor(one_hot)).sum(axis=-1)  # [batch*seq]
    neg_log_probs = target_probs.log() * Tensor(-1.0)  # [batch*seq]

    # Apply mask
    mask_flat = Tensor(loss_mask.reshape(-1).astype(np.float64))
    masked_losses = neg_log_probs * mask_flat

    # Average over masked tokens
    n_masked = float(loss_mask.sum()) + 1e-8
    loss = masked_losses.sum() * Tensor(1.0 / n_masked)

    return loss


def train_epoch_lite(model, dataloader, optimizer, n_batches, epoch):
    """Train TorchLite model for one epoch."""
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        loss_mask = batch['loss_mask']

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model.forward(input_ids)

        # Compute loss
        loss = compute_masked_loss_lite(logits, target_ids, loss_mask)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate metrics
        n_tokens = float(loss_mask.sum())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / total_tokens
            print(f"    Batch {batch_idx + 1}/{n_batches} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return {'loss': avg_loss, 'time': time.time() - start_time}


def evaluate_lite(model, dataloader, n_batches):
    """Evaluate TorchLite model."""
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        loss_mask = batch['loss_mask']

        logits = model.forward(input_ids)
        loss = compute_masked_loss_lite(logits, target_ids, loss_mask)

        n_tokens = float(loss_mask.sum())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return {'loss': avg_loss}


def run_torchlite_experiment(vocab, tokenizer, train_dataset, dev_dataset, test_dataset, config):
    """Train TorchLite models and return loss histories + final metrics."""
    from torchlite.optim import Adam
    from model.transformer_lite import TransformerDecoderLite

    results = {}
    n_train_batches = len(train_dataset) // config['batch_size']
    n_dev_batches = max(1, len(dev_dataset) // config['batch_size'])
    n_test_batches = max(1, len(test_dataset) // config['batch_size'])

    for depth in config['depths']:
        print(f"\n{'='*60}")
        print(f"TorchLite: Training depth={depth}")
        print(f"{'='*60}")

        model = TransformerDecoderLite(
            vocab_size=len(vocab),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=depth,
            d_ff=config['d_ff'],
            max_len=config['max_len'],
        )

        n_params = sum(p.data.size for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = Adam(model.parameters(), lr=config['lr'])

        train_losses = []
        dev_losses = []

        for epoch in range(1, config['n_epochs'] + 1):
            print(f"\n  Epoch {epoch}/{config['n_epochs']}")

            train_loader = create_torchlite_dataloader(
                train_dataset, config['batch_size'], shuffle=True, pad_id=vocab.pad_id
            )
            dev_loader = create_torchlite_dataloader(
                dev_dataset, config['batch_size'], shuffle=False, pad_id=vocab.pad_id
            )

            train_metrics = train_epoch_lite(
                model, train_loader, optimizer, n_train_batches, epoch
            )
            dev_metrics = evaluate_lite(model, dev_loader, n_dev_batches)

            train_losses.append(train_metrics['loss'])
            dev_losses.append(dev_metrics['loss'])

            print(f"  Epoch {epoch} | "
                  f"Train loss: {train_metrics['loss']:.4f} | "
                  f"Dev loss: {dev_metrics['loss']:.4f} | "
                  f"Time: {train_metrics['time']:.1f}s")

        # Final evaluation on test set
        test_loader = create_torchlite_dataloader(
            test_dataset, config['batch_size'], shuffle=False, pad_id=vocab.pad_id
        )
        test_metrics = evaluate_lite(model, test_loader, n_test_batches)
        print(f"\n  Test loss: {test_metrics['loss']:.4f}")

        # Exact match accuracy on test set (small subset — generation is slow)
        print(f"  Computing exact match on test set...")
        exact_match = compute_exact_match_lite(model, test_dataset, tokenizer, num_samples=20)
        print(f"  Exact match accuracy: {exact_match:.2%}")

        results[depth] = {
            'train_losses': train_losses,
            'dev_losses': dev_losses,
            'final_train_loss': train_losses[-1],
            'final_dev_loss': dev_losses[-1],
            'final_test_loss': test_metrics['loss'],
            'exact_match': exact_match,
            'n_params': n_params,
        }

        print(f"\n  >> depth={depth}: "
              f"train={train_losses[-1]:.4f} | "
              f"dev={dev_losses[-1]:.4f} | "
              f"test={test_metrics['loss']:.4f} | "
              f"exact_match={exact_match:.1%}")

    return results


# =========================
# PLOTTING
# =========================

def plot_depth_comparison(results, framework_name, config, output_path):
    """Plot train and dev loss curves for each depth."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{framework_name} — Depth Comparison (d_model={config["d_model"]})', fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

    for (depth, data), color in zip(sorted(results.items()), colors):
        epochs = list(range(1, len(data['train_losses']) + 1))
        label = f'n_layers={depth} ({data["n_params"]:,} params)'

        ax1.plot(epochs, data['train_losses'], color=color, label=label, marker='o', markersize=3)
        ax2.plot(epochs, data['dev_losses'], color=color, label=label, marker='o', markersize=3)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Dev Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_final_bar_chart(results, framework_name, config, output_path):
    """Bar chart of final train/dev/test loss and exact match per depth."""
    import matplotlib.pyplot as plt

    depths = sorted(results.keys())
    train_vals = [results[d]['final_train_loss'] for d in depths]
    dev_vals = [results[d]['final_dev_loss'] for d in depths]
    test_vals = [results[d]['final_test_loss'] for d in depths]
    em_vals = [results[d]['exact_match'] for d in depths]

    x = np.arange(len(depths))
    width = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{framework_name} — Final Metrics by Depth (d_model={config["d_model"]})', fontsize=14)

    # Loss bar chart
    ax1.bar(x - width, train_vals, width, label='Train', color='#2196F3')
    ax1.bar(x, dev_vals, width, label='Dev', color='#FF9800')
    ax1.bar(x + width, test_vals, width, label='Test', color='#4CAF50')
    ax1.set_xlabel('n_layers')
    ax1.set_ylabel('Loss')
    ax1.set_title('Final Loss')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Exact match bar chart
    bars = ax2.bar(x, [v * 100 for v in em_vals], width * 2, color='#9C27B0')
    ax2.set_xlabel('n_layers')
    ax2.set_ylabel('Exact Match (%)')
    ax2.set_title('Test Exact Match Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, em_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart to {output_path}")


def save_results(results, output_path):
    """Save results dict to JSON (convert int keys to strings)."""
    serializable = {}
    for depth, data in results.items():
        serializable[str(depth)] = data
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results to {output_path}")


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("NL2Bash Depth Experiment")
    print("="*60)

    # Load data once
    vocab, tokenizer, train_dataset, dev_dataset, test_dataset = load_data()
    print(f"\nVocab size: {len(vocab)}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(dev_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # --- PyTorch experiment ---
    print("\n" + "="*60)
    print("PYTORCH EXPERIMENT")
    print("="*60)

    pytorch_results = run_pytorch_experiment(
        vocab, tokenizer, train_dataset, dev_dataset, test_dataset, PYTORCH_CONFIG
    )

    save_results(pytorch_results, os.path.join(OUTPUT_DIR, 'pytorch_results.json'))
    plot_depth_comparison(
        pytorch_results, 'PyTorch', PYTORCH_CONFIG,
        os.path.join(OUTPUT_DIR, 'pytorch_depth_comparison.png')
    )
    plot_final_bar_chart(
        pytorch_results, 'PyTorch', PYTORCH_CONFIG,
        os.path.join(OUTPUT_DIR, 'pytorch_final_bar_chart.png')
    )

    # --- TorchLite experiment ---
    print("\n" + "="*60)
    print("TORCHLITE EXPERIMENT")
    print("="*60)

    torchlite_results = run_torchlite_experiment(
        vocab, tokenizer, train_dataset, dev_dataset, test_dataset, TORCHLITE_CONFIG
    )

    save_results(torchlite_results, os.path.join(OUTPUT_DIR, 'torchlite_results.json'))
    plot_depth_comparison(
        torchlite_results, 'TorchLite', TORCHLITE_CONFIG,
        os.path.join(OUTPUT_DIR, 'torchlite_depth_comparison.png')
    )
    plot_final_bar_chart(
        torchlite_results, 'TorchLite', TORCHLITE_CONFIG,
        os.path.join(OUTPUT_DIR, 'torchlite_final_bar_chart.png')
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
