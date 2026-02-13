import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import torchlite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lite import LiteModel # TorchLite
from model.full import FullModel # PyTorch

def load_data(filename):
    """Load data from a .dat file."""
    data = np.loadtxt(filename)
    X = data[:, :2]  # First two columns are inputs
    y = data[:, 2:3]  # Third column is output
    return X, y


def train_torchlite_network():
    """Train the TorchLite model."""
    # Load training data
    X_train, y_train = load_data('data/train.dat')

    # Create and train model
    model = \
        LiteModel(input_size=2, hidden_sizes=[16, 8], output_size=1, lr=0.01)
    model.train(X_train, y_train, n_epochs=200, log_interval=20)

    # Save training plot
    model.save_training_plot("outputs/torchlite_training.png")

    return model


def train_pytorch_network():
    """Train the PyTorch model."""
    # Load training data
    X_train, y_train = load_data('data/train.dat')

    # Create and train model
    model = \
        FullModel(input_size=2, hidden_sizes=[16, 8], output_size=1, lr=0.01)
    model.train(X_train, y_train, n_epochs=200, log_interval=20)

    # Save training plot
    model.save_training_plot("outputs/pytorch_training.png")

    return model


def run_inference_tests(model, model_name):
    """
    Run inference tests on the test dataset.

    Args:
        model: Trained model (LiteModel or FullModel)
        model_name: Name of the model for display

    Returns:
        Dictionary with test results
    """
    # Load test data
    X_test, y_test = load_data('data/test.dat')

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate error metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    max_error = np.max(np.abs(predictions - y_test))

    print(f"\n{model_name} Test Results:")
    print(f"  MSE:        {mse:.6f}")
    print(f"  MAE:        {mae:.6f}")
    print(f"  Max Error:  {max_error:.6f}")

    # Show a few predictions
    print(f"\n  Sample predictions (first 5):")
    for i in range(min(5, len(X_test))):
        print(f"    Input: [{X_test[i, 0]:7.4f}, {X_test[i, 1]:7.4f}] "
              f"-> Pred: {predictions[i, 0]:7.4f}, True: {y_test[i, 0]:7.4f}, "
              f"Error: {abs(predictions[i, 0] - y_test[i, 0]):7.4f}")

    return {
        'model_name': model_name,
        'predictions': predictions,
        'targets': y_test,
        'inputs': X_test,
        'mse': mse,
        'mae': mae,
        'max_error': max_error
    }


def graph_results(results_lite, results_full):
    """
    Create comparison graphs of both models.

    Args:
        results_lite: Results from TorchLite model
        results_full: Results from PyTorch model
    """
    os.makedirs('outputs', exist_ok=True)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TorchLite vs PyTorch Model Comparison',
                 fontsize=16, fontweight='bold')

    # Plot 1: Predictions vs Actual for TorchLite
    ax = axes[0, 0]
    ax.scatter(results_lite['targets'], results_lite['predictions'],
               alpha=0.6, label='Predictions')
    ax.plot([results_lite['targets'].min(), results_lite['targets'].max()],
            [results_lite['targets'].min(), results_lite['targets'].max()],
            'r--', label='Perfect Fit')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('TorchLite: Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Predictions vs Actual for PyTorch
    ax = axes[0, 1]
    ax.scatter(results_full['targets'], results_full['predictions'],
               alpha=0.6, label='Predictions')
    ax.plot([results_full['targets'].min(), results_full['targets'].max()],
            [results_full['targets'].min(), results_full['targets'].max()],
            'r--', label='Perfect Fit')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('PyTorch: Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction errors
    ax = axes[1, 0]
    errors_lite = np.abs(results_lite['predictions'] - results_lite['targets'])
    errors_full = np.abs(results_full['predictions'] - results_full['targets'])
    x_pos = np.arange(len(errors_lite))
    width = 0.35
    ax.bar(x_pos - width/2, errors_lite.flatten(),
           width, label='TorchLite', alpha=0.7)
    ax.bar(x_pos + width/2, errors_full.flatten(),
           width, label='PyTorch', alpha=0.7)
    ax.set_xlabel('Test Sample')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Prediction Errors by Sample')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Metrics comparison
    ax = axes[1, 1]
    metrics = ['MSE', 'MAE', 'Max Error']
    lite_vals = [results_lite['mse'], results_lite['mae'],
                 results_lite['max_error']]
    full_vals = [results_full['mse'], results_full['mae'],
                 results_full['max_error']]

    x_pos = np.arange(len(metrics))
    width = 0.35
    ax.bar(x_pos - width/2, lite_vals, width, label='TorchLite', alpha=0.7)
    ax.bar(x_pos + width/2, full_vals, width, label='PyTorch', alpha=0.7)
    ax.set_ylabel('Error Value')
    ax.set_title('Error Metrics Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(lite_vals, full_vals)):
        ax.text(i - width/2, v1, f'{v1:.4f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, v2, f'{v2:.4f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to outputs/comparison.png")


if __name__ == '__main__':
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    print("=" * 60)
    print("Training Neural Networks to Learn Quadratic Function")
    print("=" * 60)

    # Train both models
    tlite = train_torchlite_network()
    print()
    tfull = train_pytorch_network()

    # Run inference tests
    results_lite = run_inference_tests(tlite, "TorchLite")
    results_full = run_inference_tests(tfull, "PyTorch")

    # Generate comparison graphs
    print("\n" + "=" * 60)
    print("Generating comparison graphs...")
    graph_results(results_lite, results_full)

    print("\n" + "=" * 60)
    print("Done! Check outputs/ directory for:")
    print("  - torchlite_training.png: TorchLite training loss")
    print("  - pytorch_training.png: PyTorch training loss")
    print("  - comparison.png: Model comparison")
    print("  - torchlite_model_training.log: Detailed TorchLite logs")
    print("=" * 60)
