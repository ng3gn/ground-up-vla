"""
TorchLite Logging System

Simple text logging for training and inference runs.
Logs matrix values, shapes, and gradients at epoch boundaries.
"""

import numpy as np
from datetime import datetime
import os


def _format_array(arr, max_values=10):
    """Format a numpy array for pretty printing."""
    if isinstance(arr, (int, float)):
        return f"{arr:.6f}"

    arr = np.asarray(arr)
    flat = arr.flatten()

    if len(flat) <= max_values:
        values_str = ", ".join([f"{v:.6f}" for v in flat])
    else:
        first_part = ", ".join([f"{v:.6f}" for v in flat[:max_values//2]])
        last_part = ", ".join([f"{v:.6f}" for v in flat[-max_values//2:]])
        values_str = f"{first_part}, ..., {last_part}"

    return f"[{values_str}]"


def _format_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class TrainingLogger:
    """
    Logger for training runs.

    Logs forward and backward passes at epoch boundaries with:
    - Timestamps
    - Epoch markers
    - Layer-by-layer outputs and gradients
    - Matrix shapes and sample values

    Args:
        name: Base name for log file (will create <name>.training.log)
        log_dir: Directory for log files (default: current directory)
    """

    def __init__(self, name="training", log_dir="."):
        self.name = name
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"{name}.training.log")

        # Start new run section
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RUN: {_format_timestamp()}\n")
            f.write("=" * 80 + "\n\n")

    def log_epoch(self, epoch, input_data, output, loss, parameters=None):
        """
        Log a complete epoch (forward + backward pass).

        Args:
            epoch: Epoch number
            input_data: Input tensor/array
            output: Output prediction
            loss: Loss value or Loss tensor
            parameters: Dict of parameter name -> (value, gradient) tuples
                       e.g., {'weight': (w.data, w.grad), 'bias': (b.data, b.grad)}
        """
        with open(self.log_path, 'a') as f:
            # Forward pass
            f.write(f"[EPOCH {epoch} - FORWARD PASS]\n")
            f.write(f"Timestamp: {_format_timestamp()}\n\n")

            # Input
            if hasattr(input_data, 'data'):
                input_array = input_data.data
            else:
                input_array = np.asarray(input_data)

            f.write("Input:\n")
            f.write(f"  Shape: {input_array.shape}\n")
            f.write(f"  Values: {_format_array(input_array)}\n\n")

            # Output
            if hasattr(output, 'data'):
                output_array = output.data
            else:
                output_array = np.asarray(output)

            f.write("Output:\n")
            f.write(f"  Shape: {output_array.shape}\n")
            f.write(f"  Values: {_format_array(output_array)}\n\n")

            # Backward pass
            f.write(f"[EPOCH {epoch} - BACKWARD PASS]\n")
            f.write(f"Timestamp: {_format_timestamp()}\n\n")

            # Loss
            if hasattr(loss, 'data'):
                loss_val = np.asarray(loss.data).item()
            elif hasattr(loss, 'item'):
                loss_val = loss.item()
            else:
                loss_val = float(loss)

            f.write(f"Loss: {loss_val:.6f}\n\n")

            # Parameters and gradients
            if parameters:
                f.write("Parameters and Gradients:\n")
                for name, (value, grad) in parameters.items():
                    if hasattr(value, 'data'):
                        value_array = value.data
                    else:
                        value_array = np.asarray(value)

                    f.write(f"\n  {name}:\n")
                    f.write(f"    Shape: {value_array.shape}\n")
                    f.write(f"    Values: {_format_array(value_array)}\n")

                    if grad is not None:
                        if hasattr(grad, 'data'):
                            grad_array = grad.data
                        else:
                            grad_array = np.asarray(grad)
                        f.write(f"    Gradient shape: {grad_array.shape}\n")
                        f.write(f"    Gradient values: {_format_array(grad_array)}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    def log_forward(self, epoch, layers_data):
        """
        Log just the forward pass with layer-by-layer outputs.

        Args:
            epoch: Epoch number
            layers_data: List of (layer_name, output) tuples
        """
        with open(self.log_path, 'a') as f:
            f.write(f"[EPOCH {epoch} - FORWARD PASS]\n")
            f.write(f"Timestamp: {_format_timestamp()}\n\n")

            for layer_name, output in layers_data:
                if hasattr(output, 'data'):
                    output_array = output.data
                else:
                    output_array = np.asarray(output)

                f.write(f"{layer_name}:\n")
                f.write(f"  Shape: {output_array.shape}\n")
                f.write(f"  Values: {_format_array(output_array)}\n\n")

    def log_backward(self, epoch, loss, gradients_data):
        """
        Log just the backward pass with gradients.

        Args:
            epoch: Epoch number
            loss: Loss value or tensor
            gradients_data: List of (param_name, gradient) tuples
        """
        with open(self.log_path, 'a') as f:
            f.write(f"[EPOCH {epoch} - BACKWARD PASS]\n")
            f.write(f"Timestamp: {_format_timestamp()}\n\n")

            # Loss
            if hasattr(loss, 'data'):
                loss_val = np.asarray(loss.data).item()
            elif hasattr(loss, 'item'):
                loss_val = loss.item()
            else:
                loss_val = float(loss)

            f.write(f"Loss: {loss_val:.6f}\n\n")

            # Gradients
            if gradients_data:
                f.write("Gradients:\n")
                for param_name, grad in gradients_data:
                    if grad is not None:
                        if hasattr(grad, 'data'):
                            grad_array = grad.data
                        else:
                            grad_array = np.asarray(grad)

                        f.write(f"\n  {param_name}:\n")
                        f.write(f"    Shape: {grad_array.shape}\n")
                        f.write(f"    Values: {_format_array(grad_array)}\n")

            f.write("\n" + "-" * 80 + "\n\n")


class InferenceLogger:
    """
    Logger for inference runs.

    Logs forward passes (no gradients) with:
    - Timestamps
    - Step markers
    - Layer-by-layer outputs
    - Matrix shapes and sample values

    Args:
        name: Base name for log file (will create <name>.inference.log)
        log_dir: Directory for log files (default: current directory)
    """

    def __init__(self, name="inference", log_dir="."):
        self.name = name
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"{name}.inference.log")

        # Start new run section
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RUN: {_format_timestamp()}\n")
            f.write("=" * 80 + "\n\n")

    def log_step(self, step, input_data, output, layers_data=None):
        """
        Log an inference step.

        Args:
            step: Step number
            input_data: Input tensor/array
            output: Output prediction
            layers_data: Optional list of (layer_name, output) tuples for intermediate layers
        """
        with open(self.log_path, 'a') as f:
            f.write(f"[STEP {step} - INFERENCE]\n")
            f.write(f"Timestamp: {_format_timestamp()}\n\n")

            # Input
            if hasattr(input_data, 'data'):
                input_array = input_data.data
            else:
                input_array = np.asarray(input_data)

            f.write("Input:\n")
            f.write(f"  Shape: {input_array.shape}\n")
            f.write(f"  Values: {_format_array(input_array)}\n\n")

            # Layer-by-layer outputs if provided
            if layers_data:
                f.write("Layer Outputs:\n")
                for layer_name, layer_output in layers_data:
                    if hasattr(layer_output, 'data'):
                        output_array = layer_output.data
                    else:
                        output_array = np.asarray(layer_output)

                    f.write(f"\n  {layer_name}:\n")
                    f.write(f"    Shape: {output_array.shape}\n")
                    f.write(f"    Values: {_format_array(output_array)}\n")
                f.write("\n")

            # Final output
            if hasattr(output, 'data'):
                output_array = output.data
            else:
                output_array = np.asarray(output)

            f.write("Final Output:\n")
            f.write(f"  Shape: {output_array.shape}\n")
            f.write(f"  Values: {_format_array(output_array)}\n")

            f.write("\n" + "-" * 80 + "\n\n")
