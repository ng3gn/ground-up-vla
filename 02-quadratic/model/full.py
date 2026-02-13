import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

class FullModel:
    """Neural network model using PyTorch framework."""

    def __init__(self, input_size=2, hidden_sizes=[16, 8],
                 output_size=1, lr=0.01):
        """
        Initialize the model.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features
            lr: Learning rate
        """
        # Build model layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.losses = []

    def train(self, X_train, y_train, n_epochs=200, log_interval=20):
        """
        Train the model.

        Args:
            X_train: Training inputs (numpy array)
            y_train: Training targets (numpy array)
            n_epochs: Number of training epochs
            log_interval: How often to log progress

        Returns:
            List of training losses
        """
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X_train)
        y = torch.FloatTensor(y_train)

        print("Training PyTorch model...")
        for epoch in range(n_epochs):
            # Forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record
            self.losses.append(loss.item())

            # Print progress
            if epoch % log_interval == 0:
                print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}")

        print(f"  Final loss: {self.losses[-1]:.6f}")
        return self.losses

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input data (numpy array)

        Returns:
            Predictions (numpy array)
        """
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            pred = self.model(X_tensor)
        return pred.numpy()

    def save_training_plot(self, filename="outputs/pytorch_training.png"):
        """Save training loss plot."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PyTorch Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
