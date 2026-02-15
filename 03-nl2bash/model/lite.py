import numpy as np
from torchlite import Tensor
from torchlite.nn import Linear, ReLU, Sequential, MSELoss
from torchlite.optim import Adam
from torchlite.logger import TrainingLogger
from torchlite.visualize import plot_training_history

class LiteModel:
    """Neural network model using TorchLite framework."""

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
            layers.append(Linear(prev_size, hidden_size))
            layers.append(ReLU())
            prev_size = hidden_size
        layers.append(Linear(prev_size, output_size))

        self.model = Sequential(*layers)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss_fn = MSELoss()
        self.logger = TrainingLogger(name="torchlite_model", log_dir="outputs")
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
        # Convert to tensors
        X = Tensor(X_train)
        y = Tensor(y_train)

        print("Training TorchLite model...")
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

            # Log and print progress
            if epoch % log_interval == 0:
                self.logger.log_epoch(
                    epoch, X, pred, loss,
                    parameters={'layer0_weight':
                                (self.model.layers[0].weight,
                                 self.model.layers[0].weight.grad)}
                )
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
        X_tensor = Tensor(X)
        pred = self.model(X_tensor)
        return pred.data

    def save_training_plot(self, filename="outputs/torchlite_training.png"):
        """Save training loss plot."""
        plot_training_history(
            self.losses,
            title="TorchLite Training Loss",
            filename=filename
        )
