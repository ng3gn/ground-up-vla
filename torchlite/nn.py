"""
TorchLite Neural Network Layers

Simple, understandable implementations of common NN building blocks.
"""

import numpy as np
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    A module can contain parameters (weights) and define a forward pass.
    """

    def parameters(self):
        """Return all trainable parameters in this module"""
        return []

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        """Make module callable: model(x) calls model.forward(x)"""
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Fully connected linear layer: y = x @ W.T + b

    This is the building block of neural networks!

    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term (default True)
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with small random values
        # Using He initialization: scale by sqrt(2/in_features)
        # Stored as (in_features, out_features) so forward pass can do x @ W
        # directly without transposing, which keeps the computation graph intact.
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )

        # Initialize bias to zeros
        self.use_bias = bias
        if bias:
            self.bias = Tensor(
                np.zeros((out_features,)),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass: y = x @ W + b

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Matrix multiplication: (batch, in) @ (in, out) = (batch, out)
        # Weight is stored as (in, out) so we can matmul directly with the
        # Tensor object, keeping it in the computation graph for backprop.
        out = x @ self.weight

        # Add bias
        if self.use_bias:
            out = out + self.bias

        return out

    def parameters(self):
        """Return trainable parameters"""
        if self.use_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class Sequential(Module):
    """
    Container for stacking layers sequentially.

    Example:
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1)
        )
        output = model(input)
    """

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return all parameters from all layers"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_str = '\n  '.join(str(layer) for layer in self.layers)
        return f"Sequential(\n  {layer_str}\n)"


# =========================
# ACTIVATION FUNCTIONS
# =========================

class ReLU(Module):
    """
    ReLU activation: max(0, x)

    Kills negative values, passes positive values unchanged.
    Most common activation in modern neural networks.
    """

    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation: 1 / (1 + exp(-x))

    Squashes values to range (0, 1).
    Often used for binary classification.
    """

    def forward(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """
    Tanh activation: tanh(x)

    Squashes values to range (-1, 1).
    Zero-centered (unlike sigmoid).
    """

    def forward(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


# =========================
# LOSS FUNCTIONS
# =========================

class MSELoss(Module):
    """
    Mean Squared Error Loss: mean((pred - target)^2)

    Used for regression problems.
    Measures average squared difference between predictions and targets.
    """

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values
            target: Target values (ground truth)

        Returns:
            Scalar loss value
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)

        diff = pred - target
        squared = diff * diff
        return squared.mean()

    def __repr__(self):
        return "MSELoss()"


class BCELoss(Module):
    """
    Binary Cross-Entropy Loss

    Used for binary classification (two classes).
    Assumes predictions are probabilities in range (0, 1).

    Loss = -mean(target * log(pred) + (1-target) * log(1-pred))
    """

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (should be in range 0-1)
            target: Target labels (0 or 1)

        Returns:
            Scalar loss value
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Clip predictions to avoid log(0)
        eps = 1e-7
        pred_clipped = Tensor(np.clip(pred.data, eps, 1 - eps), requires_grad=pred.requires_grad)

        # BCE formula: -mean(y*log(p) + (1-y)*log(1-p))
        term1 = target * Tensor(np.log(pred_clipped.data))
        term2 = (Tensor(1.0) - target) * Tensor(np.log(1 - pred_clipped.data))
        return -(term1 + term2).mean()

    def __repr__(self):
        return "BCELoss()"


def softmax(x, axis=-1):
    """
    Softmax function: converts logits to probabilities

    For numerical stability, we subtract the max value first.

    Args:
        x: Input tensor (logits)
        axis: Axis along which to compute softmax

    Returns:
        Probabilities that sum to 1 along the specified axis
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    return Tensor(exp_x / np.sum(exp_x, axis=axis, keepdims=True))
