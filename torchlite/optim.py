"""
TorchLite Optimizers

Optimizers update parameters based on their gradients.
The goal: minimize the loss function by adjusting weights.
"""

import numpy as np


class Optimizer:
    """Base class for all optimizers"""

    def __init__(self, parameters):
        """
        Args:
            parameters: List of tensors to optimize
        """
        self.parameters = list(parameters)

    def zero_grad(self):
        """Reset gradients to zero before backward pass"""
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        """Update parameters (must be implemented by subclass)"""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    Updates weights by moving in the opposite direction of the gradient:
        weight = weight - learning_rate * gradient

    This is the simplest and most fundamental optimizer!

    Args:
        parameters: List of tensors to optimize
        lr: Learning rate (how big of a step to take)
        momentum: Momentum factor (default 0 = no momentum)
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum

        # Track velocity for momentum (one for each parameter)
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """
        Update all parameters using their gradients.

        Without momentum:
            param = param - lr * grad

        With momentum (smoother updates):
            velocity = momentum * velocity - lr * grad
            param = param + velocity
        """
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            if self.momentum == 0:
                # Simple SGD: step in opposite direction of gradient
                param.data -= self.lr * param.grad
            else:
                # SGD with momentum: accumulate velocity
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                param.data += self.velocities[i]

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation

    Combines ideas from momentum and RMSprop:
    - Maintains running averages of gradients (momentum)
    - Maintains running averages of squared gradients (adaptive learning rate)

    Usually works better than SGD with less tuning!

    Args:
        parameters: List of tensors to optimize
        lr: Learning rate (default 0.001)
        beta1: Decay rate for first moment (default 0.9)
        beta2: Decay rate for second moment (default 0.999)
        eps: Small constant for numerical stability (default 1e-8)
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # First moment (mean of gradients)
        self.m = [np.zeros_like(p.data) for p in self.parameters]

        # Second moment (mean of squared gradients)
        self.v = [np.zeros_like(p.data) for p in self.parameters]

        # Time step (for bias correction)
        self.t = 0

    def step(self):
        """
        Update all parameters using adaptive learning rates.

        Algorithm:
            1. Update biased first moment:  m = beta1*m + (1-beta1)*grad
            2. Update biased second moment: v = beta2*v + (1-beta2)*grad^2
            3. Correct bias: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
            4. Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            # Update first moment (running mean of gradients)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update second moment (running mean of squared gradients)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction (moments start at 0, need to correct early on)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters with adaptive learning rate
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
