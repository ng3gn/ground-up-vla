"""
TorchLite Tensor - A simple tensor with automatic differentiation

This is a minimal implementation to understand how autograd works.
Every operation builds a computation graph that we can traverse backwards
to compute gradients.
"""

import numpy as np
from typing import Optional, List, Tuple, Callable


class Tensor:
    """
    A tensor that tracks operations for automatic differentiation.

    Think of this as a box that contains:
    1. Data (the actual numbers)
    2. Gradient (how much changing this affects the final output)
    3. History (what operation created this tensor)
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        """
        Create a new tensor.

        Args:
            data: The actual numbers (can be a number, list, or numpy array)
            requires_grad: Whether to compute gradients for this tensor
            _children: Parent tensors in the computation graph (internal)
            _op: The operation that created this tensor (internal)
        """
        # Convert data to numpy array for easy math
        if isinstance(data, (int, float)):
            data = np.array([data])
        elif isinstance(data, list):
            data = np.array(data)

        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad

        # Gradient starts as None, gets filled during backward pass
        self.grad = None

        # For building the computation graph
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set(_children)     # Parent tensors
        self._op = _op                  # Operation name (for debugging/visualization)

    @property
    def shape(self):
        """Return the shape of the tensor"""
        return self.data.shape

    def __repr__(self):
        """String representation for debugging"""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        """Reset gradients to zero (call this before each backward pass)"""
        self.grad = np.zeros_like(self.data)

    # =========================
    # BASIC OPERATIONS
    # =========================

    def __add__(self, other):
        """
        Addition: out = self + other

        During backward pass:
        - Gradient flows equally to both inputs
        - d(out)/d(self) = 1, so grad_self += grad_out
        - d(out)/d(other) = 1, so grad_other += grad_out
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                # Handle broadcasting: sum out dimensions that were broadcast
                grad = out.grad
                # Sum over dimensions that were added by broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                # Sum over dimensions that were broadcast (size 1 -> size n)
                for i, (dim, orig_dim) in enumerate(zip(grad.shape, self.data.shape)):
                    if orig_dim == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = out.grad
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, orig_dim) in enumerate(zip(grad.shape, other.data.shape)):
                    if orig_dim == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Element-wise multiplication: out = self * other

        During backward pass:
        - d(out)/d(self) = other
        - d(out)/d(other) = self
        - grad_self += other * grad_out
        - grad_other += self * grad_out
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, orig_dim) in enumerate(zip(grad.shape, self.data.shape)):
                    if orig_dim == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = self.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, orig_dim) in enumerate(zip(grad.shape, other.data.shape)):
                    if orig_dim == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        Matrix multiplication: out = self @ other

        During backward pass:
        - d(out)/d(self) = other.T
        - d(out)/d(other) = self.T
        - grad_self += grad_out @ other.T
        - grad_other += self.T @ grad_out
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Use swapaxes for batched matmul support (works for 2D, 3D, 4D)
                grad = out.grad @ np.swapaxes(other.data, -2, -1)
                # Handle broadcasting: sum out extra leading dims
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                self.grad += grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad = np.swapaxes(self.data, -2, -1) @ out.grad
                # Handle broadcasting: sum out extra leading dims
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                other.grad += grad

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: out = -self"""
        return self * -1

    def __sub__(self, other):
        """Subtraction: out = self - other"""
        return self + (-other)

    def __truediv__(self, other):
        """Division: out = self / other"""
        return self * (other ** -1)

    def __pow__(self, power):
        """
        Power: out = self ** power

        During backward pass:
        - d(x^n)/dx = n * x^(n-1)
        - grad_self += power * (self^(power-1)) * grad_out
        """
        assert isinstance(power, (int, float)), "Only scalar powers supported"
        out = Tensor(self.data ** power,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # Support reverse operations (e.g., 2 + tensor)
    __radd__ = __add__
    __rmul__ = __mul__

    def __rtruediv__(self, other):
        """Reverse division: other / self"""
        return Tensor(other) / self

    def __rsub__(self, other):
        """Reverse subtraction: other - self"""
        return Tensor(other) - self

    # =========================
    # ACTIVATION FUNCTIONS
    # =========================

    def relu(self):
        """
        ReLU activation: out = max(0, self)

        During backward pass:
        - d(ReLU(x))/dx = 1 if x > 0, else 0
        - grad_self += (self > 0) * grad_out
        """
        out = Tensor(np.maximum(0, self.data),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='ReLU')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoid activation: out = 1 / (1 + exp(-self))

        During backward pass:
        - d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        - grad_self += sigmoid(self) * (1 - sigmoid(self)) * grad_out
        """
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='Sigmoid')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Derivative: sig * (1 - sig)
                self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Tanh activation: out = tanh(self)

        During backward pass:
        - d(tanh(x))/dx = 1 - tanh(x)^2
        - grad_self += (1 - tanh(self)^2) * grad_out
        """
        t = np.tanh(self.data)
        out = Tensor(t,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='Tanh')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Derivative: 1 - tanh^2
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # =========================
    # REDUCTION OPERATIONS
    # =========================

    def sum(self, axis=None, keepdims=False):
        """
        Sum elements: out = sum(self)

        During backward pass:
        - Gradient broadcasts back to all inputs
        - grad_self += ones_like(self) * grad_out
        """
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                # Gradient of sum is 1 everywhere
                grad = out.grad

                # Reshape gradient to match original shape
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    # Broadcast to original shape
                    grad = np.broadcast_to(grad, self.data.shape)

                self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean of elements"""
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    # =========================
    # BACKPROPAGATION
    # =========================

    def backward(self):
        """
        Compute gradients for all tensors in the computation graph.

        This is the magic of automatic differentiation!
        We traverse the graph in reverse topological order and
        apply the chain rule at each step.
        """
        # Build topological order (children before parents)
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        # Initialize gradient of output to 1 (dL/dL = 1)
        self.grad = np.ones_like(self.data)

        # Go through graph in reverse and apply chain rule
        for tensor in reversed(topo):
            tensor._backward()

    # =========================
    # UTILITY METHODS
    # =========================

    def reshape(self, *shape):
        """
        Reshape the tensor with gradient support.

        During backward pass:
        - Gradient is reshaped back to original shape
        """
        out = Tensor(self.data.reshape(*shape),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape')
        original_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """
        Transpose (permute) axes of the tensor.

        During backward pass:
        - Gradient is transposed with inverse permutation
        """
        out = Tensor(self.data.transpose(*axes),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='transpose')
        # Compute inverse permutation for backward
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.transpose(*inv_axes)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """
        Softmax: converts logits to probabilities.

        During backward pass:
        - d_input = s * (d_out - sum(d_out * s, axis, keepdims=True))
        """
        # Numerical stability: subtract max
        x_max = np.max(self.data, axis=axis, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        s = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        out = Tensor(s, requires_grad=self.requires_grad,
                    _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Softmax backward: s * (d_out - sum(d_out * s, axis))
                dot = np.sum(out.grad * s, axis=axis, keepdims=True)
                self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def log(self):
        """
        Natural logarithm.

        During backward pass:
        - d_input = d_out / data
        """
        eps = 1e-12
        out = Tensor(np.log(self.data + eps),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad / (self.data + eps)

        out._backward = _backward
        return out

    def exp(self):
        """
        Exponential.

        During backward pass:
        - d_input = exp(data) * d_out
        """
        exp_data = np.exp(self.data)
        out = Tensor(exp_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += exp_data * out.grad

        out._backward = _backward
        return out

    def item(self):
        """Get the value as a Python scalar (for single-element tensors)"""
        return self.data.item()

    def numpy(self):
        """Get the underlying numpy array"""
        return self.data

    @staticmethod
    def randn(*shape, requires_grad=False):
        """Create a tensor with random normal values"""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape, requires_grad=False):
        """Create a tensor of zeros"""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        """Create a tensor of ones"""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
