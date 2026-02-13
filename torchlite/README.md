# TorchLite Framework Documentation

A minimal neural network framework built from scratch for educational purposes.

## Overview

TorchLite is a ~1,200 line implementation of core neural network functionality,
including automatic differentiation, common layers, optimizers, and
visualization tools. It's designed to be readable and understandable, not
optimized for performance.

**Design Goals:**
- Every line should be understandable by someone learning neural networks
- No external ML libraries (only numpy, matplotlib, networkx)
- Full automatic differentiation like PyTorch

## Architecture

### Core Modules

#### `tensor.py` (~410 lines)
The foundation of TorchLite. Implements tensors with automatic differentiation.

**Key Classes:**
- `Tensor` - Wraps numpy arrays with gradient tracking

**Features:**
- Automatic gradient computation (autograd)
- Computation graph construction
- Backward pass implementation
- Operations: add, multiply, matmul, power, sum, mean
- Activations: ReLU, Sigmoid, Tanh
- Tracks operation history for backpropagation

**Example:**
```python
from torchlite import Tensor

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
w = Tensor([0.5, -0.3, 0.8], requires_grad=True)

# Forward pass builds computation graph
y = (x * w).sum()

# Backward pass computes gradients
y.backward()

print(x.grad)  # Gradients w.r.t. x
print(w.grad)  # Gradients w.r.t. w
```

**Tensor Operations:**
```python
# Element-wise operations
z = x + y        # Addition
z = x - y        # Subtraction
z = x * y        # Multiplication
z = x / y        # Division
z = x ** 2       # Power

# Matrix operations
z = x @ w        # Matrix multiplication

# Reductions
z = x.sum()      # Sum all elements
z = x.mean()     # Mean of all elements

# Activations
z = x.relu()     # ReLU activation
z = x.sigmoid()  # Sigmoid activation
z = x.tanh()     # Tanh activation
```

#### `nn.py` (~270 lines)
Neural network building blocks: layers, activations, and loss functions.

**Base Classes:**
- `Module` - Base class for all neural network modules

**Layers:**
- `Linear(in_features, out_features)` - Fully connected layer
- `Sequential(*layers)` - Container for sequential layers

**Activations:**
- `ReLU()` - Rectified Linear Unit
- `Sigmoid()` - Sigmoid activation
- `Tanh()` - Hyperbolic tangent

**Loss Functions:**
- `MSELoss()` - Mean Squared Error
- `BCELoss()` - Binary Cross Entropy

**Functions:**
- `softmax(x, dim=-1)` - Softmax function

**Example:**
```python
from torchlite import Tensor
from torchlite.nn import Linear, ReLU, Sequential, MSELoss

# Build a network
model = Sequential(
    Linear(2, 8),   # Input layer: 2 -> 8
    ReLU(),
    Linear(8, 4),   # Hidden layer: 8 -> 4
    ReLU(),
    Linear(4, 1)    # Output layer: 4 -> 1
)

# Forward pass
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
output = model(x)

# Compute loss
target = Tensor([[3.0], [7.0]])
loss_fn = MSELoss()
loss = loss_fn(output, target)

# Get model parameters
params = model.parameters()  # List of all trainable tensors
```

**Linear Layer Details:**
```python
# What happens inside Linear(2, 3):
# 1. Initializes weight matrix (3, 2) with random values
# 2. Initializes bias vector (3,) with zeros
# 3. Forward pass: output = input @ weight.T + bias

layer = Linear(2, 3)
x = Tensor([[1.0, 2.0]])  # Shape: (1, 2)
y = layer(x)               # Shape: (1, 3)
# Computation: y = x @ W^T + b
#              (1,3) = (1,2) @ (2,3) + (3,)
```

#### `optim.py` (~150 lines)
Optimization algorithms for training neural networks.

**Optimizers:**
- `SGD(params, lr, momentum=0)` - Stochastic Gradient Descent
- `Adam(params, lr, betas=(0.9, 0.999), eps=1e-8)` - Adam optimizer

**Example:**
```python
from torchlite.optim import SGD, Adam

# SGD with momentum
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model(x)
    loss = loss_fn(pred, target)

    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
```

**Optimizer Details:**

*SGD:*
```python
# Standard SGD: param = param - lr * grad
# With momentum: velocity = momentum * velocity - lr * grad
#                param = param + velocity
```

*Adam:*
```python
# Adaptive learning rates per parameter
# Combines momentum with per-parameter learning rates
# Generally converges faster than SGD
```

#### `visualize.py` (~310 lines)
Visualization utilities for understanding neural networks.

**Functions:**
- `plot_computation_graph(tensor, filename=None)` - Visualize autograd graph
- `plot_neuron(weight, bias, x_range, activation, filename)` - Single neuron behavior
- `plot_network_architecture(layer_sizes, filename)` - Network diagram
- `plot_decision_boundary(model, X, y, title, filename)` - Classification boundaries
- `plot_training_history(losses, title, filename)` - Loss curves

**Example:**
```python
from torchlite.visualize import (
    plot_computation_graph,
    plot_network_architecture,
    plot_training_history
)

# Visualize computation graph
loss = model(x).mean()
plot_computation_graph(loss, filename='graph.png')

# Plot network architecture
plot_network_architecture([2, 8, 4, 1], filename='architecture.png')

# Show training progress
losses = []
for epoch in range(100):
    # ... training code ...
    losses.append(loss.item())

plot_training_history(losses, filename='training.png')
```

#### `logger.py` (~290 lines)
Text logging system for training and inference.

**Classes:**
- `TrainingLogger(name, log_dir='.')` - Log training runs
- `InferenceLogger(name, log_dir='.')` - Log inference runs

**Features:**
- Timestamps for each epoch/step
- Logs input/output shapes and values
- Records gradients and parameter updates
- Pretty-printed format
- Append mode (keeps history)

**Example:**
```python
from torchlite.logger import TrainingLogger

logger = TrainingLogger(name="my_model", log_dir="outputs")

for epoch in range(n_epochs):
    # Forward pass
    pred = model(x)
    loss = loss_fn(pred, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log every 10 epochs
    if epoch % 10 == 0:
        logger.log_epoch(
            epoch=epoch,
            input_data=x,
            output=pred,
            loss=loss,
            parameters={
                'weight': (model.layers[0].weight, model.layers[0].weight.grad),
                'bias': (model.layers[0].bias, model.layers[0].bias.grad)
            }
        )
```

**Log Format:**
```
================================================================================
RUN: 2026-02-01 23:37:39
================================================================================

[EPOCH 0 - FORWARD PASS]
Timestamp: 2026-02-01 23:37:39.500

Input:
  Shape: (3, 1)
  Values: [1.000000, 2.000000, 3.000000]

Output:
  Shape: (1,)
  Values: [1.351878]

[EPOCH 0 - BACKWARD PASS]
Timestamp: 2026-02-01 23:37:39.500

Loss: 18.686244

Parameters and Gradients:

  weight:
    Shape: (1, 1)
    Values: [0.496714]
    Gradient shape: (1, 1)
    Gradient values: [-18.583725]
```

## Complete Training Example

Here's a complete example putting it all together:

```python
import numpy as np
from torchlite import Tensor
from torchlite.nn import Linear, ReLU, Sequential, MSELoss
from torchlite.optim import Adam
from torchlite.logger import TrainingLogger
from torchlite.visualize import plot_training_history

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).reshape(-1, 1)

# Convert to tensors
X = Tensor(X_train)
y = Tensor(y_train)

# Build model
model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 1)
)

# Set up training
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = MSELoss()
logger = TrainingLogger(name="regression_model")

# Training loop
losses = []
n_epochs = 200

print("Training...")
for epoch in range(n_epochs):
    # Forward pass
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record
    losses.append(loss.item())

    # Log and print progress
    if epoch % 20 == 0:
        logger.log_epoch(epoch, X, pred, loss,
                        parameters={'layer0_weight': (model.layers[0].weight,
                                                     model.layers[0].weight.grad)})
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Visualize training
plot_training_history(losses, title="Training Loss",
                     filename="outputs/training.png")

print("Done! Check outputs/ for logs and plots.")
```

## Implementation Details

### Automatic Differentiation

TorchLite implements reverse-mode automatic differentiation (backpropagation):

1. **Forward Pass:** Build computation graph
   - Each operation creates a new Tensor
   - Store operation type and parent tensors
   - Store backward function for gradient computation

2. **Backward Pass:** Traverse graph in reverse
   - Start from output tensor
   - Apply chain rule at each node
   - Accumulate gradients at leaf nodes

**Example:**
```python
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Forward: build graph
z = x * y          # z = 6.0, stores: op='mul', parents=(x,y)
w = z + x          # w = 8.0, stores: op='add', parents=(z,x)

# Backward: compute gradients
w.backward()       # Starts with grad=1.0 at w

# Chain rule applied automatically:
# dw/dz = 1.0, dw/dx = 1.0
# dz/dx = y = 3.0, dz/dy = x = 2.0
# Final: x.grad = 4.0, y.grad = 2.0
```

### Gradient Accumulation

Gradients accumulate across multiple backward passes:

```python
x = Tensor([1.0], requires_grad=True)

# First backward
y1 = x * 2
y1.backward()
print(x.grad)  # [2.0]

# Second backward (accumulates!)
y2 = x * 3
y2.backward()
print(x.grad)  # [5.0] = 2.0 + 3.0

# Must zero gradients between iterations
x.zero_grad()
print(x.grad)  # [0.0]
```

### Parameter Management

The `Module` class provides parameter management:

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(randn(out_features, in_features),
                           requires_grad=True)
        self.bias = Tensor(zeros(out_features),
                         requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]
```

Sequential models aggregate parameters:

```python
model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
params = model.parameters()  # Returns [layer1.weight, layer1.bias,
                             #          layer2.weight, layer2.bias]
```

## Limitations

TorchLite is educational software with intentional limitations:

**Performance:**
- Uses Python loops (no vectorization)
- No GPU support
- No batch optimization
- Slow compared to PyTorch/TensorFlow

**Features:**
- Limited operation set
- No convolutions or recurrent layers
- No advanced optimizers (RMSprop, AdaGrad, etc.)
- No learning rate scheduling
- No regularization (dropout, weight decay, etc.)

**Use Cases:**
- ✅ Learning how neural networks work
- ✅ Understanding autograd internals
- ✅ Prototyping simple networks
- ✅ Educational demonstrations
- ❌ Production ML systems
- ❌ Large-scale training
- ❌ Real-time inference

## Comparison to PyTorch

| Feature | TorchLite | PyTorch |
|---------|-----------|---------|
| Automatic differentiation | ✅ | ✅ |
| GPU support | ❌ | ✅ |
| Readable source | ✅ | ❌ |
| Production ready | ❌ | ✅ |
| Lines of code | ~1,200 | ~1,000,000+ |
| Dependencies | numpy | cuda, cudnn, etc. |
| Learning curve | Gentle | Steep |

**When to use TorchLite:**
- You're learning neural networks for the first time
- You want to understand how autograd works
- You need to see every implementation detail
- You're teaching neural networks

**When to use PyTorch:**
- You're building real ML systems
- You need GPU acceleration
- You need advanced features
- You're working with large models

## Testing

Basic tests can be run to verify functionality:

```python
# Test autograd
x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * 2).sum()
y.backward()
assert np.allclose(x.grad, [2.0, 2.0])

# Test linear layer
layer = Linear(2, 3)
x = Tensor([[1.0, 2.0]])
y = layer(x)
assert y.data.shape == (1, 3)

# Test optimizer
params = [Tensor([1.0], requires_grad=True)]
opt = SGD(params, lr=0.1)
params[0].grad = Tensor([1.0])
opt.step()
assert params[0].data[0] == 0.9  # 1.0 - 0.1 * 1.0
```

## Contributing

TorchLite is intentionally minimal. If you find bugs or have suggestions for clearer implementations, contributions are welcome. Keep in mind:

- Educational clarity is the priority
- Keep it simple and readable
- Every line should be understandable
- Document everything thoroughly

## License

This is educational software intended for learning purposes.

## Acknowledgments

TorchLite's design is inspired by:
- PyTorch's API design
- Andrej Karpathy's micrograd
- The Feynman principle: "What I cannot create, I do not understand"

Built with love for learners by learners! 🚀
