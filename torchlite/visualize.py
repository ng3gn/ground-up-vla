"""
TorchLite Visualization Utilities

Make neural networks visible and understandable!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patches
import networkx as nx


def plot_computation_graph(tensor, filename=None):
    """
    Visualize the computation graph that created a tensor.

    Shows how operations are connected and which tensors have gradients.

    Args:
        tensor: The output tensor to trace back from
        filename: If provided, save to this file instead of showing
    """
    G = nx.DiGraph()
    visited = set()

    def build_graph(t, parent_id=None):
        """Recursively build the graph"""
        t_id = id(t)

        if t_id in visited:
            return t_id

        visited.add(t_id)

        # Add node for this tensor
        label = f"{t._op if t._op else 'Input'}\n{t.data.shape}"
        color = 'lightblue' if t.requires_grad else 'lightgray'
        G.add_node(t_id, label=label, color=color)

        # Add edges from parents
        for parent in t._prev:
            parent_id = build_graph(parent, t_id)
            G.add_edge(parent_id, t_id)

        return t_id

    build_graph(tensor)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Get node colors and labels
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}

    # Draw
    nx.draw(G, pos, labels=labels, node_color=colors,
            node_size=3000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray',
            arrowstyle='->', connectionstyle='arc3,rad=0.1')

    plt.title("Computation Graph\n(Blue = gradients tracked, Gray = no gradients)",
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved computation graph to {filename}")
    else:
        plt.show()

    plt.close()


def plot_neuron(weight, bias, x_range=(-5, 5), activation=None, filename=None):
    """
    Visualize a single neuron's behavior.

    Shows how the neuron transforms inputs to outputs.

    Args:
        weight: Neuron's weight
        bias: Neuron's bias
        x_range: Range of x values to plot
        activation: Activation function name ('relu', 'sigmoid', 'tanh', or None)
        filename: If provided, save to this file
    """
    x = np.linspace(x_range[0], x_range[1], 200)

    # Linear transformation
    y_linear = weight * x + bias

    # Apply activation if specified
    if activation is None:
        y = y_linear
        title = f"Neuron: y = {weight:.2f}*x + {bias:.2f}"
    elif activation == 'relu':
        y = np.maximum(0, y_linear)
        title = f"Neuron: y = ReLU({weight:.2f}*x + {bias:.2f})"
    elif activation == 'sigmoid':
        y = 1 / (1 + np.exp(-y_linear))
        title = f"Neuron: y = Sigmoid({weight:.2f}*x + {bias:.2f})"
    elif activation == 'tanh':
        y = np.tanh(y_linear)
        title = f"Neuron: y = Tanh({weight:.2f}*x + {bias:.2f})"
    else:
        y = y_linear
        title = f"Neuron: y = {weight:.2f}*x + {bias:.2f}"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot linear transformation (before activation)
    if activation is not None:
        ax.plot(x, y_linear, 'gray', linestyle='--', alpha=0.5, label='Before activation')

    # Plot final output
    ax.plot(x, y, 'blue', linewidth=2, label='Output')

    # Highlight zero crossing
    if activation == 'relu':
        zero_x = -bias / weight if weight != 0 else 0
        ax.axvline(zero_x, color='red', linestyle=':', alpha=0.5, label='Activation threshold')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Input (x)', fontsize=12)
    ax.set_ylabel('Output (y)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved neuron plot to {filename}")
    else:
        plt.show()

    plt.close()


def plot_network_architecture(layer_sizes, filename=None):
    """
    Visualize the architecture of a neural network.

    Args:
        layer_sizes: List of integers, e.g., [2, 4, 3, 1] for 2→4→3→1 network
        filename: If provided, save to this file
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    # Spacing
    layer_spacing = 1.0 / (n_layers - 1) if n_layers > 1 else 0.5
    neuron_radius = 0.02

    # Draw neurons
    neuron_positions = {}

    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_idx * layer_spacing

        # Center neurons vertically
        neuron_spacing = 0.8 / max(n_neurons - 1, 1) if n_neurons > 1 else 0
        y_start = 0.5 - (n_neurons - 1) * neuron_spacing / 2

        for neuron_idx in range(n_neurons):
            y = y_start + neuron_idx * neuron_spacing

            # Draw neuron
            circle = plt.Circle((x, y), neuron_radius, color='lightblue', ec='black', zorder=4)
            ax.add_patch(circle)

            neuron_positions[(layer_idx, neuron_idx)] = (x, y)

    # Draw connections
    for layer_idx in range(n_layers - 1):
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx + 1]):
                x1, y1 = neuron_positions[(layer_idx, i)]
                x2, y2 = neuron_positions[(layer_idx + 1, j)]

                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, zorder=1)

    # Add layer labels
    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_idx * layer_spacing
        if layer_idx == 0:
            label = f"Input\n({n_neurons})"
        elif layer_idx == n_layers - 1:
            label = f"Output\n({n_neurons})"
        else:
            label = f"Hidden {layer_idx}\n({n_neurons})"

        ax.text(x, -0.1, label, ha='center', va='top', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.0)
    ax.axis('off')
    ax.set_aspect('equal')

    plt.title(f"Network Architecture: {' → '.join(map(str, layer_sizes))}",
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved network architecture to {filename}")
    else:
        plt.show()

    plt.close()


def plot_decision_boundary(model, X, y, title="Decision Boundary", filename=None):
    """
    Visualize how a model classifies 2D data.

    Shows the decision boundary and training points.

    Args:
        model: Neural network model
        X: Input data (N, 2) - must be 2D!
        y: Labels (N,)
        title: Plot title
        filename: If provided, save to this file
    """
    from .tensor import Tensor

    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    mesh_input = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in mesh_input:
        pred = model(Tensor(point))
        if hasattr(pred, 'data'):
            pred = pred.data
        if pred.shape == ():
            pred = np.array([pred])
        Z.append(pred[0] if len(pred) == 1 else np.argmax(pred))

    Z = np.array(Z).reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

    # Training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                        edgecolors='black', s=100, linewidth=1.5)

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved decision boundary to {filename}")
    else:
        plt.show()

    plt.close()


def plot_training_history(losses, title="Training Loss", filename=None):
    """
    Plot the loss over training iterations.

    Args:
        losses: List of loss values
        title: Plot title
        filename: If provided, save to this file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses, linewidth=2, color='blue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at final loss
    ax.axhline(losses[-1], color='red', linestyle='--', alpha=0.5,
               label=f'Final: {losses[-1]:.4f}')
    ax.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {filename}")
    else:
        plt.show()

    plt.close()
