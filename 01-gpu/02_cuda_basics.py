# Copyright (c) 2024-2025 Neeraj Gandhi
# Licensed under the MIT License - see LICENSE file for details

"""
02_cuda_basics.py - Basic GPU tensor operations

Learning objectives:
- Create tensors directly on GPU
- Move tensors between CPU and GPU
- Understand device specification
- Perform operations on GPU tensors

Run: python 02_cuda_basics.py
"""

import torch


def creating_gpu_tensors():
    """Different ways to create tensors on GPU."""
    print("=" * 60)
    print("Creating GPU Tensors")
    print("=" * 60)

    # Method 1: Specify device at creation
    print("\n1. Create directly on GPU with device='cuda':")
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f"   a = {a}")
    print(f"   a.device = {a.device}")

    # Method 2: Use .cuda() method
    print("\n2. Create on CPU, then move with .cuda():")
    b = torch.tensor([4.0, 5.0, 6.0])  # Created on CPU by default
    print(f"   b (CPU) device = {b.device}")
    b = b.cuda()
    print(f"   b (GPU) device = {b.device}")

    # Method 3: Use .to() method (more flexible)
    print("\n3. Use .to(device) for flexibility:")
    c = torch.tensor([7.0, 8.0, 9.0])
    c = c.to('cuda')  # Same as .cuda()
    print(f"   c.device = {c.device}")

    # Method 4: Use torch.cuda.FloatTensor (older style, less common now)
    print("\n4. CUDA tensor types (older style):")
    d = torch.cuda.FloatTensor([10.0, 11.0, 12.0])
    print(f"   d.device = {d.device}")

    # Creating common tensor types directly on GPU
    print("\n5. Common creation functions with device:")
    zeros = torch.zeros(3, device='cuda')
    ones = torch.ones(3, device='cuda')
    randn = torch.randn(3, device='cuda')
    arange = torch.arange(3, device='cuda', dtype=torch.float32)

    print(f"   zeros: {zeros}")
    print(f"   ones:  {ones}")
    print(f"   randn: {randn}")
    print(f"   arange: {arange}")


def moving_tensors():
    """Moving tensors between devices."""
    print("\n" + "=" * 60)
    print("Moving Tensors Between Devices")
    print("=" * 60)

    # Create on CPU
    cpu_tensor = torch.randn(3, 3)
    print(f"\nOriginal tensor on {cpu_tensor.device}:")
    print(cpu_tensor)

    # Move to GPU (creates a copy)
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"\nMoved to {gpu_tensor.device}:")
    print(gpu_tensor)

    # Move back to CPU (creates a copy)
    back_to_cpu = gpu_tensor.to('cpu')
    print(f"\nMoved back to {back_to_cpu.device}:")
    print(back_to_cpu)

    # Important: .to() creates a COPY, original is unchanged
    print("\n*** Important: .to() creates a COPY ***")
    print(f"cpu_tensor still on: {cpu_tensor.device}")
    print(f"gpu_tensor still on: {gpu_tensor.device}")

    # Converting to numpy (must be on CPU first)
    print("\n*** Converting to NumPy (requires CPU) ***")
    numpy_array = back_to_cpu.numpy()
    print(f"NumPy array:\n{numpy_array}")

    # This would fail:
    # gpu_tensor.numpy()  # RuntimeError: can't convert cuda tensor to numpy


def gpu_operations():
    """Performing operations on GPU tensors."""
    print("\n" + "=" * 60)
    print("GPU Operations")
    print("=" * 60)

    # All operations happen on GPU when tensors are on GPU
    a = torch.randn(3, 3, device='cuda')
    b = torch.randn(3, 3, device='cuda')

    print("\nBasic arithmetic (all on GPU):")
    print(f"a + b device: {(a + b).device}")
    print(f"a * b device: {(a * b).device}")
    print(f"a @ b device: {(a @ b).device}")  # Matrix multiply

    # Reduction operations
    print("\nReduction operations:")
    print(f"a.sum() = {a.sum().item():.4f} (device: {a.sum().device})")
    print(f"a.mean() = {a.mean().item():.4f}")
    print(f"a.max() = {a.max().item():.4f}")

    # In-place operations (modify tensor directly, no copy)
    print("\nIn-place operations (note the underscore):")
    c = torch.zeros(3, device='cuda')
    print(f"Before: {c}")
    c.add_(1)  # In-place add
    print(f"After c.add_(1): {c}")
    c.mul_(2)  # In-place multiply
    print(f"After c.mul_(2): {c}")


def device_mismatch_errors():
    """Understanding device mismatch errors."""
    print("\n" + "=" * 60)
    print("Device Mismatch Errors")
    print("=" * 60)

    cpu_tensor = torch.randn(3)
    gpu_tensor = torch.randn(3, device='cuda')

    print(f"\ncpu_tensor device: {cpu_tensor.device}")
    print(f"gpu_tensor device: {gpu_tensor.device}")

    print("\nAttempting: cpu_tensor + gpu_tensor")
    try:
        result = cpu_tensor + gpu_tensor
    except RuntimeError as e:
        print(f"ERROR: {e}")

    print("\nSolution: Move tensors to same device first")
    result = cpu_tensor.cuda() + gpu_tensor
    print(f"Result device: {result.device}")
    print(f"Result: {result}")


def specific_gpu_selection():
    """Selecting specific GPU (for multi-GPU systems)."""
    print("\n" + "=" * 60)
    print("GPU Selection (for multi-GPU systems)")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {num_gpus}")

    if num_gpus == 1:
        print("(You have 1 GPU, so all examples use cuda:0)")

    # Specific GPU by index
    print("\nSelecting specific GPU:")
    tensor_gpu0 = torch.randn(3, device='cuda:0')
    print(f"tensor on cuda:0: {tensor_gpu0.device}")

    # If you had multiple GPUs:
    # tensor_gpu1 = torch.randn(3, device='cuda:1')

    # Setting default device
    print("\nSetting default CUDA device:")
    print(f"Current default: cuda:{torch.cuda.current_device()}")

    # You can change default with:
    # torch.cuda.set_device(0)


def main():
    print("\n" + "=" * 60)
    print("   CUDA BASICS")
    print("=" * 60)

    creating_gpu_tensors()
    moving_tensors()
    gpu_operations()
    device_mismatch_errors()
    specific_gpu_selection()

    print("\n" + "=" * 60)
    print("   KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Use device='cuda' when creating tensors for GPU
2. Use .to('cuda') or .cuda() to move existing tensors
3. .to() creates a COPY - original tensor unchanged
4. All tensors in an operation must be on same device
5. Use .cpu() or .to('cpu') before converting to NumPy
6. In-place operations (with _) modify tensor directly
""")


if __name__ == "__main__":
    main()
