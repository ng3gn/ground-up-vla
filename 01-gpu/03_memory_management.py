# Copyright (c) 2024-2025 Neeraj Gandhi
# Licensed under the MIT License - see LICENSE file for details

"""
03_memory_management.py - Understanding and managing GPU memory

Learning objectives:
- Monitor GPU memory usage
- Understand allocated vs reserved memory
- Clear memory when needed
- Estimate if a model will fit in VRAM

Run: python 03_memory_management.py

Your GPU: GTX 1650 with 3.6 GB VRAM
"""

import torch
import gc


def bytes_to_mb(bytes_val):
    """Convert bytes to megabytes."""
    return bytes_val / 1024 / 1024


def bytes_to_gb(bytes_val):
    """Convert bytes to gigabytes."""
    return bytes_val / 1024 / 1024 / 1024


def print_memory_stats(label=""):
    """Print current GPU memory statistics."""
    if label:
        print(f"\n--- {label} ---")

    # Total memory on GPU
    total = torch.cuda.get_device_properties(0).total_memory

    # Memory currently held by PyTorch's caching allocator
    reserved = torch.cuda.memory_reserved(0)

    # Memory actually used by tensors
    allocated = torch.cuda.memory_allocated(0)

    # Free memory (total - reserved by PyTorch)
    free = total - reserved

    # Cached but not used (reserved - allocated)
    cached = reserved - allocated

    print(f"  Total:     {bytes_to_gb(total):.2f} GB")
    print(f"  Reserved:  {bytes_to_mb(reserved):.1f} MB (held by PyTorch)")
    print(f"  Allocated: {bytes_to_mb(allocated):.1f} MB (used by tensors)")
    print(f"  Cached:    {bytes_to_mb(cached):.1f} MB (reserved but unused)")
    print(f"  Free:      {bytes_to_gb(free):.2f} GB")


def memory_allocation_demo():
    """Demonstrate how memory is allocated and freed."""
    print("=" * 60)
    print("Memory Allocation Demo")
    print("=" * 60)

    print_memory_stats("Initial state")

    # Allocate a tensor
    print("\n>>> Creating 100MB tensor (25M floats * 4 bytes)")
    tensor_100mb = torch.randn(25_000_000, device='cuda')
    print_memory_stats("After 100MB allocation")

    # Allocate more
    print("\n>>> Creating another 200MB tensor")
    tensor_200mb = torch.randn(50_000_000, device='cuda')
    print_memory_stats("After 200MB allocation")

    # Delete reference (but memory stays reserved!)
    print("\n>>> Deleting tensor_100mb reference")
    del tensor_100mb
    print_memory_stats("After del (memory still reserved)")

    # Force garbage collection
    print("\n>>> Running gc.collect() and torch.cuda.empty_cache()")
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_stats("After cache clear")

    # Clean up
    del tensor_200mb
    gc.collect()
    torch.cuda.empty_cache()


def memory_clearing_techniques():
    """Different ways to free GPU memory."""
    print("\n" + "=" * 60)
    print("Memory Clearing Techniques")
    print("=" * 60)

    # Create some tensors
    tensors = [torch.randn(10_000_000, device='cuda') for _ in range(5)]
    print_memory_stats("After creating 5 tensors (200MB total)")

    print("\nTechnique 1: Delete references")
    del tensors[0]
    print_memory_stats("After del tensors[0]")

    print("\nTechnique 2: Set to None")
    tensors[1] = None
    print_memory_stats("After tensors[1] = None")

    print("\nTechnique 3: Clear list")
    tensors.clear()
    print_memory_stats("After tensors.clear()")

    print("\nTechnique 4: Garbage collect + empty cache")
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_stats("After gc + empty_cache")

    print("\n*** Key insight: del/None don't immediately free memory ***")
    print("*** PyTorch caches memory for reuse. Use empty_cache() to release. ***")


def estimate_memory_needs():
    """Estimate memory requirements for different operations."""
    print("\n" + "=" * 60)
    print("Memory Estimation Guide")
    print("=" * 60)

    total_gb = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
    print(f"\nYour GPU has {total_gb:.2f} GB VRAM")

    print("\n--- Memory per data type ---")
    dtypes = [
        ('float32', torch.float32, 4),
        ('float16', torch.float16, 2),
        ('bfloat16', torch.bfloat16, 2),
        ('int32', torch.int32, 4),
        ('int8', torch.int8, 1),
    ]

    for name, dtype, bytes_per in dtypes:
        elements_1gb = 1024**3 // bytes_per
        print(f"  {name}: {bytes_per} bytes/element -> {elements_1gb/1e9:.2f}B elements per GB")

    print("\n--- Common model sizes (rough estimates) ---")
    models = [
        ("ResNet-50", 25, "~100 MB"),
        ("ViT-Base", 86, "~350 MB"),
        ("SmolVLM (500M)", 500, "~2 GB"),
        ("SmolVLA Action Expert", 50, "~200 MB"),
        ("OpenVLA (7B)", 7000, "~28 GB (won't fit!)"),
    ]

    print(f"  {'Model':<25} {'Params':<10} {'Size (f32)':<15} {'Fits?'}")
    print("  " + "-" * 55)
    for name, params_m, size in models:
        fits = "Yes" if params_m * 4 / 1000 < total_gb else "No"
        print(f"  {name:<25} {params_m}M{'':<5} {size:<15} {fits}")

    print("\n--- Training vs Inference ---")
    print("""
  Inference: Need memory for model + activations + input
  Training:  Need 2-4x more for gradients and optimizer states

  For your 3.6GB GPU:
  - Inference: Models up to ~1.5-2GB work comfortably
  - Training: Models up to ~500MB work, larger need tricks

  Tricks for limited VRAM:
  - Use float16/bfloat16 (half the memory)
  - Gradient checkpointing (trade compute for memory)
  - Smaller batch sizes
  - CPU offloading for optimizer states
""")


def will_it_fit(param_count_millions, dtype=torch.float32, include_gradients=False):
    """Check if a model will fit in GPU memory."""
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()

    # Model parameters
    model_bytes = param_count_millions * 1e6 * bytes_per_param

    # Gradients (same size as model)
    grad_bytes = model_bytes if include_gradients else 0

    # Optimizer states (Adam uses 2x for momentum and variance)
    optimizer_bytes = 2 * model_bytes if include_gradients else 0

    total_bytes = model_bytes + grad_bytes + optimizer_bytes
    available = torch.cuda.get_device_properties(0).total_memory * 0.9  # 90% usable

    return total_bytes < available, bytes_to_gb(total_bytes)


def memory_check_demo():
    """Demo the will_it_fit function."""
    print("\n" + "=" * 60)
    print("Will It Fit? Calculator")
    print("=" * 60)

    test_cases = [
        (100, torch.float32, False, "100M params, f32, inference"),
        (100, torch.float32, True, "100M params, f32, training"),
        (100, torch.float16, True, "100M params, f16, training"),
        (500, torch.float16, False, "500M params, f16, inference"),
        (500, torch.float16, True, "500M params, f16, training"),
    ]

    print(f"\n{'Scenario':<35} {'Size':<10} {'Fits?'}")
    print("-" * 50)

    for params, dtype, training, desc in test_cases:
        fits, size_gb = will_it_fit(params, dtype, training)
        print(f"{desc:<35} {size_gb:.2f} GB    {'Yes' if fits else 'No'}")


def main():
    print("\n" + "=" * 60)
    print("   GPU MEMORY MANAGEMENT")
    print("=" * 60)

    memory_allocation_demo()
    memory_clearing_techniques()
    estimate_memory_needs()
    memory_check_demo()

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("   KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. PyTorch caches GPU memory - use empty_cache() to release
2. 'Reserved' memory is held by PyTorch, 'allocated' is in use
3. float16 uses half the memory of float32
4. Training needs 3-4x more memory than inference
5. Monitor with torch.cuda.memory_allocated/reserved()
6. For limited VRAM: smaller batches, fp16, gradient checkpointing
""")


if __name__ == "__main__":
    main()
