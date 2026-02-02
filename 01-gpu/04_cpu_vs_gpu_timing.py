# Copyright (c) 2024-2025 Neeraj Gandhi
# Licensed under the MIT License - see LICENSE file for details

"""
04_cpu_vs_gpu_timing.py - Compare CPU vs GPU performance

Learning objectives:
- Understand when GPUs provide speedup
- Learn proper benchmarking with CUDA synchronization
- See the impact of data transfer overhead
- Identify GPU-friendly vs GPU-unfriendly operations

Run: python 04_cpu_vs_gpu_timing.py
"""

import torch
import time


def benchmark(func, warmup=3, runs=10):
    """Benchmark a function with proper warmup and averaging."""
    # Warmup runs (not timed)
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(runs):
        # Synchronize GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        func()

        # Synchronize GPU after operation
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def matrix_multiply_comparison():
    """Compare matrix multiplication on CPU vs GPU."""
    print("=" * 60)
    print("Matrix Multiplication Comparison")
    print("=" * 60)

    sizes = [100, 500, 1000, 2000, 4000]

    print(f"\n{'Size':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 45)

    for n in sizes:
        # Create matrices
        a_cpu = torch.randn(n, n)
        b_cpu = torch.randn(n, n)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # Benchmark CPU
        cpu_time = benchmark(lambda: torch.matmul(a_cpu, b_cpu)) * 1000

        # Benchmark GPU
        gpu_time = benchmark(lambda: torch.matmul(a_gpu, b_gpu)) * 1000

        speedup = cpu_time / gpu_time
        print(f"{n}x{n:<6} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:.1f}x")

    print("\n*** GPU excels at large matrix operations ***")


def small_operations_comparison():
    """Show where GPU overhead hurts performance."""
    print("\n" + "=" * 60)
    print("Small Operations (GPU overhead matters)")
    print("=" * 60)

    sizes = [10, 50, 100, 500]

    print(f"\n{'Size':<10} {'CPU (us)':<12} {'GPU (us)':<12} {'Winner':<10}")
    print("-" * 45)

    for n in sizes:
        a_cpu = torch.randn(n, n)
        b_cpu = torch.randn(n, n)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # Benchmark (convert to microseconds)
        cpu_time = benchmark(lambda: a_cpu + b_cpu) * 1_000_000
        gpu_time = benchmark(lambda: a_gpu + b_gpu) * 1_000_000

        winner = "GPU" if gpu_time < cpu_time else "CPU"
        print(f"{n}x{n:<6} {cpu_time:<12.1f} {gpu_time:<12.1f} {winner}")

    print("\n*** For small tensors, CPU can be faster due to GPU overhead ***")


def data_transfer_overhead():
    """Demonstrate the cost of moving data between CPU and GPU."""
    print("\n" + "=" * 60)
    print("Data Transfer Overhead")
    print("=" * 60)

    sizes_mb = [1, 10, 50, 100, 500]

    print(f"\n{'Size (MB)':<12} {'CPU->GPU (ms)':<15} {'GPU->CPU (ms)':<15} {'Bandwidth':<12}")
    print("-" * 55)

    for size_mb in sizes_mb:
        # Create tensor (float32 = 4 bytes per element)
        num_elements = size_mb * 1024 * 1024 // 4
        cpu_tensor = torch.randn(num_elements)
        gpu_tensor = cpu_tensor.cuda()

        # Time CPU -> GPU
        to_gpu_time = benchmark(lambda: cpu_tensor.cuda(), warmup=2, runs=5) * 1000

        # Time GPU -> CPU
        to_cpu_time = benchmark(lambda: gpu_tensor.cpu(), warmup=2, runs=5) * 1000

        # Calculate bandwidth (GB/s)
        bandwidth = size_mb / to_gpu_time  # MB / ms = GB/s

        print(f"{size_mb:<12} {to_gpu_time:<15.2f} {to_cpu_time:<15.2f} {bandwidth:.1f} GB/s")

    print("""
*** Key insight: Data transfer is EXPENSIVE ***

If you constantly move data between CPU and GPU, you lose all speedup!

Good pattern:
  1. Move data to GPU once at the start
  2. Do ALL computation on GPU
  3. Move results back to CPU at the end

Bad pattern:
  for batch in data:
      batch = batch.cuda()      # Transfer
      result = model(batch)     # Compute
      result = result.cpu()     # Transfer back
      # This adds huge overhead!
""")


def batch_size_impact():
    """Show how batch size affects GPU utilization."""
    print("\n" + "=" * 60)
    print("Batch Size Impact on GPU Utilization")
    print("=" * 60)

    # Simulate a simple neural network layer
    input_size = 512
    output_size = 512

    weight_gpu = torch.randn(input_size, output_size, device='cuda')

    batch_sizes = [1, 8, 32, 128, 512, 2048]

    print(f"\n{'Batch':<10} {'Time (ms)':<12} {'Time/sample (ms)':<18} {'Samples/sec':<15}")
    print("-" * 58)

    for batch in batch_sizes:
        x_gpu = torch.randn(batch, input_size, device='cuda')

        time_ms = benchmark(lambda: torch.matmul(x_gpu, weight_gpu)) * 1000
        time_per_sample = time_ms / batch
        throughput = batch / time_ms * 1000  # samples per second

        print(f"{batch:<10} {time_ms:<12.3f} {time_per_sample:<18.6f} {throughput:<15.0f}")

    print("""
*** Larger batches = better GPU utilization ***

The GPU has thousands of cores that work in parallel.
Small batches leave most cores idle.
Larger batches keep more cores busy = higher throughput.

Trade-off: Larger batches need more memory.
Your 3.6GB GPU limits practical batch sizes for larger models.
""")


def synchronization_importance():
    """Show why CUDA synchronization matters for timing."""
    print("\n" + "=" * 60)
    print("Why CUDA Synchronization Matters")
    print("=" * 60)

    a = torch.randn(2000, 2000, device='cuda')
    b = torch.randn(2000, 2000, device='cuda')

    # WITHOUT synchronization (incorrect!)
    start = time.perf_counter()
    c = torch.matmul(a, b)  # This returns immediately! GPU still computing.
    wrong_time = (time.perf_counter() - start) * 1000

    # WITH synchronization (correct)
    torch.cuda.synchronize()  # Wait for previous op
    start = time.perf_counter()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Wait for this op to finish
    correct_time = (time.perf_counter() - start) * 1000

    print(f"\nWithout sync: {wrong_time:.3f} ms (WRONG - GPU still working!)")
    print(f"With sync:    {correct_time:.3f} ms (CORRECT)")

    print("""
*** GPU operations are ASYNCHRONOUS ***

torch.matmul() on GPU returns immediately while GPU works in background.
Without synchronize(), you're timing how fast Python can LAUNCH the op,
not how long the op actually takes.

Always use torch.cuda.synchronize() when benchmarking GPU code!
""")


def main():
    print("\n" + "=" * 60)
    print("   CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 60)

    matrix_multiply_comparison()
    small_operations_comparison()
    data_transfer_overhead()
    batch_size_impact()
    synchronization_importance()

    print("\n" + "=" * 60)
    print("   KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. GPU shines for LARGE parallel operations (matrices > 500x500)
2. For tiny operations, CPU can be faster (GPU launch overhead)
3. Data transfer is expensive - minimize CPU<->GPU movement
4. Larger batch sizes = better GPU utilization
5. Always use torch.cuda.synchronize() when timing GPU code
6. Keep data on GPU throughout computation pipeline
""")


if __name__ == "__main__":
    main()
