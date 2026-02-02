# Copyright (c) 2024-2025 Neeraj Gandhi
# Licensed under the MIT License - see LICENSE file for details

"""
01_check_nvidia.py - Verify your GPU setup is working

Learning objectives:
- Confirm NVIDIA driver is installed
- Verify PyTorch can access the GPU
- Understand what information is available about your hardware

Run: python 01_check_nvidia.py
"""

import subprocess
import sys


def check_nvidia_smi():
    """Check if nvidia-smi is available and working."""
    print("=" * 60)
    print("STEP 1: Checking NVIDIA Driver (nvidia-smi)")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total",
             "--format=csv"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Driver is installed and working!")
        print(result.stdout)
    except FileNotFoundError:
        print(
            "ERROR: nvidia-smi not found. NVIDIA driver may not be installed.")
        print("Install with: sudo apt install nvidia-driver-XXX")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: nvidia-smi failed: {e}")
        return False

    return True


def check_pytorch_cuda():
    """Check if PyTorch can use CUDA."""
    print("=" * 60)
    print("STEP 2: Checking PyTorch CUDA Support")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("Install with: pip install torch " +
              "--index-url https://download.pytorch.org/whl/cu124")
        return False

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA is not available to PyTorch.")
        print("Possible causes:")
        print("  1. PyTorch CPU-only version installed (reinstall with CUDA)")
        print("  2. NVIDIA driver not loaded (try rebooting)")
        print("  3. Driver/CUDA version mismatch")
        return False

    print(f"CUDA version:    {torch.version.cuda}")
    print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"Device count:    {torch.cuda.device_count()}")

    return True


def check_gpu_details():
    """Get detailed GPU information from PyTorch."""
    print("=" * 60)
    print("STEP 3: GPU Details")
    print("=" * 60)

    import torch

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory:       {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessors:    {props.multi_processor_count}")

        # Memory info
        total = torch.cuda.get_device_properties(i).total_memory
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = total - reserved

        print(f"  Memory reserved:    {reserved / 1024**3:.2f} GB")
        print(f"  Memory allocated:   {allocated / 1024**3:.2f} GB")
        print(f"  Memory free:        {free / 1024**3:.2f} GB")


def test_gpu_compute():
    """Run a simple computation on GPU to verify everything works."""
    print("=" * 60)
    print("STEP 4: GPU Compute Test")
    print("=" * 60)

    import torch

    print("Creating tensors on GPU...")

    # Create random tensors on GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')

    print("Performing matrix multiplication...")
    c = torch.matmul(a, b)

    # Synchronize to ensure computation is complete
    torch.cuda.synchronize()

    print(f"Result shape: {c.shape}")
    print(f"Result device: {c.device}")
    print(f"Result sample values: {c[0, :5].tolist()}")

    print("\nGPU compute test PASSED!")
    return True


def main():
    print("\n" + "=" * 60)
    print("   GPU SETUP VERIFICATION")
    print("=" * 60 + "\n")

    # Run all checks
    driver_ok = check_nvidia_smi()
    print()

    pytorch_ok = check_pytorch_cuda()
    print()

    if pytorch_ok:
        check_gpu_details()
        print()
        test_gpu_compute()

    # Summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)

    if driver_ok:# and pytorch_ok:
        print("\nAll checks PASSED! Your GPU is ready for the course.")
        print("\nNext: Run 02_cuda_basics.py to learn tensor operations.")
    else:
        print("\nSome checks FAILED. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
