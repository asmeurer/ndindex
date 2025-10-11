# GPU Testing Guide

This guide helps you test the Array API hashing implementations on GPU with CuPy and PyTorch.

## Quick Test

### 1. Install GPU Libraries

```bash
# For CuPy (CUDA)
pip install cupy-cuda12x  # Replace with your CUDA version

# For PyTorch (CUDA or MPS)
pip install torch

# Optional: array-api-strict for standards compliance testing
pip install array-api-strict
```

### 2. Run Demo and Benchmarks

```bash
# Run the demo (tests all backends automatically)
python demo_hashing.py

# Run full benchmarks (includes GPU if available)
python benchmark_array_hashing.py
```

## Detailed GPU Testing

### Test with CuPy

```python
import cupy as cp
from array_hash_optimized import hash_array_api_fast, hash_array_api_xxhash_style

# Create GPU array
gpu_arr = cp.arange(1000000, dtype=cp.int64)

print("Testing CuPy GPU array:")
print(f"Array device: {gpu_arr.device}")
print(f"Array shape: {gpu_arr.shape}")

# Test on-device hash
h1 = hash_array_api_fast(gpu_arr)
print(f"hash_array_api_fast: {h1}")

h2 = hash_array_api_xxhash_style(gpu_arr)
print(f"hash_array_api_xxhash_style: {h2}")

# Verify collision resistance
gpu_arr2 = cp.arange(1000000, dtype=cp.int64).reshape(1000, 1000)
h3 = hash_array_api_fast(gpu_arr2)
print(f"\nDifferent shape hash: {h3}")
print(f"Collision check: {h1 == h3} (should be False)")
```

### Test with PyTorch

```python
import torch
from array_hash_optimized import hash_array_api_fast

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Create GPU tensor
gpu_tensor = torch.arange(1000000, dtype=torch.int64, device=device)

print(f"Tensor device: {gpu_tensor.device}")
print(f"Tensor shape: {gpu_tensor.shape}")

# Test on-device hash
h = hash_array_api_fast(gpu_tensor)
print(f"hash_array_api_fast: {h}")

# Verify works with different shapes
gpu_tensor2 = torch.arange(1000000, dtype=torch.int64, device=device).reshape(1000, 1000)
h2 = hash_array_api_fast(gpu_tensor2)
print(f"Different shape hash: {h2}")
print(f"Collision check: {h == h2} (should be False)")
```

## Performance Benchmarks

### CuPy Performance Test

```python
import cupy as cp
import time
from array_hash_optimized import hash_array_api_fast
import numpy as np

sizes = [1000, 10000, 100000, 1000000, 10000000]

print("\nCuPy GPU Performance:")
print(f"{'Size':<12} {'Time (ms)':<12} {'Throughput (M elem/s)':<20}")
print("-" * 50)

for size in sizes:
    gpu_arr = cp.arange(size, dtype=cp.int64)

    # Warmup
    for _ in range(10):
        hash_array_api_fast(gpu_arr)

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        h = hash_array_api_fast(gpu_arr)
    elapsed = time.perf_counter() - start

    time_per_hash = elapsed / 100 * 1000  # ms
    throughput = size / (time_per_hash / 1000) / 1e6  # M elements/sec

    print(f"{size:<12,} {time_per_hash:<12.4f} {throughput:<20.2f}")
```

### Compare CPU vs GPU

```python
import numpy as np
import cupy as cp
import time
from array_hash_optimized import hash_array_api_fast

size = 10000000  # 10M elements

# CPU (NumPy)
cpu_arr = np.arange(size, dtype=np.int64)
start = time.perf_counter()
for _ in range(100):
    h = hash_array_api_fast(cpu_arr)
cpu_time = time.perf_counter() - start

# GPU (CuPy)
gpu_arr = cp.arange(size, dtype=cp.int64)
# Warmup
for _ in range(10):
    hash_array_api_fast(gpu_arr)

start = time.perf_counter()
for _ in range(100):
    h = hash_array_api_fast(gpu_arr)
gpu_time = time.perf_counter() - start

print(f"\nCPU vs GPU Performance (10M elements, 100 runs):")
print(f"CPU (NumPy):  {cpu_time*10:.2f} ms")
print(f"GPU (CuPy):   {gpu_time*10:.2f} ms")
print(f"Speedup:      {cpu_time/gpu_time:.2f}x")
```

## Expected Results

### On CPU (NumPy)
- Small arrays (100): ~0.008ms
- Medium arrays (10K): ~0.019ms
- Large arrays (1M): ~0.52ms
- Very large (10M): ~5.2ms

### On GPU (CuPy)
Expected improvements:
- Small arrays: Similar to CPU (overhead dominates)
- Medium arrays: Potentially faster
- Large arrays: **Significantly faster** - no device transfer!
- Very large: **Much faster** - GPU shines here

The key advantage: Computation stays on GPU, only final hash scalar transfers to host.

## Troubleshooting

### CuPy Not Found
```bash
# Check CUDA version
nvidia-smi

# Install matching CuPy version
# For CUDA 12.x:
pip install cupy-cuda12x
# For CUDA 11.x:
pip install cupy-cuda11x
```

### PyTorch Device Issues
```python
# Check available devices
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# If CUDA not available on Mac, try MPS:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### Memory Issues with Large Arrays
If you run out of GPU memory with very large arrays:
```python
# Use smaller test sizes
sizes = [1000, 10000, 100000, 1000000]  # Stop at 1M instead of 10M
```

## What To Look For

### ✅ Success Indicators
1. No errors when running on GPU arrays
2. Hash values are consistent for equal arrays
3. Hash values differ for arrays with different shapes
4. GPU performance is better than CPU for large arrays
5. No device→host transfer warnings

### ❌ Potential Issues
1. "Cannot convert to scalar" errors
   - May need to adjust `.item()` extraction
2. Very slow on GPU
   - Unexpected device transfers happening
3. Different hash values than CPU
   - Type conversion issues

## Report Back

When testing, please report:
1. GPU type and memory
2. CUDA/ROCm version
3. CuPy/PyTorch versions
4. Performance results for different sizes
5. Any errors or unexpected behavior

Run this to get system info:
```python
import sys
print(f"Python: {sys.version}")

try:
    import cupy as cp
    print(f"CuPy: {cp.__version__}")
    print(f"CUDA: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"GPU: {cp.cuda.Device().name}")
except:
    print("CuPy: Not available")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    print("PyTorch: Not available")
```

## Save Results

The benchmark script automatically saves results. After running:
```bash
python benchmark_array_hashing.py > gpu_benchmark_results.txt
```

Then commit and push the results file for comparison.
