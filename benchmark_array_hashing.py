"""
Benchmark script to compare different array hashing implementations.

This script tests the performance and correctness of various hashing approaches
for array-api compliant arrays.
"""

import time
import numpy as np
from array_hash_prototypes import (
    hash_dlpack_ctypes,
    hash_array_api_polynomial,
    hash_array_api_sampled,
    hash_buffer_protocol,
    hash_array_hybrid,
    hash_numpy_tobytes,
    hash_numpy_tobytes_fixed,
)

from array_hash_optimized import (
    hash_array_api_optimized,
    hash_array_api_fast,
    hash_array_api_xxhash_style,
)

# Try to import Cython implementation if available
try:
    from array_hash_cython import hash_dlpack_cython, hash_dlpack_cython_chunked
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython implementation not available. Build it with:")
    print("  python setup_cython_hash.py build_ext --inplace")
    print()


def benchmark_hash_function(func, array, name, runs=100):
    """Benchmark a single hash function."""
    results = {
        'name': name,
        'runs': runs,
        'times': [],
        'hash_value': None,
        'error': None,
    }

    # Warmup
    try:
        hash_value = func(array)
        results['hash_value'] = hash_value
    except Exception as e:
        results['error'] = str(e)
        return results

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            h = func(array)
            end = time.perf_counter()
            times.append(end - start)

            # Verify consistency
            if h != hash_value:
                results['error'] = "Hash value not consistent across runs"
                break
        except Exception as e:
            results['error'] = str(e)
            break

    results['times'] = times
    return results


def analyze_results(results):
    """Analyze and print benchmark results."""
    if results['error']:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'std': None,
        }

    times = results['times']
    if not times:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'std': None,
        }

    return {
        'mean': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
    }


def print_results(all_results, array_description):
    """Print formatted benchmark results."""
    print(f"\n{'=' * 80}")
    print(f"Benchmark Results: {array_description}")
    print(f"{'=' * 80}")
    print(f"{'Method':<30} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Status':<20}")
    print(f"{'-' * 80}")

    for result in all_results:
        stats = analyze_results(result)
        name = result['name']

        if result['error']:
            status = f"Error: {result['error'][:30]}"
            print(f"{name:<30} {'-':<12} {'-':<12} {'-':<12} {status:<20}")
        else:
            mean_ms = stats['mean'] * 1000
            min_ms = stats['min'] * 1000
            max_ms = stats['max'] * 1000
            status = "OK"
            print(f"{name:<30} {mean_ms:<12.4f} {min_ms:<12.4f} {max_ms:<12.4f} {status:<20}")

    print(f"{'=' * 80}\n")


def check_hash_consistency(all_results):
    """Check if all successful hash functions produce the same hash value."""
    hash_values = {}
    for result in all_results:
        if result['hash_value'] is not None:
            hash_values[result['name']] = result['hash_value']

    if len(set(hash_values.values())) > 1:
        print("\nWARNING: Hash values are not consistent across implementations!")
        for name, h in hash_values.items():
            print(f"  {name}: {h}")
    else:
        print(f"\nHash consistency check: PASSED (all methods produced same hash)")


def run_benchmarks():
    """Run all benchmarks with different array sizes and configurations."""

    # Test configurations
    test_cases = [
        ("Small 1D (100)", np.arange(100, dtype=np.int64)),
        ("Medium 1D (10K)", np.arange(10000, dtype=np.int64)),
        ("Large 1D (1M)", np.arange(1000000, dtype=np.int64)),
        ("Small 2D (10x10)", np.arange(100, dtype=np.int64).reshape(10, 10)),
        ("Medium 2D (100x100)", np.arange(10000, dtype=np.int64).reshape(100, 100)),
        ("Large 2D (1000x1000)", np.arange(1000000, dtype=np.int64).reshape(1000, 1000)),
        ("Float array (10K)", np.random.randn(10000)),
        ("Boolean array (10K)", np.random.randint(0, 2, 10000, dtype=bool)),
    ]

    # Hash functions to test
    hash_functions = [
        (hash_numpy_tobytes, "NumPy tobytes (old)"),
        (hash_numpy_tobytes_fixed, "NumPy tobytes (fixed)"),
        (hash_buffer_protocol, "Buffer protocol"),
        (hash_dlpack_ctypes, "DLPack + ctypes"),
        (lambda a: hash_array_api_sampled(a, max_samples=100), "Array API sampled (100)"),
        (lambda a: hash_array_api_sampled(a, max_samples=1000), "Array API sampled (1000)"),
        (hash_array_hybrid, "Hybrid approach"),
        (hash_array_api_optimized, "Array API optimized (on-device)"),
        (hash_array_api_fast, "Array API fast (on-device)"),
        (hash_array_api_xxhash_style, "Array API xxHash (on-device)"),
    ]

    # Add Cython implementations if available
    if CYTHON_AVAILABLE:
        hash_functions.extend([
            (hash_dlpack_cython, "DLPack + Cython"),
            (hash_dlpack_cython_chunked, "DLPack + Cython (chunked)"),
        ])

    # Note: hash_array_api_polynomial is excluded from main benchmarks because it's very slow
    # It can be tested separately for small arrays

    for description, array in test_cases:
        # Make array read-only (matches ndindex behavior)
        array.flags.writeable = False

        all_results = []

        for func, name in hash_functions:
            result = benchmark_hash_function(func, array, name, runs=50)
            all_results.append(result)

        print_results(all_results, description)
        check_hash_consistency(all_results)


def test_polynomial_hash_small():
    """Separately test the polynomial hash on small arrays since it's slow."""
    print("\n" + "=" * 80)
    print("Testing Polynomial Hash (slow - only on small arrays)")
    print("=" * 80)

    test_cases = [
        ("Small 1D (100)", np.arange(100, dtype=np.int64)),
        ("Small 2D (10x10)", np.arange(100, dtype=np.int64).reshape(10, 10)),
    ]

    for description, array in test_cases:
        array.flags.writeable = False

        all_results = []

        # Test baseline and polynomial
        for func, name in [
            (hash_numpy_tobytes, "NumPy tobytes (baseline)"),
            (hash_array_api_polynomial, "Array API polynomial"),
        ]:
            result = benchmark_hash_function(func, array, name, runs=20)
            all_results.append(result)

        print_results(all_results, description)
        check_hash_consistency(all_results)


def test_collision_resistance():
    """Test how well different methods resist hash collisions."""
    print("\n" + "=" * 80)
    print("Collision Resistance Test")
    print("=" * 80)

    # Create similar arrays that should have different hashes
    test_pairs = [
        ("Different values", np.array([1, 2, 3]), np.array([1, 2, 4])),
        ("Different shapes", np.array([1, 2, 3, 4]), np.array([[1, 2], [3, 4]])),
        ("Different dtypes", np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int64)),
        ("Transposed", np.array([[1, 2], [3, 4]]), np.array([[1, 3], [2, 4]])),
    ]

    hash_functions = [
        (hash_numpy_tobytes, "NumPy tobytes (old)"),
        (hash_numpy_tobytes_fixed, "NumPy tobytes (fixed)"),
        (hash_buffer_protocol, "Buffer protocol"),
        (hash_dlpack_ctypes, "DLPack + ctypes"),
        (lambda a: hash_array_api_sampled(a, max_samples=100), "Array API sampled"),
        (hash_array_hybrid, "Hybrid approach"),
    ]

    for pair_name, arr1, arr2 in test_pairs:
        print(f"\n{pair_name}:")
        print(f"  Array 1: shape={arr1.shape}, dtype={arr1.dtype}")
        print(f"  Array 2: shape={arr2.shape}, dtype={arr2.dtype}")

        for func, name in hash_functions:
            try:
                h1 = func(arr1)
                h2 = func(arr2)
                if h1 == h2:
                    print(f"  {name:<30} COLLISION! (both={h1})")
                else:
                    print(f"  {name:<30} OK (different hashes)")
            except Exception as e:
                print(f"  {name:<30} Error: {e}")


def test_array_api_compatibility():
    """Test with different array API implementations if available."""
    print("\n" + "=" * 80)
    print("Array API Compatibility Test")
    print("=" * 80)

    # Try to import different array libraries
    array_libs = []

    # NumPy (always available)
    array_libs.append(("NumPy", np))

    # Try array-api-strict
    try:
        import array_api_strict
        array_libs.append(("array-api-strict", array_api_strict))
    except ImportError:
        print("array-api-strict not available (install with: pip install array-api-strict)")

    # Try CuPy (GPU arrays)
    try:
        import cupy as cp
        array_libs.append(("CuPy", cp))
    except ImportError:
        print("CuPy not available (GPU arrays not tested)")

    # Try PyTorch
    try:
        import torch
        array_libs.append(("PyTorch", torch))
    except ImportError:
        print("PyTorch not available")

    print()

    # Test each library
    for lib_name, lib in array_libs:
        print(f"\nTesting with {lib_name}:")

        # Create a test array
        if lib_name == "PyTorch":
            arr = lib.arange(1000, dtype=lib.int64)
        elif lib_name == "CuPy":
            # For CuPy, we need to transfer to CPU for most hash methods
            arr = lib.arange(1000, dtype=lib.int64)
        else:
            arr = lib.asarray(np.arange(1000, dtype=np.int64))

        # Test methods that should work with Array API
        test_functions = [
            (lambda a: hash_array_api_sampled(a, max_samples=100), "Array API sampled"),
            (hash_array_hybrid, "Hybrid approach"),
        ]

        # Add library-specific methods
        if lib_name == "NumPy":
            test_functions.insert(0, (hash_numpy_tobytes, "NumPy tobytes"))
            test_functions.insert(1, (hash_buffer_protocol, "Buffer protocol"))

        for func, name in test_functions:
            try:
                start = time.perf_counter()
                h = func(arr)
                elapsed = time.perf_counter() - start
                print(f"  {name:<30} OK (hash={h}, time={elapsed*1000:.2f}ms)")
            except Exception as e:
                print(f"  {name:<30} Error: {str(e)[:50]}")


if __name__ == "__main__":
    print("=" * 80)
    print("Array Hashing Benchmark Suite")
    print("=" * 80)
    print("\nThis script benchmarks different approaches to hashing arrays")
    print("for Array API compliance in ndindex.\n")

    # Run main benchmarks
    run_benchmarks()

    # Test polynomial hash separately (it's slow)
    test_polynomial_hash_small()

    # Test collision resistance
    test_collision_resistance()

    # Test with different array libraries
    test_array_api_compatibility()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
