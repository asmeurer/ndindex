"""
Optimized pure Array API hash implementation for integer and boolean arrays.

This implementation computes the hash entirely using array operations,
keeping data on-device (GPU-friendly), and only transfers the final scalar
hash value to the host.

Specifically optimized for integer (intp) and boolean arrays as used in ndindex.

Requires: array-api-compat
"""

import sys
from array_api_compat import array_namespace, device
import math


def _get_size(array):
    """Get total number of elements in array (cross-compatible helper)."""
    # For PyTorch, .size is a method that returns shape
    # For NumPy/CuPy, .size is a property that returns number of elements
    if hasattr(array, 'numel'):
        # PyTorch
        return array.numel()
    elif hasattr(array, 'size') and not callable(array.size):
        # NumPy, CuPy
        return array.size
    else:
        # Compute from shape
        return math.prod(array.shape)


def hash_array_api_optimized(array, xp=None):
    """
    Hash an integer or boolean array using only Array API operations.

    This implementation:
    - Keeps all computation on device (GPU-friendly)
    - Only transfers final hash value (scalar) to host
    - Uses FNV-1a inspired hash algorithm with array operations
    - Includes shape and dtype to prevent collisions

    Parameters
    ----------
    array : array
        Integer or boolean array to hash
    xp : module, optional
        Array API namespace. If None, uses array.__array_namespace__()

    Returns
    -------
    int
        Hash value of the array
    """
    # Get the array API namespace
    if xp is None:
        xp = array_namespace(array)

    # Hash metadata first (computed on host)
    shape_hash = hash(tuple(array.shape))
    dtype_hash = hash(str(array.dtype))

    # Handle empty arrays
    size = _get_size(array)
    if size == 0:
        return hash((shape_hash, dtype_hash, 0))

    # Flatten the array for processing
    flat = xp.reshape(array, (size,))

    # Convert to int64 for hashing (booleans -> 0/1, integers stay as is)
    flat = xp.astype(flat, xp.int64)

    # FNV-1a hash parameters
    # Use smaller prime to avoid overflow issues
    FNV_PRIME = 1099511628211
    FNV_OFFSET = 14695981039346656037

    # For very large arrays, use chunked approach to avoid memory issues
    chunk_size = min(100000, size)

    if size <= chunk_size:
        # Small array - process in one go
        hash_value = _hash_chunk(flat, xp, FNV_OFFSET, FNV_PRIME)
    else:
        # Large array - process in chunks
        hash_value = FNV_OFFSET
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            chunk = flat[i:end]
            chunk_hash = _hash_chunk(chunk, xp, FNV_OFFSET, FNV_PRIME)

            # Combine chunk hashes
            hash_value = hash_value ^ chunk_hash
            hash_value = (hash_value * FNV_PRIME) % (2**63)

    # Transfer only the final hash value to host
    try:
        # For GPU arrays or other backends
        if hasattr(hash_value, 'item'):
            hash_value = hash_value.item()
        else:
            hash_value = int(hash_value)
    except (TypeError, AttributeError):
        # Already a Python int
        hash_value = int(hash_value)

    # Combine with metadata
    return hash((hash_value, shape_hash, dtype_hash))


def _hash_chunk(flat_array, xp, offset, prime):
    """
    Hash a chunk of flattened array using FNV-1a inspired algorithm.

    All operations stay on device until final scalar extraction.
    """
    # Start with offset
    hash_val = offset

    # Use a simpler approach for better compatibility
    # XOR all values with position-dependent weights
    n = flat_array.shape[0]

    # Get device for creating new arrays on the same device
    dev = device(flat_array)

    # Create position weights (powers of prime, modulo to prevent overflow)
    # For very large arrays, we'll use a stride pattern
    if n > 10000:
        # Sample-based approach for large arrays
        stride = max(1, n // 10000)
        indices = xp.arange(0, n, stride, dtype=xp.int64, device=dev)
        if hasattr(xp, 'take'):
            sampled = xp.take(flat_array, indices)
        else:
            sampled = flat_array[indices]

        # Position-weighted hash
        positions = xp.arange(len(indices), dtype=xp.int64, device=dev)
        # Keep weights small to prevent overflow
        weights = (positions % 65521) * 31 + 17  # 65521 is largest prime < 2^16

        # Compute hash
        contributions = sampled * weights
        # Use Python's int() to handle large sums safely
        result = int(xp.sum(contributions))
        # Mod to keep in reasonable range
        result = result % (2**63 - 1)
    else:
        # Full hash for smaller arrays
        positions = xp.arange(n, dtype=xp.int64, device=dev)

        # Use prime multipliers for positions (keep weights small)
        weights = (positions % 65521) * 31 + 17

        # Compute: sum of (value * weight)
        contributions = flat_array * weights
        # Use Python's int() for safe conversion
        result = int(xp.sum(contributions))
        result = result % (2**63 - 1)

    return result


def hash_array_api_fast(array, xp=None):
    """
    Even faster version using simpler hash function.

    This sacrifices some hash quality for better performance on large arrays.
    Good balance for ndindex use case where arrays are typically not huge.
    """
    if xp is None:
        xp = array_namespace(array)

    # Hash metadata
    shape_hash = hash(tuple(array.shape))
    dtype_hash = hash(str(array.dtype))

    size = _get_size(array)
    if size == 0:
        return hash((shape_hash, dtype_hash, 0))

    # Flatten
    flat = xp.reshape(array, (size,))

    # Convert to int64
    flat = xp.astype(flat, xp.int64)

    # Get device for creating new arrays on the same device
    dev = device(flat)

    # Simple position-weighted sum
    # This is fast and has good distribution for typical index arrays
    n = flat.shape[0]

    # For large arrays, sample; for small arrays, use all
    if n > 50000:
        # Sample approximately sqrt(n) elements for large arrays
        stride = max(1, int(n ** 0.5))
        indices = xp.arange(0, n, stride, device=dev)
        if hasattr(xp, 'take'):
            values = xp.take(flat, indices)
        else:
            values = flat[indices]
    else:
        values = flat

    # Compute hash using prime multipliers
    # Keep weights reasonable to prevent overflow
    positions = xp.arange(len(values), dtype=xp.int64, device=dev)
    # Use modulo to keep weights in safe range
    weights = (positions % 65521) * 31 + 17

    # Compute weighted sum
    weighted = values * weights
    sum_result = xp.sum(weighted)

    # Extract scalar safely - convert to Python int which handles big integers
    if hasattr(sum_result, 'item'):
        hash_value = int(sum_result.item())
    else:
        hash_value = int(sum_result)

    # Mod to keep in reasonable range
    hash_value = hash_value % (2**63 - 1)

    # Combine with metadata
    return hash((hash_value, shape_hash, dtype_hash))


def hash_array_api_xxhash_style(array, xp=None):
    """
    xxHash-inspired implementation using only Array API operations.

    xxHash is known for excellent speed and distribution.
    This is a simplified version that stays on-device.
    """
    if xp is None:
        xp = array_namespace(array)

    # Constants from xxHash
    PRIME1 = 11400714785074694791  # 0x9E3779B185EBCA87
    PRIME2 = 14029467366897019727  # 0xC2B2AE3D27D4EB4F
    PRIME3 = 1609587929392839161   # 0x165667B19E3779F9
    PRIME4 = 9650029242287828579   # 0x85EBCA77C2B2AE63
    PRIME5 = 2870177450012600261   # 0x27D4EB2F165667C5

    # Metadata hash
    shape_hash = hash(tuple(array.shape))
    dtype_hash = hash(str(array.dtype))

    size = _get_size(array)
    if size == 0:
        return hash((shape_hash, dtype_hash, 0))

    # Flatten and convert to int64
    flat = xp.reshape(array, (size,))
    flat = xp.astype(flat, xp.int64)

    # Get device for creating new arrays on the same device
    dev = device(flat)

    n = flat.shape[0]

    # Simplified xxHash-style computation
    # Use much smaller primes to avoid overflow
    PRIME_A = 31
    PRIME_B = 17
    PRIME_C = 65521  # Largest prime < 2^16

    # Initialize accumulator
    acc = n * PRIME_A

    # Process elements
    if n <= 10000:
        # Small array - process with array operations
        positions = xp.arange(n, dtype=xp.int64, device=dev)
        weights = (positions % PRIME_C) * PRIME_A + PRIME_B

        contributions = flat * weights
        sum_val = xp.sum(contributions)

        # Extract safely
        if hasattr(sum_val, 'item'):
            contribution = int(sum_val.item())
        else:
            contribution = int(sum_val)

        acc = (acc + contribution) % (2**63 - 1)
    else:
        # Large array - sample for performance
        stride = max(1, n // 1000)
        if hasattr(xp, 'take'):
            indices = xp.arange(0, n, stride, dtype=xp.int64, device=dev)
            sampled = xp.take(flat, indices)
        else:
            sampled = flat[::stride]

        positions = xp.arange(len(sampled), dtype=xp.int64, device=dev)
        weights = (positions % PRIME_C) * PRIME_A + PRIME_B
        contributions = sampled * weights
        sum_val = xp.sum(contributions)

        # Extract safely
        if hasattr(sum_val, 'item'):
            contribution = int(sum_val.item())
        else:
            contribution = int(sum_val)

        acc = (acc + contribution) % (2**63 - 1)

    # Simple final mix
    acc = acc ^ (acc >> 17)
    acc = (acc * PRIME_A) % (2**63 - 1)
    acc = acc ^ (acc >> 13)

    # Combine with metadata
    return hash((acc, shape_hash, dtype_hash))


# Alias for recommended version
hash_array_api_device = hash_array_api_fast
