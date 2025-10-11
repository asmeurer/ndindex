"""
Prototypes for different array hashing approaches compatible with Array API standard.

This module contains several implementations of array hashing that work with
array-api compliant arrays, not just NumPy.
"""

import sys
from typing import Any


# ============================================================================
# Option 1: DLPack + ctypes
# ============================================================================

def hash_dlpack_ctypes(array) -> int:
    """
    Hash an array using DLPack with ctypes to access raw memory buffer.

    This approach uses the __dlpack__() method to get a PyCapsule containing
    the DLManagedTensor, then uses ctypes to access the raw data pointer.

    Limitations:
    - Only works for CPU arrays (or requires device transfer)
    - Complex implementation with ctypes
    - Requires understanding of DLPack C structure
    """
    import ctypes

    # Make a writable copy if needed (DLPack may not work with readonly arrays)
    if hasattr(array, 'flags') and not array.flags.writeable:
        # Create a writable copy
        if hasattr(array, 'copy'):
            array = array.copy()
        else:
            # Fallback for non-NumPy arrays
            import numpy as np
            array = np.array(array, copy=True)

    # Get the DLPack capsule
    try:
        capsule = array.__dlpack__()
    except AttributeError:
        raise TypeError("Array does not support __dlpack__ protocol")

    # Define the DLPack structure
    # This is a simplified version - full implementation would need complete struct
    class DLDevice(ctypes.Structure):
        _fields_ = [
            ("device_type", ctypes.c_int32),
            ("device_id", ctypes.c_int32),
        ]

    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int32),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        _fields_ = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    # Get pointer to the DLManagedTensor from PyCapsule
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    # Access the managed tensor
    managed_tensor_ptr = PyCapsule_GetPointer(capsule, b"dltensor")
    if not managed_tensor_ptr:
        raise ValueError("Failed to get DLManagedTensor from capsule")

    managed_tensor = ctypes.cast(managed_tensor_ptr, ctypes.POINTER(DLManagedTensor)).contents
    dl_tensor = managed_tensor.dl_tensor

    # Check if on CPU
    if dl_tensor.device.device_type != 1:  # 1 = kDLCPU
        raise ValueError("Array must be on CPU for this hashing method")

    # Calculate total size in bytes
    total_elements = 1
    for i in range(dl_tensor.ndim):
        total_elements *= dl_tensor.shape[i]

    element_size = dl_tensor.dtype.bits // 8
    total_bytes = total_elements * element_size

    # Create a buffer from the data pointer
    data_ptr = dl_tensor.data + dl_tensor.byte_offset
    buffer = (ctypes.c_ubyte * total_bytes).from_address(data_ptr)

    # Hash the bytes along with shape and dtype to avoid collisions
    # (e.g., [1,2,3,4] vs [[1,2],[3,4]] should have different hashes)
    shape_tuple = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
    dtype_info = (dl_tensor.dtype.code, dl_tensor.dtype.bits, dl_tensor.dtype.lanes)

    return hash((bytes(buffer), shape_tuple, dtype_info))


# ============================================================================
# Option 2: Pure Array API - Polynomial Rolling Hash
# ============================================================================

def hash_array_api_polynomial(array, xp=None) -> int:
    """
    Hash an array using only Array API operations with a polynomial rolling hash.

    This uses a simple polynomial hash similar to Python's string hashing:
    hash = (hash * prime + value) for each value

    Pros:
    - Works with any Array API backend
    - No C dependencies
    - Portable

    Cons:
    - Slower than direct memory access
    - May trigger device→host transfers
    - Need to handle large integers carefully
    """
    # Get the array API namespace
    if xp is None:
        try:
            xp = array.__array_namespace__()
        except AttributeError:
            # Fallback to numpy
            import numpy as xp

    # Constants for hashing
    PRIME = 31
    MODULUS = 2**61 - 1  # Large Mersenne prime to avoid overflow

    # Start with hash of shape and dtype
    shape_hash = hash(tuple(array.shape))
    dtype_hash = hash(str(array.dtype))
    result = (shape_hash * PRIME + dtype_hash) % MODULUS

    # Flatten the array for iteration
    # Array API doesn't have flatten, so we need to reshape
    flat = xp.reshape(array, (array.size,))

    # For large arrays, we need to be smart about this
    # We'll hash in chunks to avoid memory issues
    chunk_size = min(10000, array.size)

    for i in range(0, array.size, chunk_size):
        end = min(i + chunk_size, array.size)
        chunk = flat[i:end]

        # Convert chunk to Python for hashing
        # This is the bottleneck - requires device→host transfer
        # Different backends may have different ways to do this
        if hasattr(chunk, 'tolist'):
            values = chunk.tolist()
        elif hasattr(chunk, '__iter__'):
            # Try iteration
            values = list(chunk)
        else:
            # Last resort: index each element
            values = [chunk[j] for j in range(end - i)]

        # Hash the values
        for val in values:
            # Convert to int representation for hashing
            if isinstance(val, float):
                # Use float.hex() for deterministic float hashing
                val_hash = hash(float(val).hex())
            elif isinstance(val, (bool, int)):
                val_hash = hash(int(val))
            elif isinstance(val, complex):
                val_hash = hash((val.real.hex(), val.imag.hex()))
            else:
                val_hash = hash(val)

            result = (result * PRIME + val_hash) % MODULUS

    # Convert to Python's hash range
    return result if result <= sys.maxsize else result - 2**(sys.maxsize.bit_length() + 1)


# ============================================================================
# Option 3: Pure Array API - Chunk-based Sampling Hash
# ============================================================================

def hash_array_api_sampled(array, xp=None, max_samples=1000) -> int:
    """
    Hash an array by sampling strategic elements rather than hashing all data.

    This is faster than hashing the entire array but has higher collision risk.
    Good for large arrays where full hashing is expensive.

    Strategy:
    - Always include shape, dtype, size
    - Sample elements from beginning, middle, and end
    - Use stride pattern to get diverse samples
    """
    # Get the array API namespace
    if xp is None:
        try:
            xp = array.__array_namespace__()
        except AttributeError:
            import numpy as xp

    # Start with metadata
    h = hash((tuple(array.shape), str(array.dtype), array.size))

    if array.size == 0:
        return h

    # Flatten the array
    flat = xp.reshape(array, (array.size,))

    # Determine sampling strategy
    if array.size <= max_samples:
        # Small array - hash everything
        indices = list(range(array.size))
    else:
        # Large array - sample strategically
        indices = []

        # First and last elements
        indices.extend([0, array.size - 1])

        # Evenly distributed samples
        step = array.size // (max_samples - 2)
        indices.extend(range(step, array.size - 1, step))

        # Remove duplicates and sort
        indices = sorted(set(indices))[:max_samples]

    # Extract and hash the sampled elements
    # Use array indexing to get samples
    sampled_values = []
    for idx in indices:
        val = flat[idx]
        # Convert to Python scalar
        if hasattr(val, 'item'):
            val = val.item()
        elif hasattr(val, '__int__'):
            val = int(val)
        elif hasattr(val, '__float__'):
            val = float(val)
        sampled_values.append(val)

    # Combine with metadata hash
    return hash((h, tuple(sampled_values)))


# ============================================================================
# Option 4: Buffer Protocol Fallback
# ============================================================================

def hash_buffer_protocol(array) -> int:
    """
    Hash an array using Python's buffer protocol if available.

    This is a fallback that works when the array supports memoryview(),
    which is not required by Array API but is supported by many implementations.

    Pros:
    - Fast when available
    - Standard Python

    Cons:
    - Not guaranteed by Array API standard
    - May not work for GPU arrays
    """
    try:
        # Try to create a memoryview
        mv = memoryview(array)
        # Convert to bytes and hash, include shape and dtype to avoid collisions
        data_hash = hash(mv.tobytes())
        shape_hash = hash(tuple(array.shape)) if hasattr(array, 'shape') else 0
        dtype_hash = hash(str(array.dtype)) if hasattr(array, 'dtype') else 0
        return hash((data_hash, shape_hash, dtype_hash))
    except (TypeError, AttributeError, BufferError):
        raise TypeError("Array does not support buffer protocol")


# ============================================================================
# Option 5: Hybrid Approach with Fallbacks
# ============================================================================

def hash_array_hybrid(array, xp=None) -> int:
    """
    Hybrid approach that tries multiple methods in order of preference.

    1. Try buffer protocol (fastest)
    2. Try DLPack with ctypes (fast, but CPU only)
    3. Fall back to sampled hash (slower but always works)
    """
    # First, try buffer protocol
    try:
        return hash_buffer_protocol(array)
    except (TypeError, AttributeError, BufferError):
        pass

    # Second, try DLPack for CPU arrays
    try:
        return hash_dlpack_ctypes(array)
    except (TypeError, ValueError, AttributeError):
        pass

    # Finally, fall back to sampled hash
    return hash_array_api_sampled(array, xp)


# ============================================================================
# Option 6: NumPy compatibility (current implementation)
# ============================================================================

def hash_numpy_tobytes(array) -> int:
    """
    Original implementation using NumPy's tobytes().

    This is the current implementation in ndindex - included for comparison.
    Only works with NumPy arrays.

    Note: This has a collision issue - arrays with same data but different
    shapes will hash the same (e.g., [1,2,3,4] vs [[1,2],[3,4]]).
    """
    return hash(array.tobytes())


def hash_numpy_tobytes_fixed(array) -> int:
    """
    Improved NumPy implementation that includes shape and dtype in hash.

    This fixes the collision issue in the original implementation.
    """
    return hash((array.tobytes(), array.shape, str(array.dtype)))
