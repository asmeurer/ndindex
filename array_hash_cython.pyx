"""
Cython implementation for fast array hashing using DLPack.

This module provides a high-performance implementation of array hashing
by directly accessing the memory buffer through DLPack protocol.

To compile:
    python setup_cython_hash.py build_ext --inplace
"""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint64_t
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid


# DLPack structure definitions
cdef extern from *:
    """
    typedef enum {
        kDLCPU = 1,
        kDLCUDA = 2,
        kDLCUDAHost = 3,
        kDLOpenCL = 4,
        kDLVulkan = 7,
        kDLMetal = 8,
        kDLVPI = 9,
        kDLROCM = 10,
    } DLDeviceType;

    typedef struct {
        int32_t device_type;
        int32_t device_id;
    } DLDevice;

    typedef struct {
        uint8_t code;
        uint8_t bits;
        uint16_t lanes;
    } DLDataType;

    typedef struct {
        void* data;
        DLDevice device;
        int32_t ndim;
        DLDataType dtype;
        int64_t* shape;
        int64_t* strides;
        uint64_t byte_offset;
    } DLTensor;

    typedef struct {
        DLTensor dl_tensor;
        void* manager_ctx;
        void (*deleter)(void*);
    } DLManagedTensor;
    """
    ctypedef enum DLDeviceType:
        kDLCPU = 1

    ctypedef struct DLDevice:
        int32_t device_type
        int32_t device_id

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void* deleter


cdef uint64_t hash_bytes(const uint8_t* data, uint64_t length) nogil:
    """
    Fast hash function for byte arrays.

    Uses FNV-1a hash algorithm which is fast and has good distribution.
    """
    cdef uint64_t hash_value = 14695981039346656037UL  # FNV offset basis
    cdef uint64_t fnv_prime = 1099511628211UL
    cdef uint64_t i

    for i in range(length):
        hash_value ^= data[i]
        hash_value *= fnv_prime

    return hash_value


def hash_dlpack_cython(array):
    """
    Hash an array using DLPack protocol with Cython for maximum performance.

    Parameters
    ----------
    array : array-like
        An array object that supports the __dlpack__() protocol

    Returns
    -------
    int
        Hash value of the array

    Raises
    ------
    TypeError
        If the array doesn't support __dlpack__()
    ValueError
        If the array is not on CPU (GPU arrays not supported yet)
    """
    cdef DLManagedTensor* managed_tensor
    cdef DLTensor* dl_tensor
    cdef uint8_t* data_ptr
    cdef uint64_t total_elements
    cdef uint64_t total_bytes
    cdef uint64_t hash_value
    cdef int i

    # Make a writable copy if needed (DLPack may not work with readonly arrays)
    if hasattr(array, 'flags') and not array.flags.writeable:
        if hasattr(array, 'copy'):
            array = array.copy()
        else:
            import numpy as np
            array = np.array(array, copy=True)

    # Get the DLPack capsule
    try:
        capsule = array.__dlpack__()
    except AttributeError:
        raise TypeError("Array does not support __dlpack__ protocol")

    # Validate the capsule
    if not PyCapsule_IsValid(capsule, b"dltensor"):
        raise ValueError("Invalid DLPack capsule")

    # Get the managed tensor
    managed_tensor = <DLManagedTensor*>PyCapsule_GetPointer(capsule, b"dltensor")
    if managed_tensor == NULL:
        raise ValueError("Failed to get DLManagedTensor from capsule")

    dl_tensor = &managed_tensor.dl_tensor

    # Check if on CPU
    if dl_tensor.device.device_type != kDLCPU:
        raise ValueError(f"Array must be on CPU (device type: {dl_tensor.device.device_type})")

    # Calculate total size
    total_elements = 1
    for i in range(dl_tensor.ndim):
        total_elements *= dl_tensor.shape[i]

    element_size = dl_tensor.dtype.bits // 8
    total_bytes = total_elements * element_size

    # Get data pointer
    data_ptr = <uint8_t*>dl_tensor.data + dl_tensor.byte_offset

    # Hash the bytes (release GIL for performance)
    with nogil:
        hash_value = hash_bytes(data_ptr, total_bytes)

    # Build shape and dtype info for complete hash
    # This prevents collisions between arrays with same data but different shapes
    shape_tuple = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
    dtype_info = (dl_tensor.dtype.code, dl_tensor.dtype.bits, dl_tensor.dtype.lanes)

    # Combine data hash with shape and dtype
    return hash((hash_value, shape_tuple, dtype_info))


def hash_dlpack_cython_chunked(array, chunk_size=1024*1024):
    """
    Hash an array using DLPack protocol with chunked processing.

    This version processes the array in chunks, which can be more cache-friendly
    for very large arrays.

    Parameters
    ----------
    array : array-like
        An array object that supports the __dlpack__() protocol
    chunk_size : int, optional
        Size of chunks to process (in bytes), default 1MB

    Returns
    -------
    int
        Hash value of the array
    """
    cdef DLManagedTensor* managed_tensor
    cdef DLTensor* dl_tensor
    cdef uint8_t* data_ptr
    cdef uint64_t total_bytes
    cdef uint64_t offset
    cdef uint64_t current_chunk_size
    cdef uint64_t hash_value
    cdef uint64_t chunk_hash
    cdef int i

    # Make a writable copy if needed
    if hasattr(array, 'flags') and not array.flags.writeable:
        if hasattr(array, 'copy'):
            array = array.copy()
        else:
            import numpy as np
            array = np.array(array, copy=True)

    # Get the DLPack capsule
    try:
        capsule = array.__dlpack__()
    except AttributeError:
        raise TypeError("Array does not support __dlpack__ protocol")

    if not PyCapsule_IsValid(capsule, b"dltensor"):
        raise ValueError("Invalid DLPack capsule")

    managed_tensor = <DLManagedTensor*>PyCapsule_GetPointer(capsule, b"dltensor")
    if managed_tensor == NULL:
        raise ValueError("Failed to get DLManagedTensor from capsule")

    dl_tensor = &managed_tensor.dl_tensor

    if dl_tensor.device.device_type != kDLCPU:
        raise ValueError("Array must be on CPU")

    # Calculate total size
    total_elements = 1
    for i in range(dl_tensor.ndim):
        total_elements *= dl_tensor.shape[i]

    element_size = dl_tensor.dtype.bits // 8
    total_bytes = total_elements * element_size

    data_ptr = <uint8_t*>dl_tensor.data + dl_tensor.byte_offset

    # Hash in chunks
    hash_value = 14695981039346656037UL  # FNV offset basis
    offset = 0

    while offset < total_bytes:
        current_chunk_size = min(chunk_size, total_bytes - offset)

        with nogil:
            chunk_hash = hash_bytes(data_ptr + offset, current_chunk_size)

        # Combine chunk hashes
        hash_value ^= chunk_hash
        hash_value *= 1099511628211UL  # FNV prime

        offset += current_chunk_size

    # Build shape and dtype info for complete hash
    shape_tuple = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
    dtype_info = (dl_tensor.dtype.code, dl_tensor.dtype.bits, dl_tensor.dtype.lanes)

    return hash((hash_value, shape_tuple, dtype_info))
