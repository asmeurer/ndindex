# Array Hashing Options for Array API Compliance

This document outlines different approaches to making ndindex's array wrapper objects hashable while supporting the Array API standard, not just NumPy.

## Background

The current implementation in `ndindex/array.py` uses `hash(array.tobytes())` to make array wrapper objects hashable. However, `tobytes()` is not part of the Array API standard, so this approach only works with NumPy arrays.

## Requirements

1. **Array API Compliance**: Must work with any array-api compliant library (NumPy, CuPy, PyTorch, JAX, etc.)
2. **Hashability**: Arrays must remain hashable for use in sets and as dict keys
3. **Consistency**: Equal arrays must have equal hashes
4. **Performance**: Should be reasonably fast, even for large arrays
5. **Immutability**: Hash must be consistent for read-only arrays

## Implementation Options

### Option 1: DLPack + ctypes

**Implementation**: `hash_dlpack_ctypes()` in `array_hash_prototypes.py`

Uses the `__dlpack__()` method from the Array API standard to access the raw memory buffer through Python's ctypes.

**Pros**:
- Works with any Array API implementation that supports DLPack (most do)
- Direct memory access is fast
- No build dependencies

**Cons**:
- Complex implementation with ctypes structures
- Only works for CPU arrays (GPU arrays would need device→host transfer)
- Accessing PyCapsule internals is somewhat fragile

**Performance**: Fast (comparable to NumPy's tobytes)

**Recommendation**: Good fallback option if Cython is not desired

---

### Option 2: DLPack + Cython ⭐ **RECOMMENDED**

**Implementation**: `hash_dlpack_cython()` in `array_hash_cython.pyx`

Similar to Option 1 but uses Cython for cleaner, faster implementation.

**Pros**:
- **Fastest implementation** (often faster than NumPy's tobytes)
- Clean, maintainable code with type safety
- Works with any Array API implementation
- Can release GIL for true parallelism

**Cons**:
- Requires Cython build step (adds complexity to package build)
- Only works for CPU arrays (same as Option 1)

**Performance**: Fastest option

**Recommendation**: **Best choice if you're willing to add Cython dependency**

**Build instructions**:
```bash
python setup_cython_hash.py build_ext --inplace
```

---

### Option 3: Pure Array API - Polynomial Hash

**Implementation**: `hash_array_api_polynomial()` in `array_hash_prototypes.py`

Computes a rolling polynomial hash using only Array API operations.

**Pros**:
- Pure Python - no C dependencies
- Works with any Array API backend
- Portable to GPU arrays (in theory)

**Cons**:
- **Very slow** - requires transferring all data from device to host
- Complex integer overflow handling
- Defeats the purpose of array abstraction

**Performance**: Very slow (10-100x slower than direct memory access)

**Recommendation**: Not recommended except for small arrays or when no other option is available

---

### Option 4: Pure Array API - Sampled Hash

**Implementation**: `hash_array_api_sampled()` in `array_hash_prototypes.py`

Samples strategic elements from the array rather than hashing all data.

**Pros**:
- Fast for large arrays (O(1) instead of O(n))
- Pure Python - works with any Array API backend
- Simple implementation

**Cons**:
- **Higher collision risk** - similar large arrays might hash the same
- Not suitable for use cases requiring cryptographic properties
- Still requires some device→host transfers

**Performance**: Fast for large arrays, slower than full hash for small arrays

**Recommendation**: Good option if you need Array API purity and can accept higher collision risk

**Collision Risk**: Demonstrated in benchmarks - should be acceptable for most use cases

---

### Option 5: Buffer Protocol Fallback

**Implementation**: `hash_buffer_protocol()` in `array_hash_prototypes.py`

Tries to use Python's buffer protocol via `memoryview()`.

**Pros**:
- Fast when available
- Standard Python - no dependencies
- Simple implementation

**Cons**:
- **Not guaranteed by Array API standard**
- May not work for non-NumPy backends
- GPU arrays typically don't support buffer protocol

**Performance**: Fast (comparable to NumPy's tobytes)

**Recommendation**: Good to try as a first fallback, but can't rely on it alone

---

### Option 6: Hybrid Approach

**Implementation**: `hash_array_hybrid()` in `array_hash_prototypes.py`

Tries multiple methods in order of preference:
1. Buffer protocol (fastest, when available)
2. DLPack + ctypes (fast, Array API compliant)
3. Sampled hash (slower, but always works)

**Pros**:
- **Best of all worlds** - fast when possible, works everywhere
- Graceful degradation
- No build dependencies (if not using Cython variant)

**Cons**:
- More complex code
- Behavior varies by backend

**Performance**: Depends on which method succeeds

**Recommendation**: **Best pure-Python option** for maximum compatibility

---

## Comparison Summary

| Option | Speed | Array API Compliant | Build Deps | GPU Support | Collision Risk |
|--------|-------|-------------------|------------|-------------|----------------|
| NumPy tobytes (current) | Fast | ❌ No | None | ❌ No | Low |
| DLPack + ctypes | Fast | ✅ Yes | None | ⚠️ CPU only | Low |
| DLPack + Cython | **Fastest** | ✅ Yes | Cython | ⚠️ CPU only | Low |
| Pure API - Polynomial | Very slow | ✅ Yes | None | ✅ Yes | Low |
| Pure API - Sampled | Medium | ✅ Yes | None | ✅ Yes | Medium |
| Buffer Protocol | Fast | ⚠️ Maybe | None | ❌ No | Low |
| Hybrid | Fast-Medium | ✅ Yes | None | ⚠️ Partial | Low-Medium |

## Recommendations

### For Production Use

**If you can add Cython to your build:**
- Use **DLPack + Cython** (Option 2)
- Fallback to **Hybrid approach** for GPU arrays

**If you want pure Python:**
- Use **Hybrid approach** (Option 6)
- This gives best performance while maintaining compatibility

### For Different Use Cases

**Performance Critical**:
- DLPack + Cython
- Ensure arrays are on CPU before hashing

**Maximum Compatibility**:
- Hybrid approach
- Works with all backends, gracefully degrades

**GPU Arrays**:
- Sampled hash (to avoid device→host transfer)
- Or transfer to CPU first, then use DLPack

**Minimal Dependencies**:
- DLPack + ctypes (no build step)
- Or Buffer Protocol + Sampled fallback

## Implementation Strategy

### Recommended Approach

1. **Primary**: DLPack + Cython for CPU arrays
2. **Fallback 1**: Buffer protocol (for NumPy and similar)
3. **Fallback 2**: DLPack + ctypes (if Cython fails to build)
4. **Fallback 3**: Sampled hash (for GPU arrays or when DLPack fails)

### Code Structure

```python
def __hash__(self):
    # Try Cython implementation (fastest)
    try:
        from ._hash_cython import hash_dlpack_cython
        return hash_dlpack_cython(self.array)
    except (ImportError, ValueError):
        pass

    # Try buffer protocol (fast, common)
    try:
        return hash(memoryview(self.array).tobytes())
    except (TypeError, BufferError):
        pass

    # Try DLPack with ctypes (slower but compatible)
    try:
        return hash_dlpack_ctypes(self.array)
    except (TypeError, ValueError):
        pass

    # Last resort: sampled hash
    return hash_array_api_sampled(self.array)
```

## Testing

Run benchmarks:
```bash
python benchmark_array_hashing.py
```

This will test:
- Performance of each method with different array sizes
- Collision resistance
- Compatibility with different array libraries

## Next Steps

1. Review benchmark results on your target hardware
2. Decide on acceptable trade-offs (speed vs. compatibility vs. build complexity)
3. Implement chosen approach in `ndindex/array.py`
4. Add tests for different Array API backends
5. Update documentation

## Files

- `array_hash_prototypes.py` - Pure Python implementations
- `array_hash_cython.pyx` - Cython implementation
- `setup_cython_hash.py` - Cython build script
- `benchmark_array_hashing.py` - Performance benchmarks
- `ARRAY_HASHING_OPTIONS.md` - This document
