# Array API Hashing Implementation - Master Index

## 🚀 Quick Start (30 seconds)

1. Read: `README_START_HERE.md` (6.4 KB)
2. Run: `python demo_hashing.py`
3. Use: Copy `hash_array_api_fast()` from `array_hash_optimized.py`

**Bottom line**: On-device hash is **5x faster** for large arrays, fully Array API compliant, and fixes the collision bug.

---

## 📁 File Guide

### Start Here 👈
| File | Size | Purpose |
|------|------|---------|
| **`README_START_HERE.md`** | 6.4 KB | **Start here!** Quick overview and decision tree |
| **`FINAL_RECOMMENDATION.md`** | 7.1 KB | Complete recommendation with code examples |
| **`demo_hashing.py`** | 7.3 KB | Interactive demo - run this to see everything |

### Implementation Files (Copy From Here)
| File | Size | Purpose |
|------|------|---------|
| **`array_hash_optimized.py`** | 9.4 KB | **✅ On-device implementations** (RECOMMENDED) |
| `array_hash_prototypes.py` | 12 KB | Alternative approaches (hybrid, sampled, etc.) |
| `array_hash_cython.pyx` | 7.3 KB | Cython version (slower, not recommended) |

### Documentation & Analysis
| File | Size | Purpose |
|------|------|---------|
| `SUMMARY.md` | 8.2 KB | Original comprehensive summary |
| `BENCHMARK_RESULTS.md` | 7.4 KB | Detailed performance analysis |
| `ARRAY_HASHING_OPTIONS.md` | 7.8 KB | Deep dive into all 6+ approaches |
| `README_ARRAY_HASHING.md` | 7.0 KB | Integration guide |

### Testing & Benchmarking
| File | Size | Purpose |
|------|------|---------|
| `benchmark_array_hashing.py` | 11 KB | Comprehensive benchmark suite |
| `demo_hashing.py` | 7.3 KB | Interactive demonstration |

### Build Files (Optional)
| File | Size | Purpose |
|------|------|---------|
| `setup_cython_hash.py` | 0.5 KB | Cython build script (not needed) |
| `array_hash_cython.c` | 530 KB | Generated C code (not needed) |

---

## 📊 The Problem

Your current implementation:
```python
def __hash__(self):
    return hash(self.array.tobytes())
```

Has two critical issues:
1. ❌ **Collision bug**: `[1,2,3,4]` and `[[1,2],[3,4]]` hash the same
2. ❌ **NumPy-only**: `tobytes()` is not in the Array API standard

---

## 🎯 The Solution

**Use on-device hash from `array_hash_optimized.py`:**

```python
from array_hash_optimized import hash_array_api_fast

class ArrayIndex(NDIndex):
    def __hash__(self):
        return hash_array_api_fast(self.array)
```

### Why This Works

| Aspect | Result |
|--------|--------|
| **Performance** | 5x faster for large arrays (1M elements) |
| **Array API** | ✅ Uses only standard operations |
| **GPU Support** | ✅ Computes on device, no host transfer |
| **Collision Bug** | ✅ Fixed - includes shape and dtype |
| **Dependencies** | ✅ None - pure Python |
| **Backend Support** | ✅ NumPy, PyTorch, CuPy, JAX, etc. |

---

## 📈 Performance Summary

### Large Arrays (1M elements) - The Key Result
- **Current (buggy)**: 2.71 ms
- **On-device hash**: 0.52 ms
- **Speedup**: **5.2x faster** 🚀

### Medium Arrays (10K elements)
- **Current**: 0.021 ms
- **On-device hash**: 0.019 ms
- **Speedup**: 1.1x faster

### Small Arrays (100 elements)
- **Current**: 0.0004 ms
- **On-device hash**: 0.0076 ms
- **Result**: 19x slower but still <0.01ms (negligible)

---

## 🔍 Understanding The Approaches

### 6 Implementation Options Explored

1. **On-Device Hash** ⭐ **RECOMMENDED**
   - Computes hash using Array API operations on device
   - 5x faster for large arrays
   - File: `array_hash_optimized.py`

2. **Hybrid Approach**
   - Tries buffer protocol, falls back to DLPack/sampled
   - Good for mixed workloads
   - File: `array_hash_prototypes.py` → `hash_array_hybrid()`

3. **DLPack + ctypes**
   - Direct memory access via DLPack protocol
   - Good but slower than on-device
   - File: `array_hash_prototypes.py` → `hash_dlpack_ctypes()`

4. **Sampled Hash**
   - O(1) - only hashes sample of elements
   - 62x faster but higher collision risk
   - File: `array_hash_prototypes.py` → `hash_array_api_sampled()`

5. **DLPack + Cython**
   - Cython implementation
   - Actually 4x slower due to copy overhead
   - File: `array_hash_cython.pyx`

6. **Buffer Protocol**
   - Fastest when available
   - Not guaranteed by Array API
   - File: `array_hash_prototypes.py` → `hash_buffer_protocol()`

---

## 🎬 How To Use This

### Path 1: I Want The Answer (5 minutes)
1. Read `README_START_HERE.md`
2. Run `python demo_hashing.py`
3. Copy `hash_array_api_fast()` from `array_hash_optimized.py`
4. Done!

### Path 2: I Want To Understand (20 minutes)
1. Read `README_START_HERE.md`
2. Read `FINAL_RECOMMENDATION.md`
3. Run `python demo_hashing.py`
4. Run `python benchmark_array_hashing.py`
5. Review `BENCHMARK_RESULTS.md`

### Path 3: I Want Deep Dive (1 hour)
1. All of Path 2, plus:
2. Read `ARRAY_HASHING_OPTIONS.md` - detailed explanation of each approach
3. Read `SUMMARY.md` - original comprehensive analysis
4. Review `array_hash_optimized.py` - implementation details
5. Review `array_hash_prototypes.py` - all alternatives

---

## 🧪 Testing

### Run The Demo
```bash
python demo_hashing.py
```

Shows:
- The collision bug
- All implementations fixing it
- Performance comparison
- Backend compatibility (NumPy, PyTorch, etc.)

### Run Full Benchmarks
```bash
python benchmark_array_hashing.py
```

Takes ~2 minutes, tests:
- Multiple array sizes (100 to 1M elements)
- Different data types (int, float, bool)
- Various shapes (1D, 2D)
- Collision resistance
- Multiple backends

---

## 🔧 Integration Checklist

- [ ] Read `README_START_HERE.md`
- [ ] Read `FINAL_RECOMMENDATION.md`
- [ ] Run `python demo_hashing.py`
- [ ] Choose implementation (recommended: `hash_array_api_fast`)
- [ ] Copy code from `array_hash_optimized.py`
- [ ] Update `ArrayIndex.__hash__()` in `ndindex/array.py`
- [ ] Add test for collision resistance (different shapes)
- [ ] Add test for hash consistency (equal arrays)
- [ ] Test with array-api-strict (optional)
- [ ] Update documentation about Array API support
- [ ] Add to CHANGELOG

---

## 💡 Key Insights

### Why On-Device Hash Is Faster

**Old approach**:
1. Call `array.tobytes()` → creates full copy of array as bytes
2. Hash the bytes object
3. For large arrays, copying is expensive

**New approach**:
1. Compute weighted sum using array operations (on device)
2. Extract single scalar to host
3. For large arrays, avoids expensive copy

### Why It Works On GPU

**Old approach with GPU array**:
1. Transfer entire array from GPU → CPU (`tobytes()` requires CPU)
2. Hash on CPU
3. Slow for large GPU arrays

**New approach with GPU array**:
1. Compute weighted sum on GPU
2. Transfer only final hash value (one scalar) to CPU
3. Fast! Only one number crosses device boundary

---

## 📚 Research Summary

This implementation is based on:
- Analysis of Array API standard 2024.12
- Benchmarking of 6+ different approaches
- Testing with NumPy, PyTorch, CuPy, array-api-strict
- Inspiration from xxHash algorithm
- Stack Overflow discussions on tensor hashing

Key findings:
1. `tobytes()` is not in Array API standard
2. Current implementation has collision bug
3. On-device computation is faster for large arrays
4. Only integer/boolean arrays needed (ndindex use case)
5. DLPack has issues with readonly arrays
6. Cython is slower due to copy overhead
7. Sampled hashing is O(1) but has collision risks

---

## 🎓 What Was Learned

### About Array API
- DLPack is the standard interchange protocol
- No `tobytes()` or direct memory access
- Must use array operations for device-agnostic code
- `sum()`, `reshape()`, `arange()` are sufficient

### About Performance
- Copying large arrays is expensive
- Device→host transfer is bottleneck for GPU
- Python hash() is fast enough for single values
- O(n) on-device is faster than O(n) copy + O(n) hash

### About Hashing
- Must include shape and dtype to avoid collisions
- Position-weighted sums have good distribution
- Sampling trades quality for speed
- Prime multipliers help distribution

---

## 🏆 Final Recommendation

**Use `hash_array_api_fast()` from `array_hash_optimized.py`**

It's the best choice because:
- ✅ 5x faster for large arrays (where it matters)
- ✅ Same speed for medium arrays
- ✅ Fixes collision bug
- ✅ Full Array API compliance
- ✅ GPU-friendly
- ✅ No dependencies
- ✅ Simple to integrate

See `FINAL_RECOMMENDATION.md` for complete details.

---

## 📞 Questions?

All questions are answered in:
- `README_START_HERE.md` - FAQ section
- `FINAL_RECOMMENDATION.md` - Q&A section
- `BENCHMARK_RESULTS.md` - Performance questions
- `ARRAY_HASHING_OPTIONS.md` - Technical details

---

**Created**: 2025-10-11
**Total Files**: 11 implementation/documentation files + 2 build artifacts
**Total Size**: ~600 KB (mostly generated C code)
**Useful Size**: ~75 KB (implementation + docs)
**Key File**: `array_hash_optimized.py` (9.4 KB)
