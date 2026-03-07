# FP16 KV Cache Implementation Notes

## Summary
Converted the GPU KV cache from FP32 to FP16 (`__half`). Halves VRAM usage (120 MB → 60 MB for 397B) with no measurable speed impact and no quality degradation. Performance was ~7.1 tok/s, same as F32 baseline.

## Why It Didn't Speed Things Up
GQA attention is only **~7% of per-token time** (9-10 ms out of ~140 ms total). Even halving bandwidth doesn't meaningfully move the needle. The real bottlenecks are MoE FFN pipeline (55%) and SSM/DeltaNet (24%).

## Changes Required (4 locations in gpu_kernels.hip)

### 1. Struct declaration (~line 459)
```cpp
// OLD:
float *d_kv_k;  // [n_full_attn * ctx_len * n_kv_heads * head_dim]
float *d_kv_v;  // same

// NEW:
__half *d_kv_k;  // [n_full_attn * ctx_len * n_kv_heads * head_dim]
__half *d_kv_v;  // same
```

### 2. KV cache allocation (~line 3038)
```cpp
// OLD:
size_t kv_layer_bytes = (size_t)ctx->ctx_len * kv_dim * sizeof(float);

// NEW:
size_t kv_layer_bytes = (size_t)ctx->ctx_len * kv_dim * sizeof(__half);
```

### 3. KV update kernel (~line 1791)
```cpp
// OLD:
__global__ void kernel_kv_update(float *kv_k, float *kv_v,
                                  const float *k_in, const float *v_in, ...) {
    kv_k[base + i] = k_in[i];
    kv_v[base + i] = v_in[i];
}

// NEW:
__global__ void kernel_kv_update(__half *kv_k, __half *kv_v,
                                  const float *k_in, const float *v_in, ...) {
    kv_k[base + i] = __float2half(k_in[i]);
    kv_v[base + i] = __float2half(v_in[i]);
}
```

### 4. GQA attention kernel (~line 1810)
```cpp
// Change signature:
const __half *d_kv_k,  // was: const float *d_kv_k
const __half *d_kv_v,  // was: const float *d_kv_v

// Change K reads in Phase 1:
dot += q_head[d] * __half2float(d_kv_k[k_base + d]);  // was: d_kv_k[k_base + d]

// Change V reads in Phase 3:
acc += scores[t] * __half2float(d_kv_v[v_base + d]);  // was: d_kv_v[v_base + d]
```

## Online Softmax Approach (explored, not faster)
Also tried rewriting the 3-phase attention kernel with single-pass online softmax (FlashAttention style). Two variants tested:

### Variant A: 128 threads, 4 warps, cross-warp sync per timestep
- Problem: `__syncthreads()` per timestep (100 barriers at seq_len=100)
- Result: ~6.74 tok/s (slightly slower than baseline)

### Variant B: 32 threads, 1 warp per head, warp shuffles only
- Design: each thread handles head_dim/32=8 elements, warp-level reduction via `__shfl_xor_sync`
- No shared memory, no `__syncthreads`
- Problem: only 32 blocks × 32 threads = 1024 total threads, very low GPU occupancy
- Result: ~6.69 tok/s (no improvement)

### Why online softmax didn't help
At seq_len=100, the old 3-phase kernel is fine:
- Phase 1: threads split across time steps (embarrassingly parallel)
- Phase 2: 5 sync barriers total (not per-timestep)
- Phase 3: threads split across head_dim (embarrassingly parallel)
The old structure has better parallelism than online softmax for short sequences.

### HIP note for `__shfl_xor_sync`
ROCm 7.1 requires 64-bit mask: `0xFFFFFFFFFFFFFFFFULL` (not `0xFFFFFFFF`)

## Known Issue: VRAM Fragmentation
The expert cache allocation (5.2 GB contiguous) sometimes fails even with 6.1 GB free due to VRAM fragmentation. When this happens, the model still works but is very slow (no VRAM cache). The `gpu_seed_vram_cache` function returns early when ecache.max_slots==0, which also skips RAM cache seeding. This could be fixed by seeding RAM cache independently of VRAM cache.

## What Would Actually Speed Up Attention
1. The real opportunity is **SSM/DeltaNet** (24% of time vs 7% for GQA)
2. Flash Linear Attention research shows 3.3× speedup from kernel fusion
3. Current DeltaNet kernel makes 3 sequential passes over 128×128 state with no shared memory
