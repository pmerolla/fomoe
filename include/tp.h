/*
 * tp.h — Tensor Parallelism communication primitives for 2-GPU inference
 *
 * TP-2 strategy:
 *   Column-parallel: split output rows across GPUs (gate/up/Q/K/V projections)
 *     - No communication needed, each GPU computes its portion
 *   Row-parallel: split input columns across GPUs (down/O projections)
 *     - AllReduce needed to sum partial results
 *
 * Communication path: PCIe Gen5 x8 (~16 GB/s per direction)
 * Typical AllReduce payload: 3072 floats = 12 KB → ~1μs at wire speed
 * Actual measured: ~20-50μs including kernel launch + sync overhead
 *
 * For 2 GPUs without NVLink, we use host-staged AllReduce:
 *   GPU0 D2H partial → CPU add → H2D result to both GPUs
 * This avoids P2P topology issues on consumer AMD boards.
 */

#ifndef FR_TP_H
#define FR_TP_H

#include "gpu_compat.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TP context: manages 2-GPU communication state
typedef struct tp_ctx {
    int n_devices;        // always 2 for TP-2
    int device_ids[2];    // HIP device IDs

    // Per-device streams
    hipStream_t compute[2];    // main compute streams
    hipStream_t transfer[2];   // H2D/D2H transfer streams

    // Sync events
    hipEvent_t reduce_ready[2]; // GPU signals partial result ready
    hipEvent_t reduce_done[2];  // signals AllReduce complete

    // Host-staged AllReduce buffers (pinned)
    float *h_partial[2];       // [max_reduce_size] per GPU
    float *h_reduced;          // [max_reduce_size] final reduced result
    int max_reduce_size;       // max elements per AllReduce

    // Scratch device buffers for AllReduce results
    float *d_reduced[2];       // [max_reduce_size] on each GPU
} tp_ctx_t;

// Initialize TP context with 2 GPUs.
// max_reduce_size: largest vector we'll ever AllReduce (e.g. n_embd=3072)
tp_ctx_t *tp_init(int max_reduce_size);

// Free TP context
void tp_free(tp_ctx_t *ctx);

// AllReduce: sum partial results from both GPUs → broadcast result to both.
// d_partial[rank]: device pointer on GPU[rank] with partial result
// d_out[rank]: device pointer on GPU[rank] to receive reduced result
// n: number of float elements
// Uses compute streams. Blocks until complete.
void tp_allreduce_sum(tp_ctx_t *ctx, float *d_partial0, float *d_partial1,
                      float *d_out0, float *d_out1, int n);

// Async AllReduce: non-blocking version.
// After calling, d_out[rank] will be valid after tp_allreduce_wait().
void tp_allreduce_sum_async(tp_ctx_t *ctx, float *d_partial0, float *d_partial1,
                            float *d_out0, float *d_out1, int n);

// Wait for async AllReduce to complete on both GPUs.
void tp_allreduce_wait(tp_ctx_t *ctx);

// Broadcast: copy data from GPU src_rank to GPU dst_rank.
// d_src: device pointer on GPU[src_rank]
// d_dst: device pointer on GPU[dst_rank]
// bytes: number of bytes to copy
void tp_broadcast(tp_ctx_t *ctx, int src_rank, int dst_rank,
                  const void *d_src, void *d_dst, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // FR_TP_H
