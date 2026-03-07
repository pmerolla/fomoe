/*
 * tp.hip — Tensor Parallelism communication for 2-GPU TP
 *
 * Host-staged AllReduce for PCIe-connected GPUs (no NVLink).
 * For 12KB payloads (n_embd=3072), host staging is simpler and
 * competitive with P2P since PCIe round-trip dominates either way.
 */

#include "tp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" tp_ctx_t *tp_init(int max_reduce_size) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count < 2) {
        fprintf(stderr, "TP: need 2 GPUs, found %d\n", device_count);
        return nullptr;
    }

    tp_ctx_t *ctx = (tp_ctx_t *)calloc(1, sizeof(tp_ctx_t));
    if (!ctx) return nullptr;

    ctx->n_devices = 2;
    ctx->device_ids[0] = 0;
    ctx->device_ids[1] = 1;
    ctx->max_reduce_size = max_reduce_size;

    // Print device info
    for (int d = 0; d < 2; d++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, d));
        fprintf(stderr, "TP: GPU%d: %s (%zu MB VRAM, %d CUs)\n",
                d, props.name, props.totalGlobalMem / (1024 * 1024),
                props.multiProcessorCount);
    }

    // Create streams and events on each device
    for (int d = 0; d < 2; d++) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamCreate(&ctx->compute[d]));
        HIP_CHECK(hipStreamCreate(&ctx->transfer[d]));
        HIP_CHECK(hipEventCreate(&ctx->reduce_ready[d]));
        HIP_CHECK(hipEventCreate(&ctx->reduce_done[d]));

        // Scratch buffer for AllReduce result on each GPU
        HIP_CHECK(hipMalloc(&ctx->d_reduced[d], max_reduce_size * sizeof(float)));
    }

    // Allocate pinned host buffers for staging
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipHostMalloc(&ctx->h_partial[0], max_reduce_size * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&ctx->h_partial[1], max_reduce_size * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&ctx->h_reduced, max_reduce_size * sizeof(float)));

    HIP_CHECK(hipSetDevice(0));

    fprintf(stderr, "TP: initialized 2-GPU TP, max_reduce_size=%d (%.1f KB)\n",
            max_reduce_size, max_reduce_size * sizeof(float) / 1024.0f);

    return ctx;
}

extern "C" void tp_free(tp_ctx_t *ctx) {
    if (!ctx) return;

    for (int d = 0; d < 2; d++) {
        HIP_CHECK(hipSetDevice(d));
        hipStreamDestroy(ctx->compute[d]);
        hipStreamDestroy(ctx->transfer[d]);
        hipEventDestroy(ctx->reduce_ready[d]);
        hipEventDestroy(ctx->reduce_done[d]);
        hipFree(ctx->d_reduced[d]);
    }

    HIP_CHECK(hipSetDevice(0));
    hipHostFree(ctx->h_partial[0]);
    hipHostFree(ctx->h_partial[1]);
    hipHostFree(ctx->h_reduced);

    free(ctx);
}

extern "C" void tp_allreduce_sum(tp_ctx_t *ctx,
                                  float *d_partial0, float *d_partial1,
                                  float *d_out0, float *d_out1, int n) {
    // Step 1: D2H both partials (in parallel on separate streams)
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpyAsync(ctx->h_partial[0], d_partial0,
                              n * sizeof(float), hipMemcpyDeviceToHost,
                              ctx->compute[0]));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[0]));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpyAsync(ctx->h_partial[1], d_partial1,
                              n * sizeof(float), hipMemcpyDeviceToHost,
                              ctx->compute[1]));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[1]));

    // Step 2: CPU reduce
    for (int i = 0; i < n; i++) {
        ctx->h_reduced[i] = ctx->h_partial[0][i] + ctx->h_partial[1][i];
    }

    // Step 3: H2D reduced result to both GPUs
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpyAsync(d_out0, ctx->h_reduced,
                              n * sizeof(float), hipMemcpyHostToDevice,
                              ctx->compute[0]));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpyAsync(d_out1, ctx->h_reduced,
                              n * sizeof(float), hipMemcpyHostToDevice,
                              ctx->compute[1]));

    // Sync both
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[1]));

    HIP_CHECK(hipSetDevice(0));
}

extern "C" void tp_allreduce_sum_async(tp_ctx_t *ctx,
                                        float *d_partial0, float *d_partial1,
                                        float *d_out0, float *d_out1, int n) {
    // D2H on transfer streams (non-blocking on compute)
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventRecord(ctx->reduce_ready[0], ctx->compute[0]));
    HIP_CHECK(hipStreamWaitEvent(ctx->transfer[0], ctx->reduce_ready[0], 0));
    HIP_CHECK(hipMemcpyAsync(ctx->h_partial[0], d_partial0,
                              n * sizeof(float), hipMemcpyDeviceToHost,
                              ctx->transfer[0]));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipEventRecord(ctx->reduce_ready[1], ctx->compute[1]));
    HIP_CHECK(hipStreamWaitEvent(ctx->transfer[1], ctx->reduce_ready[1], 0));
    HIP_CHECK(hipMemcpyAsync(ctx->h_partial[1], d_partial1,
                              n * sizeof(float), hipMemcpyDeviceToHost,
                              ctx->transfer[1]));

    // Store params for wait
    // (We store d_out pointers in reduced buffers temporarily — will need them in wait)
    // Actually, just use the sync version internally for now since the payload is tiny
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(ctx->transfer[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamSynchronize(ctx->transfer[1]));

    // CPU reduce
    for (int i = 0; i < n; i++) {
        ctx->h_reduced[i] = ctx->h_partial[0][i] + ctx->h_partial[1][i];
    }

    // H2D to both
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpyAsync(d_out0, ctx->h_reduced,
                              n * sizeof(float), hipMemcpyHostToDevice,
                              ctx->compute[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpyAsync(d_out1, ctx->h_reduced,
                              n * sizeof(float), hipMemcpyHostToDevice,
                              ctx->compute[1]));

    HIP_CHECK(hipSetDevice(0));
}

extern "C" void tp_allreduce_wait(tp_ctx_t *ctx) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[1]));
    HIP_CHECK(hipSetDevice(0));
}

extern "C" void tp_broadcast(tp_ctx_t *ctx, int src_rank, int dst_rank,
                               const void *d_src, void *d_dst, size_t bytes) {
    // Host-staged copy: src GPU → pinned host → dst GPU
    HIP_CHECK(hipSetDevice(src_rank));
    HIP_CHECK(hipMemcpyAsync(ctx->h_partial[0], d_src, bytes,
                              hipMemcpyDeviceToHost, ctx->transfer[src_rank]));
    HIP_CHECK(hipStreamSynchronize(ctx->transfer[src_rank]));

    HIP_CHECK(hipSetDevice(dst_rank));
    HIP_CHECK(hipMemcpyAsync(d_dst, ctx->h_partial[0], bytes,
                              hipMemcpyHostToDevice, ctx->compute[dst_rank]));
    HIP_CHECK(hipStreamSynchronize(ctx->compute[dst_rank]));

    HIP_CHECK(hipSetDevice(0));
}
