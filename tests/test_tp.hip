/*
 * test_tp.hip — TP-2 correctness and latency benchmarks
 *
 * Tests:
 *   1. AllReduce correctness (sum of known vectors)
 *   2. AllReduce latency for various payload sizes
 *   3. Column-parallel Q4_K matvec: each GPU does half the output rows
 *   4. Row-parallel Q4_K matvec: each GPU does half the input, AllReduce
 *   5. Combined TP matvec end-to-end latency vs single-GPU baseline
 *
 * Build: make test_tp USE_GPU=1
 * Run via rq: ./submit.py --name "TP bench" "./frontrunner-tp/tests/test_tp"
 */

#include "gpu_compat.h"
#include "tp.h"
#include "quant_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================
// FP16 helpers
// ============================================================

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = ldexpf((float)mant, -24);
        return sign ? -val : val;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float val = ldexpf((float)(mant + 1024), (int)exp - 25);
    return sign ? -val : val;
}

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } bits = { .f = f };
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int exp = ((bits.u >> 23) & 0xFF) - 127;
    uint32_t mant = bits.u & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (mant >> 13);
}

// ============================================================
// Q4_K CPU reference matvec
// ============================================================

static void cpu_q4k_matvec(const block_q4_K *weights, const float *input,
                            float *output, int M, int K) {
    int nblocks = K / QK_K;
    for (int row = 0; row < M; row++) {
        const block_q4_K *row_blocks = weights + (size_t)row * nblocks;
        float sum = 0.0f;
        for (int b = 0; b < nblocks; b++) {
            float d = fp16_to_fp32(row_blocks[b].d);
            float dmin = fp16_to_fp32(row_blocks[b].dmin);
            const uint8_t *sc = row_blocks[b].scales;
            const uint8_t *qs = row_blocks[b].qs;

            // Decode 6-bit packed scales/mins for 8 sub-blocks of 32
            for (int sb = 0; sb < 8; sb++) {
                int scale, min_val;
                if (sb < 4) {
                    scale = sc[sb] & 0x3F;
                    min_val = sc[sb + 4] & 0x3F;
                } else {
                    scale = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
                    min_val = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
                }

                for (int j = 0; j < 32; j++) {
                    int idx = sb * 32 + j;
                    int q;
                    if (idx < 128) {
                        q = qs[idx] & 0xF;
                    } else {
                        q = qs[idx - 128] >> 4;
                    }
                    float w = d * scale * q - dmin * min_val;
                    sum += w * input[b * QK_K + idx];
                }
            }
        }
        output[row] = sum;
    }
}

// ============================================================
// Q4_K GPU matvec kernel (1 warp per row, shared memory)
// From the prior codebase, adapted for standalone use
// ============================================================

__device__ static float device_fp16_to_fp32(uint16_t h) {
    __half hv;
    memcpy(&hv, &h, sizeof(uint16_t));
    return __half2float(hv);
}

// Single-row Q4_K matvec: 1 warp (32 threads) per row
// Each thread processes QK_K/32 = 8 elements per block
__global__ void kernel_q4k_matvec(
    const block_q4_K *__restrict__ W,  // [M, K/QK_K] blocks
    const float *__restrict__ x,        // [K]
    float *__restrict__ out,            // [M]
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int tid = threadIdx.x;  // 0..31
    int nblocks = K / QK_K;
    const block_q4_K *row_blocks = W + (size_t)row * nblocks;

    float sum = 0.0f;

    for (int b = 0; b < nblocks; b++) {
        float d = device_fp16_to_fp32(row_blocks[b].d);
        float dmin = device_fp16_to_fp32(row_blocks[b].dmin);
        const uint8_t *sc = row_blocks[b].scales;
        const uint8_t *qs = row_blocks[b].qs;

        // Each thread handles 8 elements (tid covers 32 elements per sub-block)
        // Process 2 sub-blocks per thread-group, 4 iterations
        for (int it = 0; it < 4; it++) {
            int sb = it * 2 + (tid >= 16 ? 1 : 0);
            int j = tid & 15;

            int scale, min_val;
            if (sb < 4) {
                scale = sc[sb] & 0x3F;
                min_val = sc[sb + 4] & 0x3F;
            } else {
                scale = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
                min_val = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
            }

            // Low nibble
            int idx_lo = sb * 32 + j;
            int q_lo = (idx_lo < 128) ? (qs[idx_lo] & 0xF) : (qs[idx_lo - 128] >> 4);
            float w_lo = d * scale * q_lo - dmin * min_val;
            sum += w_lo * x[b * QK_K + idx_lo];

            // High nibble
            int idx_hi = sb * 32 + j + 16;
            int q_hi = (idx_hi < 128) ? (qs[idx_hi] & 0xF) : (qs[idx_hi - 128] >> 4);
            float w_hi = d * scale * q_hi - dmin * min_val;
            sum += w_hi * x[b * QK_K + idx_hi];
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        out[row] = sum;
    }
}

// ============================================================
// Test helpers
// ============================================================

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void fill_random_q4k(block_q4_K *blocks, int nblocks) {
    srand(42);
    for (int b = 0; b < nblocks; b++) {
        blocks[b].d = fp32_to_fp16(0.01f * (rand() % 100 + 1));
        blocks[b].dmin = fp32_to_fp16(0.005f * (rand() % 50 + 1));
        for (int i = 0; i < K_SCALE_SIZE; i++)
            blocks[b].scales[i] = rand() & 0xFF;
        for (int i = 0; i < QK_K / 2; i++)
            blocks[b].qs[i] = rand() & 0xFF;
    }
}

static void fill_random_f32(float *arr, int n) {
    srand(123);
    for (int i = 0; i < n; i++)
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

static float max_rel_diff(const float *a, const float *b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float denom = fabsf(a[i]) > 1e-6f ? fabsf(a[i]) : 1e-6f;
        float d = fabsf(a[i] - b[i]) / denom;
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// ============================================================
// Test 1: AllReduce correctness
// ============================================================

static int test_allreduce_correctness(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 1: AllReduce Correctness ===\n");

    const int N = 3072;  // n_embd
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_result = (float *)malloc(N * sizeof(float));

    // Known values: a[i] = i, b[i] = 1000 + i, expected = 1000 + 2*i
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = 1000.0f + (float)i;
    }

    float *d_a, *d_b, *d_out0, *d_out1;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out0, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out1, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(0));

    // AllReduce
    tp_allreduce_sum(tp, d_a, d_b, d_out0, d_out1, N);

    // Verify GPU0 result
    HIP_CHECK(hipMemcpy(h_result, d_out0, N * sizeof(float), hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = 1000.0f + 2.0f * i;
        if (fabsf(h_result[i] - expected) > 0.001f) {
            if (errors < 5)
                fprintf(stderr, "  MISMATCH at %d: got %.3f, expected %.3f\n",
                        i, h_result[i], expected);
            errors++;
        }
    }

    // Verify GPU1 result
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(h_result, d_out1, N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));

    for (int i = 0; i < N; i++) {
        float expected = 1000.0f + 2.0f * i;
        if (fabsf(h_result[i] - expected) > 0.001f) {
            errors++;
        }
    }

    if (errors == 0)
        fprintf(stderr, "  PASS: AllReduce sum correct on both GPUs (%d elements)\n", N);
    else
        fprintf(stderr, "  FAIL: %d mismatches\n", errors);

    hipFree(d_a); hipFree(d_out0);
    HIP_CHECK(hipSetDevice(1));
    hipFree(d_b); hipFree(d_out1);
    HIP_CHECK(hipSetDevice(0));
    free(h_a); free(h_b); free(h_result);

    return errors == 0 ? 0 : 1;
}

// ============================================================
// Test 2: AllReduce latency benchmark
// ============================================================

static void test_allreduce_latency(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 2: AllReduce Latency ===\n");

    int sizes[] = {3072, 8192, 16384, 32768};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup = 50;
    int iters = 200;

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        if (N > tp->max_reduce_size) continue;

        float *d_a, *d_b, *d_out0, *d_out1;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out0, N * sizeof(float)));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out1, N * sizeof(float)));
        HIP_CHECK(hipSetDevice(0));

        // Warmup
        for (int i = 0; i < warmup; i++)
            tp_allreduce_sum(tp, d_a, d_b, d_out0, d_out1, N);

        // Benchmark
        double t0 = get_time_ms();
        for (int i = 0; i < iters; i++)
            tp_allreduce_sum(tp, d_a, d_b, d_out0, d_out1, N);
        double t1 = get_time_ms();

        double avg_us = (t1 - t0) / iters * 1000.0;
        fprintf(stderr, "  N=%5d (%6.1f KB): %.1f us/allreduce\n",
                N, N * 4.0f / 1024.0f, avg_us);

        hipFree(d_a); hipFree(d_out0);
        HIP_CHECK(hipSetDevice(1));
        hipFree(d_b); hipFree(d_out1);
        HIP_CHECK(hipSetDevice(0));
    }
}

// ============================================================
// Test 3: Column-parallel Q4_K matvec correctness
// ============================================================

static int test_column_parallel_matvec(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 3: Column-Parallel Q4_K MatVec ===\n");

    // Simulate Q/K/V projection: [M, K] × [K] → [M]
    // Column-parallel: GPU0 does rows [0, M/2), GPU1 does rows [M/2, M)
    const int M = 3072;   // output dim (n_embd or n_heads * head_dim)
    const int K = 3072;   // input dim (n_embd)
    const int M_half = M / 2;
    int nblocks_per_row = K / QK_K;
    size_t total_blocks = (size_t)M * nblocks_per_row;

    // Allocate and fill weights + input on host
    block_q4_K *h_W = (block_q4_K *)malloc(total_blocks * sizeof(block_q4_K));
    float *h_x = (float *)malloc(K * sizeof(float));
    float *h_ref = (float *)malloc(M * sizeof(float));
    float *h_out = (float *)malloc(M * sizeof(float));

    fill_random_q4k(h_W, total_blocks);
    fill_random_f32(h_x, K);

    // CPU reference: full matvec
    cpu_q4k_matvec(h_W, h_x, h_ref, M, K);

    // GPU0: first half of rows
    block_q4_K *h_W0 = h_W;  // rows [0, M/2)
    block_q4_K *h_W1 = h_W + (size_t)M_half * nblocks_per_row;  // rows [M/2, M)

    block_q4_K *d_W0, *d_W1;
    float *d_x0, *d_x1, *d_out0, *d_out1;

    size_t half_weight_bytes = (size_t)M_half * nblocks_per_row * sizeof(block_q4_K);

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_W0, half_weight_bytes));
    HIP_CHECK(hipMalloc(&d_x0, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out0, M_half * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_W0, h_W0, half_weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x0, h_x, K * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMalloc(&d_W1, half_weight_bytes));
    HIP_CHECK(hipMalloc(&d_x1, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out1, M_half * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_W1, h_W1, half_weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x1, h_x, K * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(0));

    // Launch column-parallel matvec on both GPUs
    constexpr int RPB = 8;
    dim3 block(32, RPB);

    HIP_CHECK(hipSetDevice(0));
    dim3 grid0((M_half + RPB - 1) / RPB);
    kernel_q4k_matvec<<<grid0, block, 0, tp->compute[0]>>>(
        d_W0, d_x0, d_out0, M_half, K);
    HIP_CHECK(hipStreamSynchronize(tp->compute[0]));

    HIP_CHECK(hipSetDevice(1));
    dim3 grid1((M_half + RPB - 1) / RPB);
    kernel_q4k_matvec<<<grid1, block, 0, tp->compute[1]>>>(
        d_W1, d_x1, d_out1, M_half, K);
    HIP_CHECK(hipStreamSynchronize(tp->compute[1]));

    // Copy results back
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpy(h_out, d_out0, M_half * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(h_out + M_half, d_out1, M_half * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));

    // Compare
    float maxd = max_abs_diff(h_ref, h_out, M);
    float maxr = max_rel_diff(h_ref, h_out, M);

    int pass = (maxr < 0.01f);  // 1% relative tolerance for quantized ops
    fprintf(stderr, "  %s: max_abs_diff=%.6f, max_rel_diff=%.4f%%\n",
            pass ? "PASS" : "FAIL", maxd, maxr * 100.0f);
    fprintf(stderr, "  [M=%d, K=%d, M/2=%d per GPU]\n", M, K, M_half);

    // Cleanup
    hipFree(d_W0); hipFree(d_x0); hipFree(d_out0);
    HIP_CHECK(hipSetDevice(1));
    hipFree(d_W1); hipFree(d_x1); hipFree(d_out1);
    HIP_CHECK(hipSetDevice(0));
    free(h_W); free(h_x); free(h_ref); free(h_out);

    return pass ? 0 : 1;
}

// ============================================================
// Test 4: Row-parallel Q4_K matvec with AllReduce
// ============================================================

static int test_row_parallel_matvec(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 4: Row-Parallel Q4_K MatVec + AllReduce ===\n");

    // Simulate down/O projection: [M, K] × [K] → [M]
    // Row-parallel: split K in half. Each GPU has [M, K/2] weights.
    // Each GPU computes partial output, then AllReduce.
    const int M = 3072;   // output dim (n_embd)
    const int K = 1024;   // input dim (expert_intermediate or head_dim * n_heads)
    const int K_half = K / 2;

    int nblocks_full = K / QK_K;
    int nblocks_half = K_half / QK_K;

    // Full weights for CPU reference
    block_q4_K *h_W = (block_q4_K *)malloc((size_t)M * nblocks_full * sizeof(block_q4_K));
    float *h_x = (float *)malloc(K * sizeof(float));
    float *h_ref = (float *)malloc(M * sizeof(float));
    float *h_out = (float *)malloc(M * sizeof(float));

    fill_random_q4k(h_W, (size_t)M * nblocks_full);
    fill_random_f32(h_x, K);

    // CPU reference
    cpu_q4k_matvec(h_W, h_x, h_ref, M, K);

    // Split weights: GPU0 gets first K/2 columns, GPU1 gets last K/2 columns
    // For Q4_K, a "row" has nblocks_full blocks. We split each row in half.
    size_t half_row_blocks = nblocks_half;
    block_q4_K *h_W0 = (block_q4_K *)malloc((size_t)M * half_row_blocks * sizeof(block_q4_K));
    block_q4_K *h_W1 = (block_q4_K *)malloc((size_t)M * half_row_blocks * sizeof(block_q4_K));

    for (int row = 0; row < M; row++) {
        memcpy(h_W0 + (size_t)row * half_row_blocks,
               h_W + (size_t)row * nblocks_full,
               half_row_blocks * sizeof(block_q4_K));
        memcpy(h_W1 + (size_t)row * half_row_blocks,
               h_W + (size_t)row * nblocks_full + half_row_blocks,
               half_row_blocks * sizeof(block_q4_K));
    }

    // Split input: GPU0 gets x[0:K/2], GPU1 gets x[K/2:K]
    float *h_x0 = h_x;
    float *h_x1 = h_x + K_half;

    block_q4_K *d_W0, *d_W1;
    float *d_x0, *d_x1, *d_partial0, *d_partial1, *d_out0, *d_out1;

    size_t half_weight_bytes = (size_t)M * half_row_blocks * sizeof(block_q4_K);

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_W0, half_weight_bytes));
    HIP_CHECK(hipMalloc(&d_x0, K_half * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial0, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out0, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_W0, h_W0, half_weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x0, h_x0, K_half * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMalloc(&d_W1, half_weight_bytes));
    HIP_CHECK(hipMalloc(&d_x1, K_half * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial1, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out1, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_W1, h_W1, half_weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x1, h_x1, K_half * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipSetDevice(0));

    // Row-parallel matvec: each GPU computes partial output
    constexpr int RPB = 8;
    dim3 block(32, RPB);
    dim3 grid((M + RPB - 1) / RPB);

    HIP_CHECK(hipSetDevice(0));
    kernel_q4k_matvec<<<grid, block, 0, tp->compute[0]>>>(
        d_W0, d_x0, d_partial0, M, K_half);

    HIP_CHECK(hipSetDevice(1));
    kernel_q4k_matvec<<<grid, block, 0, tp->compute[1]>>>(
        d_W1, d_x1, d_partial1, M, K_half);

    // AllReduce to sum partials
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamSynchronize(tp->compute[1]));
    HIP_CHECK(hipSetDevice(0));

    tp_allreduce_sum(tp, d_partial0, d_partial1, d_out0, d_out1, M);

    // Copy result from GPU0
    HIP_CHECK(hipMemcpy(h_out, d_out0, M * sizeof(float), hipMemcpyDeviceToHost));

    // Compare
    float maxd = max_abs_diff(h_ref, h_out, M);
    float maxr = max_rel_diff(h_ref, h_out, M);

    int pass = (maxr < 0.01f);
    fprintf(stderr, "  %s: max_abs_diff=%.6f, max_rel_diff=%.4f%%\n",
            pass ? "PASS" : "FAIL", maxd, maxr * 100.0f);
    fprintf(stderr, "  [M=%d, K=%d, K/2=%d per GPU, AllReduce %d floats]\n",
            M, K, K_half, M);

    // Cleanup
    hipFree(d_W0); hipFree(d_x0); hipFree(d_partial0); hipFree(d_out0);
    HIP_CHECK(hipSetDevice(1));
    hipFree(d_W1); hipFree(d_x1); hipFree(d_partial1); hipFree(d_out1);
    HIP_CHECK(hipSetDevice(0));
    free(h_W); free(h_W0); free(h_W1); free(h_x); free(h_ref); free(h_out);

    return pass ? 0 : 1;
}

// ============================================================
// Test 5: TP matvec latency vs single-GPU baseline
// ============================================================

static void test_tp_matvec_latency(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 5: TP MatVec Latency vs Single-GPU ===\n");

    // Simulate the key matvecs in the model:
    //   - QKV projection: [3072, 3072] (column-parallel, no allreduce)
    //   - O projection: [3072, 8192] (row-parallel, needs allreduce)
    //   - Router: [3072, 512] (column-parallel, small)
    //   - Expert gate+up: [1024, 3072] (column-parallel per expert)
    //   - Expert down: [3072, 1024] (row-parallel per expert)

    struct { const char *name; int M; int K; bool row_parallel; } configs[] = {
        {"QKV proj  [3072,3072]", 3072, 3072, false},
        {"O proj    [3072,8192]", 3072, 8192, true},
        {"Router    [512,3072]",   512, 3072, false},
        {"Expert GU [1024,3072]", 1024, 3072, false},
        {"Expert Dn [3072,1024]", 3072, 1024, true},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    int warmup = 20;
    int iters = 100;

    for (int ci = 0; ci < nconfigs; ci++) {
        int M = configs[ci].M;
        int K = configs[ci].K;

        int nblocks = K / QK_K;
        size_t weight_bytes = (size_t)M * nblocks * sizeof(block_q4_K);

        block_q4_K *h_W = (block_q4_K *)malloc(weight_bytes);
        float *h_x = (float *)malloc(K * sizeof(float));
        fill_random_q4k(h_W, (size_t)M * nblocks);
        fill_random_f32(h_x, K);

        // --- Single GPU baseline ---
        block_q4_K *d_W_full;
        float *d_x_full, *d_out_full;

        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&d_W_full, weight_bytes));
        HIP_CHECK(hipMalloc(&d_x_full, K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out_full, M * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_W_full, h_W, weight_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_x_full, h_x, K * sizeof(float), hipMemcpyHostToDevice));

        constexpr int RPB = 8;
        dim3 block(32, RPB);
        dim3 grid_full((M + RPB - 1) / RPB);

        for (int i = 0; i < warmup; i++) {
            kernel_q4k_matvec<<<grid_full, block, 0, tp->compute[0]>>>(
                d_W_full, d_x_full, d_out_full, M, K);
            HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
        }

        double t0 = get_time_ms();
        for (int i = 0; i < iters; i++) {
            kernel_q4k_matvec<<<grid_full, block, 0, tp->compute[0]>>>(
                d_W_full, d_x_full, d_out_full, M, K);
            HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
        }
        double single_ms = (get_time_ms() - t0) / iters;

        hipFree(d_W_full); hipFree(d_x_full); hipFree(d_out_full);

        // --- TP-2 ---
        double tp_ms;

        if (configs[ci].row_parallel) {
            // Row-parallel: each GPU has [M, K/2], AllReduce after
            int K_half = K / 2;
            int nb_half = K_half / QK_K;

            block_q4_K *h_W0 = (block_q4_K *)malloc((size_t)M * nb_half * sizeof(block_q4_K));
            block_q4_K *h_W1 = (block_q4_K *)malloc((size_t)M * nb_half * sizeof(block_q4_K));
            for (int r = 0; r < M; r++) {
                memcpy(h_W0 + (size_t)r * nb_half, h_W + (size_t)r * nblocks,
                       nb_half * sizeof(block_q4_K));
                memcpy(h_W1 + (size_t)r * nb_half, h_W + (size_t)r * nblocks + nb_half,
                       nb_half * sizeof(block_q4_K));
            }

            block_q4_K *d_W0, *d_W1;
            float *d_x0, *d_x1, *d_p0, *d_p1, *d_o0, *d_o1;

            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMalloc(&d_W0, (size_t)M * nb_half * sizeof(block_q4_K)));
            HIP_CHECK(hipMalloc(&d_x0, K_half * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_p0, M * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_o0, M * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_W0, h_W0, (size_t)M * nb_half * sizeof(block_q4_K), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_x0, h_x, K_half * sizeof(float), hipMemcpyHostToDevice));

            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipMalloc(&d_W1, (size_t)M * nb_half * sizeof(block_q4_K)));
            HIP_CHECK(hipMalloc(&d_x1, K_half * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_p1, M * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_o1, M * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_W1, h_W1, (size_t)M * nb_half * sizeof(block_q4_K), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_x1, h_x + K_half, K_half * sizeof(float), hipMemcpyHostToDevice));

            HIP_CHECK(hipSetDevice(0));
            dim3 grid((M + RPB - 1) / RPB);

            for (int i = 0; i < warmup; i++) {
                HIP_CHECK(hipSetDevice(0));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[0]>>>(d_W0, d_x0, d_p0, M, K_half);
                HIP_CHECK(hipSetDevice(1));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[1]>>>(d_W1, d_x1, d_p1, M, K_half);
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipStreamSynchronize(tp->compute[1]));
                HIP_CHECK(hipSetDevice(0));
                tp_allreduce_sum(tp, d_p0, d_p1, d_o0, d_o1, M);
            }

            t0 = get_time_ms();
            for (int i = 0; i < iters; i++) {
                HIP_CHECK(hipSetDevice(0));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[0]>>>(d_W0, d_x0, d_p0, M, K_half);
                HIP_CHECK(hipSetDevice(1));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[1]>>>(d_W1, d_x1, d_p1, M, K_half);
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipStreamSynchronize(tp->compute[1]));
                HIP_CHECK(hipSetDevice(0));
                tp_allreduce_sum(tp, d_p0, d_p1, d_o0, d_o1, M);
            }
            tp_ms = (get_time_ms() - t0) / iters;

            hipFree(d_W0); hipFree(d_x0); hipFree(d_p0); hipFree(d_o0);
            HIP_CHECK(hipSetDevice(1));
            hipFree(d_W1); hipFree(d_x1); hipFree(d_p1); hipFree(d_o1);
            HIP_CHECK(hipSetDevice(0));
            free(h_W0); free(h_W1);
        } else {
            // Column-parallel: each GPU has [M/2, K], no AllReduce
            int M_half = M / 2;
            int nb = nblocks;

            block_q4_K *d_W0, *d_W1;
            float *d_x0, *d_x1, *d_o0, *d_o1;

            size_t half_bytes = (size_t)M_half * nb * sizeof(block_q4_K);

            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMalloc(&d_W0, half_bytes));
            HIP_CHECK(hipMalloc(&d_x0, K * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_o0, M_half * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_W0, h_W, half_bytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_x0, h_x, K * sizeof(float), hipMemcpyHostToDevice));

            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipMalloc(&d_W1, half_bytes));
            HIP_CHECK(hipMalloc(&d_x1, K * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_o1, M_half * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_W1, (char *)h_W + half_bytes, half_bytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_x1, h_x, K * sizeof(float), hipMemcpyHostToDevice));

            HIP_CHECK(hipSetDevice(0));
            dim3 grid((M_half + RPB - 1) / RPB);

            for (int i = 0; i < warmup; i++) {
                HIP_CHECK(hipSetDevice(0));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[0]>>>(d_W0, d_x0, d_o0, M_half, K);
                HIP_CHECK(hipSetDevice(1));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[1]>>>(d_W1, d_x1, d_o1, M_half, K);
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipStreamSynchronize(tp->compute[1]));
                HIP_CHECK(hipSetDevice(0));
            }

            t0 = get_time_ms();
            for (int i = 0; i < iters; i++) {
                HIP_CHECK(hipSetDevice(0));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[0]>>>(d_W0, d_x0, d_o0, M_half, K);
                HIP_CHECK(hipSetDevice(1));
                kernel_q4k_matvec<<<grid, block, 0, tp->compute[1]>>>(d_W1, d_x1, d_o1, M_half, K);
                HIP_CHECK(hipSetDevice(0));
                HIP_CHECK(hipStreamSynchronize(tp->compute[0]));
                HIP_CHECK(hipSetDevice(1));
                HIP_CHECK(hipStreamSynchronize(tp->compute[1]));
                HIP_CHECK(hipSetDevice(0));
            }
            tp_ms = (get_time_ms() - t0) / iters;

            hipFree(d_W0); hipFree(d_x0); hipFree(d_o0);
            HIP_CHECK(hipSetDevice(1));
            hipFree(d_W1); hipFree(d_x1); hipFree(d_o1);
            HIP_CHECK(hipSetDevice(0));
        }

        float speedup = (float)(single_ms / tp_ms);
        fprintf(stderr, "  %s: single=%.3fms  TP-2=%.3fms  speedup=%.2fx%s\n",
                configs[ci].name, single_ms, tp_ms, speedup,
                configs[ci].row_parallel ? " (incl. AllReduce)" : "");

        free(h_W); free(h_x);
    }
}

// ============================================================
// Test 6: VRAM capacity estimate
// ============================================================

static void test_vram_capacity(tp_ctx_t *tp) {
    fprintf(stderr, "\n=== Test 6: VRAM Capacity Estimate ===\n");

    // Query free memory on each GPU
    for (int d = 0; d < 2; d++) {
        HIP_CHECK(hipSetDevice(d));
        size_t free_mem, total_mem;
        HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
        fprintf(stderr, "  GPU%d: %.1f MB free / %.1f MB total\n",
                d, free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));

        // Estimate: how much VRAM for non-expert model weights?
        // Attention/SSM per layer: ~25 MB (Q4_K/Q6_K/MXFP4)
        // 60 layers → 1500 MB for all attention/SSM weights
        // With TP-2, each GPU holds half → ~750 MB
        // KV cache: 2 * 15 full-attn layers * 4096 * 2 * 256 * 2 bytes ≈ 120 MB (FP16)
        //   With TP-2 each GPU has half the heads → 60 MB
        // SSM states: 45 layers * 64 * 128 * 128 * 4 bytes ≈ 189 MB
        //   With TP-2 each GPU has half the heads → 95 MB
        // Scratch buffers: ~50 MB
        // Total non-expert overhead per GPU: ~955 MB
        // Available for expert cache: free - 955 MB

        float overhead_mb = 955.0f;
        float expert_stride_mb = 7.25f;  // 397B expert size
        float cache_mb = free_mem / (1024.0f * 1024.0f) - overhead_mb;
        if (cache_mb < 0) cache_mb = 0;
        int n_expert_slots = (int)(cache_mb / expert_stride_mb);
        int slots_per_layer = n_expert_slots / 60;

        fprintf(stderr, "  GPU%d: ~%.0f MB for expert cache → %d slots (%d/layer)\n",
                d, cache_mb, n_expert_slots, slots_per_layer);
    }

    // Combined estimate
    fprintf(stderr, "\n  Combined TP-2 expert cache (both GPUs, non-overlapping):\n");
    fprintf(stderr, "  Estimated ~1400-1500 slots total (~24/layer for 397B)\n");
    fprintf(stderr, "  vs single-GPU: ~720 slots (~12/layer)\n");
    fprintf(stderr, "  Hit rate improvement: ~48%% → ~70%%+ estimated\n");

    HIP_CHECK(hipSetDevice(0));
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv) {
    fprintf(stderr, "=== Frontrunner TP-2 Test Suite ===\n");

    // Max reduce size: largest we'll need is vocab_size (151936) for final logits
    // But for layer-level ops, 8192 is the max (ssm_inner_size)
    tp_ctx_t *tp = tp_init(32768);
    if (!tp) {
        fprintf(stderr, "FATAL: tp_init failed\n");
        return 1;
    }

    int failures = 0;

    failures += test_allreduce_correctness(tp);
    test_allreduce_latency(tp);
    failures += test_column_parallel_matvec(tp);
    failures += test_row_parallel_matvec(tp);
    test_tp_matvec_latency(tp);
    test_vram_capacity(tp);

    fprintf(stderr, "\n=== Summary: %d failures ===\n", failures);

    tp_free(tp);
    return failures;
}
