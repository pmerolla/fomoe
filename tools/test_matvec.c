#include "quant.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test mat_vec_mul with Q4_K against dequant + F32 dot
static int test_q4k_matvec(void) {
    printf("=== Q4_K mat_vec_mul vs dequant+F32 ===\n");

    int K = 3072;  // input dim (like n_embd)
    int M = 16;    // output dim (small for test)
    int blocks_per_row = K / QK_K;  // 12

    // Allocate Q4_K weight matrix [M rows of K elements]
    int total_blocks = M * blocks_per_row;
    block_q4_K *mat = calloc(total_blocks, sizeof(block_q4_K));

    // Fill with deterministic pattern
    for (int i = 0; i < total_blocks; i++) {
        // Give each block non-trivial values
        mat[i].d = fp32_to_fp16(0.5f + 0.1f * (i % 7));
        mat[i].dmin = fp32_to_fp16(0.1f + 0.05f * (i % 5));
        for (int j = 0; j < 12; j++) {
            mat[i].scales[j] = (uint8_t)(10 + (i + j) % 50);
        }
        for (int j = 0; j < QK_K / 2; j++) {
            mat[i].qs[j] = (uint8_t)((i * 7 + j * 3) & 0xFF);
        }
    }

    // Create input vector
    float *vec = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) {
        vec[i] = sinf((float)i * 0.01f) * 0.1f;
    }

    // Method 1: Quantized mat_vec_mul
    float *out_quant = calloc(M, sizeof(float));
    mat_vec_mul(out_quant, mat, GGML_TYPE_Q4_K, vec, M, K);

    // Method 2: Dequant to F32, then F32 dot product
    float *out_f32 = calloc(M, sizeof(float));
    float *dequant_row = malloc(K * sizeof(float));

    for (int i = 0; i < M; i++) {
        dequantize_row_q4_K(mat + i * blocks_per_row, dequant_row, K);
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += dequant_row[j] * vec[j];
        }
        out_f32[i] = sum;
    }

    // Compare
    float max_err = 0, max_rel = 0;
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        float err = fabsf(out_quant[i] - out_f32[i]);
        float rel = (fabsf(out_f32[i]) > 1e-6f) ? err / fabsf(out_f32[i]) : err;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
        if (rel > 0.01f) {  // 1% tolerance
            printf("  MISMATCH row %d: quant=%.6f, f32=%.6f, err=%.6f (%.4f%%)\n",
                   i, out_quant[i], out_f32[i], err, rel * 100);
            mismatches++;
        }
    }
    printf("  max_abs_err=%.6f, max_rel_err=%.6f, mismatches=%d/%d\n",
           max_err, max_rel, mismatches, M);

    free(mat); free(vec); free(out_quant); free(out_f32); free(dequant_row);
    return mismatches;
}

// Test mat_vec_mul with Q6_K against dequant + F32 dot
static int test_q6k_matvec(void) {
    printf("\n=== Q6_K mat_vec_mul vs dequant+F32 ===\n");

    int K = 3072;
    int M = 16;
    int blocks_per_row = K / QK_K;

    int total_blocks = M * blocks_per_row;
    block_q6_K *mat = calloc(total_blocks, sizeof(block_q6_K));

    for (int i = 0; i < total_blocks; i++) {
        mat[i].d = fp32_to_fp16(0.3f + 0.1f * (i % 5));
        for (int j = 0; j < QK_K / 2; j++) {
            mat[i].ql[j] = (uint8_t)((i * 5 + j * 7) & 0xFF);
        }
        for (int j = 0; j < QK_K / 4; j++) {
            mat[i].qh[j] = (uint8_t)((i * 3 + j * 11) & 0xFF);
        }
        for (int j = 0; j < QK_K / 16; j++) {
            mat[i].scales[j] = (int8_t)((i + j * 3) % 64 - 32);
        }
    }

    float *vec = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) {
        vec[i] = cosf((float)i * 0.01f) * 0.1f;
    }

    float *out_quant = calloc(M, sizeof(float));
    mat_vec_mul(out_quant, mat, GGML_TYPE_Q6_K, vec, M, K);

    float *out_f32 = calloc(M, sizeof(float));
    float *dequant_row = malloc(K * sizeof(float));

    for (int i = 0; i < M; i++) {
        dequantize_row_q6_K(mat + i * blocks_per_row, dequant_row, K);
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += dequant_row[j] * vec[j];
        }
        out_f32[i] = sum;
    }

    float max_err = 0, max_rel = 0;
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        float err = fabsf(out_quant[i] - out_f32[i]);
        float rel = (fabsf(out_f32[i]) > 1e-6f) ? err / fabsf(out_f32[i]) : err;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
        if (rel > 0.01f) {
            printf("  MISMATCH row %d: quant=%.6f, f32=%.6f, err=%.6f (%.4f%%)\n",
                   i, out_quant[i], out_f32[i], err, rel * 100);
            mismatches++;
        }
    }
    printf("  max_abs_err=%.6f, max_rel_err=%.6f, mismatches=%d/%d\n",
           max_err, max_rel, mismatches, M);

    free(mat); free(vec); free(out_quant); free(out_f32); free(dequant_row);
    return mismatches;
}

// Test with REAL model weights (gate expert from GGUF)
static int test_real_weights(const char *gguf_path) {
    printf("\n=== Real weight mat_vec_mul test (GGUF layer 0 expert 0 gate) ===\n");

    gguf_ctx_t *ctx = gguf_open(gguf_path);
    if (!ctx) return 1;

    int64_t tid = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (tid < 0) { gguf_close(ctx); return 1; }

    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    uint64_t per_expert = ti->size / ti->dims[2];

    // Read expert 5 (more likely to have non-zero weights)
    void *full = malloc(ti->size);
    gguf_read_tensor(ctx, tid, full, ti->size);
    block_q4_K *expert_gate = (block_q4_K *)((uint8_t *)full + 5 * per_expert);

    int K = (int)ti->dims[0];  // 3072
    int M = (int)ti->dims[1];  // 1024
    int blocks_per_row = K / QK_K;

    printf("  K=%d, M=%d, blocks_per_row=%d\n", K, M, blocks_per_row);

    // Create test input (simulating a normalized hidden state)
    float *vec = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) vec[i] = sinf((float)i * 0.05f) * 0.3f;

    // Method 1: Quantized
    float *out_quant = calloc(M, sizeof(float));
    mat_vec_mul(out_quant, expert_gate, GGML_TYPE_Q4_K, vec, M, K);

    // Method 2: Dequant + F32
    float *out_f32 = calloc(M, sizeof(float));
    float *dequant_row = malloc(K * sizeof(float));

    for (int i = 0; i < M; i++) {
        dequantize_row_q4_K(expert_gate + i * blocks_per_row, dequant_row, K);
        float sum = 0.0f;
        for (int j = 0; j < K; j++) sum += dequant_row[j] * vec[j];
        out_f32[i] = sum;
    }

    // Compare
    float max_err = 0, max_rel = 0;
    int mismatches = 0;
    int zeros_quant = 0, zeros_f32 = 0;
    for (int i = 0; i < M; i++) {
        if (out_quant[i] == 0.0f) zeros_quant++;
        if (out_f32[i] == 0.0f) zeros_f32++;
        float err = fabsf(out_quant[i] - out_f32[i]);
        float rel = (fabsf(out_f32[i]) > 1e-6f) ? err / fabsf(out_f32[i]) : err;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
        if (rel > 0.01f && err > 0.001f) {
            if (mismatches < 5)
                printf("  MISMATCH row %d: quant=%.6f, f32=%.6f, err=%.6f (%.4f%%)\n",
                       i, out_quant[i], out_f32[i], err, rel * 100);
            mismatches++;
        }
    }
    printf("  max_abs_err=%.6f, max_rel_err=%.6f, mismatches=%d/%d\n",
           max_err, max_rel, mismatches, M);
    printf("  zeros: quant=%d, f32=%d out of %d\n", zeros_quant, zeros_f32, M);

    // Print first 16 values from both
    printf("  quant[0..15]: ");
    for (int i = 0; i < 16; i++) printf("%.4f ", out_quant[i]);
    printf("\n  f32[0..15]:   ");
    for (int i = 0; i < 16; i++) printf("%.4f ", out_f32[i]);
    printf("\n");

    free(full); free(vec); free(out_quant); free(out_f32); free(dequant_row);
    gguf_close(ctx);
    return mismatches;
}

int main(int argc, char **argv) {
    int fails = 0;
    fails += test_q4k_matvec();
    fails += test_q6k_matvec();

    if (argc > 1) {
        fails += test_real_weights(argv[1]);
    }

    printf("\n=== Total: %d failures ===\n", fails);
    return fails > 0 ? 1 : 0;
}
