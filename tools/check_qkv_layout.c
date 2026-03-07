#include "gguf.h"
#include "quant.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Check if the attn_qkv output is in contiguous [Q,K,V] or grouped interleaved format
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    gguf_ctx_t *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    // Read embedding for token 0
    int64_t embd_tid = gguf_find_tensor(ctx, "token_embd.weight");
    const gguf_tensor_info_t *embd_info = &ctx->tensors[embd_tid];
    int n_embd = (int)embd_info->dims[0];  // 3072

    void *embd_raw = malloc(embd_info->size);
    gguf_read_tensor(ctx, embd_tid, embd_raw, embd_info->size);

    // Dequant token 1000's embedding
    int token = 1000;
    int blocks_per_row = n_embd / QK_K;
    float *x = malloc(n_embd * sizeof(float));
    dequantize_row_q4_K((block_q4_K *)embd_raw + token * blocks_per_row, x, n_embd);
    free(embd_raw);

    // RMS norm with unit weight (just normalize)
    float ss = 0;
    for (int i = 0; i < n_embd; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n_embd + 1e-6f);
    for (int i = 0; i < n_embd; i++) x[i] *= ss;

    printf("Input norm: %.4f\n", ss);

    // Read attn_qkv weight for layer 0
    int64_t qkv_tid = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    const gguf_tensor_info_t *qkv_info = &ctx->tensors[qkv_tid];
    int qkv_dim = (int)qkv_info->dims[1];  // 12288

    printf("attn_qkv: type=%s, dims=[%ld,%ld]\n",
           ggml_type_name(qkv_info->type), qkv_info->dims[0], qkv_info->dims[1]);

    void *qkv_data = malloc(qkv_info->size);
    gguf_read_tensor(ctx, qkv_tid, qkv_data, qkv_info->size);

    // Compute QKV = W @ x
    float *qkv = calloc(qkv_dim, sizeof(float));
    mat_vec_mul(qkv, qkv_data, qkv_info->type, x, qkv_dim, n_embd);

    int ng = 16;   // number of key groups
    int sd = 128;  // key/state dim per group
    int kd = ng * sd;  // 2048
    int hpg = 4;   // value heads per group
    int vd = 128;  // value dim per head
    int inner = 64 * vd;  // 8192

    printf("\n=== Hypothesis A: Contiguous [Q(2048), K(2048), V(8192)] ===\n");
    {
        float *Q = qkv;
        float *K = qkv + kd;
        float *V = qkv + kd + kd;

        // Check Q norms per group
        printf("  Q per-group L2 norms: ");
        for (int g = 0; g < ng; g++) {
            float norm = 0;
            for (int i = 0; i < sd; i++) norm += Q[g*sd+i] * Q[g*sd+i];
            printf("%.3f ", sqrtf(norm));
        }
        printf("\n");

        printf("  K per-group L2 norms: ");
        for (int g = 0; g < ng; g++) {
            float norm = 0;
            for (int i = 0; i < sd; i++) norm += K[g*sd+i] * K[g*sd+i];
            printf("%.3f ", sqrtf(norm));
        }
        printf("\n");

        printf("  V mean_abs per head (first 8): ");
        for (int h = 0; h < 8; h++) {
            float s = 0;
            for (int i = 0; i < vd; i++) s += fabsf(V[h*vd+i]);
            printf("%.4f ", s / vd);
        }
        printf("\n");

        // Q[0..7]:
        printf("  Q[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", Q[i]);
        printf("\n  K[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", K[i]);
        printf("\n  V[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", V[i]);
        printf("\n");
    }

    printf("\n=== Hypothesis B: Interleaved [16 × (Q(128), K(128), V(512))] ===\n");
    {
        int per_group = sd + sd + hpg * vd;  // 128+128+512 = 768
        // Separate into contiguous arrays
        float *Q = calloc(kd, sizeof(float));
        float *K = calloc(kd, sizeof(float));
        float *V = calloc(inner, sizeof(float));

        for (int g = 0; g < ng; g++) {
            float *grp = qkv + g * per_group;
            memcpy(Q + g * sd, grp, sd * sizeof(float));
            memcpy(K + g * sd, grp + sd, sd * sizeof(float));
            memcpy(V + g * hpg * vd, grp + sd + sd, hpg * vd * sizeof(float));
        }

        printf("  Q per-group L2 norms: ");
        for (int g = 0; g < ng; g++) {
            float norm = 0;
            for (int i = 0; i < sd; i++) norm += Q[g*sd+i] * Q[g*sd+i];
            printf("%.3f ", sqrtf(norm));
        }
        printf("\n");

        printf("  K per-group L2 norms: ");
        for (int g = 0; g < ng; g++) {
            float norm = 0;
            for (int i = 0; i < sd; i++) norm += K[g*sd+i] * K[g*sd+i];
            printf("%.3f ", sqrtf(norm));
        }
        printf("\n");

        printf("  V mean_abs per head (first 8): ");
        for (int h = 0; h < 8; h++) {
            float s = 0;
            for (int i = 0; i < vd; i++) s += fabsf(V[h*vd+i]);
            printf("%.4f ", s / vd);
        }
        printf("\n");

        printf("  Q[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", Q[i]);
        printf("\n  K[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", K[i]);
        printf("\n  V[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", V[i]);
        printf("\n");

        // Check variance of norms as a diagnostic:
        // For the correct split, all Q group norms should be similar (model is trained this way)
        // For wrong split, norms would vary wildly
        float q_mean = 0, q_var = 0;
        float k_mean = 0, k_var = 0;
        float q_norms[16], k_norms[16];
        for (int g = 0; g < ng; g++) {
            float qn = 0, kn = 0;
            for (int i = 0; i < sd; i++) {
                qn += Q[g*sd+i] * Q[g*sd+i];
                kn += K[g*sd+i] * K[g*sd+i];
            }
            q_norms[g] = sqrtf(qn);
            k_norms[g] = sqrtf(kn);
            q_mean += q_norms[g];
            k_mean += k_norms[g];
        }
        q_mean /= ng; k_mean /= ng;
        for (int g = 0; g < ng; g++) {
            q_var += (q_norms[g] - q_mean) * (q_norms[g] - q_mean);
            k_var += (k_norms[g] - k_mean) * (k_norms[g] - k_mean);
        }
        printf("  Q norm variance: %.6f (mean=%.4f)\n", q_var / ng, q_mean);
        printf("  K norm variance: %.6f (mean=%.4f)\n", k_var / ng, k_mean);

        free(Q); free(K); free(V);
    }

    // Also check Hypothesis A's norm variance
    {
        float *Q = qkv;
        float *K = qkv + kd;
        float q_mean = 0, q_var = 0, k_mean = 0, k_var = 0;
        float q_norms[16], k_norms[16];
        for (int g = 0; g < ng; g++) {
            float qn = 0, kn = 0;
            for (int i = 0; i < sd; i++) {
                qn += Q[g*sd+i] * Q[g*sd+i];
                kn += K[g*sd+i] * K[g*sd+i];
            }
            q_norms[g] = sqrtf(qn);
            k_norms[g] = sqrtf(kn);
            q_mean += q_norms[g]; k_mean += k_norms[g];
        }
        q_mean /= ng; k_mean /= ng;
        for (int g = 0; g < ng; g++) {
            q_var += (q_norms[g] - q_mean) * (q_norms[g] - q_mean);
            k_var += (k_norms[g] - k_mean) * (k_norms[g] - k_mean);
        }
        printf("\n  Hypothesis A norm variance: Q=%.6f (mean=%.4f), K=%.6f (mean=%.4f)\n",
               q_var / ng, q_mean, k_var / ng, k_mean);
    }

    free(qkv); free(qkv_data); free(x);
    gguf_close(ctx);
    return 0;
}
