#include "inference.h"
#include "quant.h"
#include "tensor.h"
#ifdef QMOE_GPU
#include "gpu.h"
#endif

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ---- Routing prediction experiment state ----
static int g_pred_enabled = -1;
static int g_pred_skip_ids[16], g_pred_same_ids[16], g_pred_n = 0;
static int g_pred_total_skip = 0, g_pred_total_same = 0, g_pred_total_n = 0;

// ---- Runtime debug stats ----
static int g_debug = -1;
static void dbg_stats(const char *label, const float *buf, int n, int layer) {
    if (g_debug <= 0) return;
    float mn = buf[0], mx = buf[0], s = 0;
    int nans = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(buf[i]) || isinf(buf[i])) { nans++; continue; }
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        s += fabsf(buf[i]);
    }
    fprintf(stderr, "  [L%02d] %-24s min=%10.4f max=%10.4f mean_abs=%8.4f",
            layer, label, mn, mx, s / n);
    if (nans) fprintf(stderr, " NaN/Inf=%d", nans);
    fprintf(stderr, "\n");
}

// ---- Partial RoPE ----
// NEOX-style partial RoPE: rotates pairs (j, j + rope_dim/2) for j in [0, rope_dim/2).
// Frequencies use head_dim as denominator (matching ggml's theta_scale = freq_base^(-2/ne0)):
//   freq[j] = 1 / freq_base^(2*j / head_dim)
// Only the first rope_dim dimensions are rotated; remaining dimensions pass through.
static void apply_rope_partial(float *q, float *k, int pos, int head_dim,
                                int rope_dim, int n_heads, int n_kv_heads,
                                float freq_base) {
    int half = rope_dim / 2;
    for (int h = 0; h < n_heads + n_kv_heads; h++) {
        float *vec = (h < n_heads) ? q + h * head_dim : k + (h - n_heads) * head_dim;
        for (int j = 0; j < half; j++) {
            // Use rope_dim as denominator - matches ggml's theta_scale = freq_base^(-2/n_dims)
            float freq = 1.0f / powf(freq_base, (float)(2 * j) / rope_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float v0 = vec[j];
            float v1 = vec[j + half];
            vec[j]        = v0 * cos_a - v1 * sin_a;
            vec[j + half] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// ---- Per-head RMS normalization ----
static void per_head_rms_norm(float *out, const float *x, const float *weight,
                               int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + h * head_dim;
        float *oh = out + h * head_dim;
        rms_norm(oh, xh, weight, head_dim, eps);
    }
}

// ---- Gated Full Attention ----
// Q projection produces [2 * n_heads * head_dim]: first half is Q, second half is output gate.
// After attention, output is element-wise multiplied by sigmoid(gate).
static void gated_gqa_attention(model_t *m, int layer, int pos) {
    const model_hparams_t *hp = &m->hparams;
    layer_weights_t *l = &m->layers[layer];
    int head_dim = hp->head_dim;
    int n_heads = hp->n_heads;
    int n_kv_heads = hp->n_kv_heads;
    int inner_dim = n_heads * head_dim;  // 8192

    float *h = m->buf_h;

    // Q projection: produces [full_q_dim] = [2 * inner_dim]
    // Output is INTERLEAVED per-head: [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...]
    mat_vec_mul(m->buf_qkv, l->wq, l->wq_type, h, hp->full_q_dim, hp->n_embd);

    // Deinterleave Q and gate into contiguous blocks: [Q_all(inner_dim), Gate_all(inner_dim)]
    float *tmp_qg = malloc(hp->full_q_dim * sizeof(float));
    if (!tmp_qg) return;
    memcpy(tmp_qg, m->buf_qkv, hp->full_q_dim * sizeof(float));
    for (int h_idx = 0; h_idx < n_heads; h_idx++) {
        memcpy(m->buf_qkv + h_idx * head_dim,
               tmp_qg + h_idx * 2 * head_dim, head_dim * sizeof(float));
        memcpy(m->buf_qkv + inner_dim + h_idx * head_dim,
               tmp_qg + h_idx * 2 * head_dim + head_dim, head_dim * sizeof(float));
    }
    free(tmp_qg);

    float *q = m->buf_qkv;              // [inner_dim]
    float *gate = m->buf_qkv + inner_dim; // [inner_dim]

    // K, V projections
    uint32_t kv_dim = n_kv_heads * head_dim;
    mat_vec_mul(m->buf_k, l->wk, l->wk_type, h, kv_dim, hp->n_embd);
    mat_vec_mul(m->buf_v, l->wv, l->wv_type, h, kv_dim, hp->n_embd);

    // Per-head RMS norm
    if (l->q_norm)
        per_head_rms_norm(q, q, l->q_norm, n_heads, head_dim, hp->rms_norm_eps);
    if (l->k_norm)
        per_head_rms_norm(m->buf_k, m->buf_k, l->k_norm, n_kv_heads, head_dim, hp->rms_norm_eps);

    // Partial RoPE
    apply_rope_partial(q, m->buf_k, pos, head_dim, hp->rope_dim,
                       n_heads, n_kv_heads, hp->rope_freq_base);

    // Update KV cache
    memcpy(m->kv_cache[layer].k + pos * kv_dim, m->buf_k, kv_dim * sizeof(float));
    memcpy(m->kv_cache[layer].v + pos * kv_dim, m->buf_v, kv_dim * sizeof(float));

    // GQA attention
    int heads_per_kv = n_heads / n_kv_heads;
    int seq_len = pos + 1;
    float *attn_out = m->buf_attn;
    memset(attn_out, 0, inner_dim * sizeof(float));

    float *scores = malloc(seq_len * sizeof(float));
    if (!scores) return;

    for (int gh = 0; gh < n_heads; gh++) {
        int kv_h = gh / heads_per_kv;
        float *q_head = q + gh * head_dim;
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int t = 0; t < seq_len; t++) {
            float *k_t = m->kv_cache[layer].k + t * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_t[d];
            scores[t] = dot * scale;
        }
        softmax(scores, seq_len);

        float *out_head = attn_out + gh * head_dim;
        for (int t = 0; t < seq_len; t++) {
            float *v_t = m->kv_cache[layer].v + t * kv_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_head[d] += scores[t] * v_t[d];
        }
    }
    free(scores);

    // Apply output gate: attn_out *= sigmoid(gate)
    for (int i = 0; i < inner_dim; i++) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        attn_out[i] *= g;
    }

    // Output projection: [inner_dim] -> [n_embd]
    mat_vec_mul(m->buf_ffn, l->wo, l->wo_type, attn_out, hp->n_embd, inner_dim);
}

// ---- Gated DeltaNet SSM Attention ----
// Architecture (Qwen3.5 Gated DeltaNet):
//   1. h -> attn_qkv -> [Q(ng*sd), K(ng*sd), V(inner)]
//   2. Conv1d on the fused QKV, then SiLU on ALL of Q+K+V
//   3. L2-normalize Q and K per group
//   4. h -> ssm_alpha -> decay, h -> ssm_beta -> beta (sigmoid)
//   5. Delta rule state update: S = decay*S + outer(k, beta*(v - k^T@S))
//   6. Output: y = (S^T @ q) / sqrt(sd)
//   7. Per-head RMS norm, gated by SiLU(attn_gate @ h)
//   8. ssm_out projection

static void deltanet_ssm(model_t *m, int layer) {
    const model_hparams_t *hp = &m->hparams;
    layer_weights_t *l = &m->layers[layer];
    ssm_state_t *ss = &m->ssm_state[layer];

    uint32_t inner   = hp->ssm_inner_size;      // 8192
    uint32_t sd      = hp->ssm_state_size;       // 128 (key/state dim per group)
    uint32_t ng      = hp->ssm_group_count;      // 16
    uint32_t nh      = hp->ssm_time_step_rank;   // 64 (number of heads)
    uint32_t vd      = hp->ssm_value_dim;        // 128 (value dim per head)
    uint32_t qkv_dim = hp->ssm_qkv_dim;         // 12288
    uint32_t conv_k  = hp->ssm_conv_kernel;      // 4
    uint32_t kd      = ng * sd;                   // 2048 (total key dim)

    float *h = m->buf_h;
    float *qkv = m->buf_qkv;

    // Step 1: QKV projection
    mat_vec_mul(qkv, l->attn_qkv, l->attn_qkv_type, h, qkv_dim, hp->n_embd);

    // Step 2: Causal Conv1d + SiLU on entire QKV
    if (l->ssm_conv1d) {
        float *tmp = malloc(qkv_dim * sizeof(float));
        if (!tmp) return;

        // Current position uses weight[channel][conv_k-1] (newest kernel position)
        for (uint32_t j = 0; j < qkv_dim; j++) {
            tmp[j] = l->ssm_conv1d[j * conv_k + (conv_k - 1)] * qkv[j];
        }

        // Past positions from circular buffer
        for (uint32_t t = 1; t < conv_k; t++) {
            int past_idx = (int)ss->conv_pos - (int)t;
            if (past_idx < 0) continue;
            past_idx = past_idx % (int)(conv_k - 1);
            float *past = ss->conv_buf + past_idx * qkv_dim;
            uint32_t w_idx = conv_k - 1 - t;
            for (uint32_t j = 0; j < qkv_dim; j++) {
                tmp[j] += l->ssm_conv1d[j * conv_k + w_idx] * past[j];
            }
        }

        // FIX #4: SiLU activation on ALL of QKV (not just value portion)
        for (uint32_t j = 0; j < qkv_dim; j++) {
            tmp[j] = tmp[j] / (1.0f + expf(-tmp[j]));
        }

        // Store current input in circular buffer AFTER computing conv output
        int buf_pos = ss->conv_pos % (int)(conv_k - 1);
        memcpy(ss->conv_buf + buf_pos * qkv_dim, qkv, qkv_dim * sizeof(float));
        ss->conv_pos++;

        memcpy(qkv, tmp, qkv_dim * sizeof(float));
        free(tmp);
    }

    // FIX #3: Correct QKV split order: [Q(kd), K(kd), V(inner)]
    // Q = query (ng * sd = 2048), K = key (ng * sd = 2048), V = value (inner = 8192)
    float *Q = qkv;                        // [ng * sd] = [2048] query
    float *K = qkv + kd;                   // [ng * sd] = [2048] key
    float *V = qkv + kd + kd;              // [inner]   = [8192] value

    // FIX #2: L2-normalize Q and K per group
    for (uint32_t g = 0; g < ng; g++) {
        float *q_g = Q + g * sd;
        float *k_g = K + g * sd;
        float q_norm = 0.0f, k_norm = 0.0f;
        for (uint32_t i = 0; i < sd; i++) {
            q_norm += q_g[i] * q_g[i];
            k_norm += k_g[i] * k_g[i];
        }
        q_norm = 1.0f / sqrtf(q_norm + 1e-6f);
        k_norm = 1.0f / sqrtf(k_norm + 1e-6f);
        for (uint32_t i = 0; i < sd; i++) {
            q_g[i] *= q_norm;
            k_g[i] *= k_norm;
        }
    }

    // Step 4: Compute decay and beta
    mat_vec_mul(m->buf_ssm_dt, l->ssm_alpha, l->ssm_alpha_type, h, nh, hp->n_embd);
    mat_vec_mul(m->buf_ssm_beta, l->ssm_beta, l->ssm_beta_type, h, nh, hp->n_embd);

    for (uint32_t i = 0; i < nh; i++) {
        if (l->ssm_dt_bias)
            m->buf_ssm_dt[i] += l->ssm_dt_bias[i];
        // softplus
        m->buf_ssm_dt[i] = logf(1.0f + expf(m->buf_ssm_dt[i]));
        // Decay = exp(ssm_a * softplus(alpha + bias))
        // ssm_a already stores -exp(A_log) from the GGUF converter
        float A = l->ssm_a ? l->ssm_a[i] : -1.0f;
        m->buf_ssm_dt[i] = expf(A * m->buf_ssm_dt[i]);

        // beta = sigmoid
        m->buf_ssm_beta[i] = 1.0f / (1.0f + expf(-m->buf_ssm_beta[i]));
    }

    // Step 5: Delta rule state update and output
    // State: [nh, sd, vd] = [64, 128, 128]
    // GGUF converter reorders V heads from grouped to tiled order:
    //   Original: [G0_v0..v3, G1_v0..v3, ..., G15_v0..v3]
    //   Tiled:    [G0_v0,G1_v0,...,G15_v0, G0_v1,G1_v1,...,G15_v1, ...]
    // All V-head-indexed tensors (V, decay, beta, attn_gate, ssm_out, conv1d)
    // are consistently reordered. In tiled order, head h maps to K group h % ng.

    float *y = m->buf_attn;  // output [inner] = [8192]
    memset(y, 0, inner * sizeof(float));

    // FIX #5: Query scale factor
    float q_scale = 1.0f / sqrtf((float)sd);

    for (uint32_t head = 0; head < nh; head++) {
        uint32_t g = head % ng;     // tiled order: head h → K group h % ng
        float *k_g = K + g * sd;    // key vector [sd] (L2-normalized)
        float *q_g = Q + g * sd;    // query vector [sd] (L2-normalized)
        float *v_h = V + head * vd; // value vector [vd]
        float *state = ss->state + head * sd * vd;  // [sd, vd]

        float decay = m->buf_ssm_dt[head];
        float beta  = m->buf_ssm_beta[head];

        // FIX #1: Gated Delta Rule state update
        // Step a: Apply decay: S = decay * S
        for (uint32_t si = 0; si < sd; si++) {
            for (uint32_t vi = 0; vi < vd; vi++) {
                state[si * vd + vi] *= decay;
            }
        }

        // Step b: Compute k^T @ S (what state currently predicts for this key)
        // kv_mem[vi] = sum_si k[si] * S[si][vi]
        float kv_mem[128];  // vd = 128
        for (uint32_t vi = 0; vi < vd; vi++) {
            float sum = 0.0f;
            for (uint32_t si = 0; si < sd; si++) {
                sum += k_g[si] * state[si * vd + vi];
            }
            kv_mem[vi] = sum;
        }

        // Step c: Delta = beta * (v - kv_mem)
        // Step d: S += outer(k, delta)
        for (uint32_t si = 0; si < sd; si++) {
            float ki = k_g[si];
            for (uint32_t vi = 0; vi < vd; vi++) {
                float delta = beta * (v_h[vi] - kv_mem[vi]);
                state[si * vd + vi] += ki * delta;
            }
        }

        // Output: y_h = S^T @ (q * scale)
        float *y_h = y + head * vd;
        for (uint32_t vi = 0; vi < vd; vi++) {
            float sum = 0.0f;
            for (uint32_t si = 0; si < sd; si++) {
                sum += state[si * vd + vi] * q_g[si];
            }
            y_h[vi] = sum * q_scale;
        }
    }

    // Step 6: Per-head RMS norm on output
    if (l->ssm_norm) {
        for (uint32_t head = 0; head < nh; head++) {
            float *y_h = y + head * vd;
            rms_norm(y_h, y_h, l->ssm_norm, vd, hp->rms_norm_eps);
        }
    }

    // FIX #6: Output gate uses SiLU (not sigmoid)
    if (l->attn_gate) {
        mat_vec_mul(m->buf_attn_gate, l->attn_gate, l->attn_gate_type,
                    h, inner, hp->n_embd);
        for (uint32_t i = 0; i < inner; i++) {
            float z = m->buf_attn_gate[i];
            float gate = z / (1.0f + expf(-z));  // SiLU(z) = z * sigmoid(z)
            y[i] *= gate;
        }
    }

    // Step 8: Output projection [inner] -> [n_embd]
    if (l->ssm_out) {
        mat_vec_mul(m->buf_ffn, l->ssm_out, l->ssm_out_type, y, hp->n_embd, inner);
    }
}

// ---- Shared Expert FFN ----
static void shared_expert_ffn(model_t *m, const float *h, layer_weights_t *l) {
    const model_hparams_t *hp = &m->hparams;
    uint32_t inter = hp->shared_expert_intermediate;

    mat_vec_mul(m->buf_gate, l->shared_gate, l->shared_gate_type, h, inter, hp->n_embd);
    silu(m->buf_gate, inter);

    mat_vec_mul(m->buf_up, l->shared_up, l->shared_up_type, h, inter, hp->n_embd);

    vec_mul(m->buf_gate, m->buf_gate, m->buf_up, inter);

    mat_vec_mul(m->buf_down, l->shared_down, l->shared_down_type,
                m->buf_gate, hp->n_embd, inter);
}

// Debug NaN detection
#ifdef DEBUG_NAN
static int has_nan(const float *buf, int n) {
    for (int i = 0; i < n; i++) {
        if (isnan(buf[i]) || isinf(buf[i])) return 1;
    }
    return 0;
}
#define CHECK_NAN(buf, n, label) do { \
    if (has_nan(buf, n)) fprintf(stderr, "NaN at layer %u: %s\n", il, label); \
} while(0)
#else
#define CHECK_NAN(buf, n, label) ((void)0)
#endif

// ---- Forward pass for one token ----
float *forward(model_t *model, int token_id, int pos) {
    const model_hparams_t *hp = &model->hparams;

    float *x = model->buf_x;

    // ---- Embedding lookup (always on CPU, needed before GPU path too) ----
    if (model->token_embd_type == GGML_TYPE_Q4_K) {
        size_t row_blocks = hp->n_embd / QK_K;
        const block_q4_K *embd = (const block_q4_K *)model->token_embd;
        dequantize_row_q4_K(embd + (size_t)token_id * row_blocks, x, hp->n_embd);
    } else if (model->token_embd_type == GGML_TYPE_Q6_K) {
        size_t row_blocks = hp->n_embd / QK_K;
        const block_q6_K *embd = (const block_q6_K *)model->token_embd;
        dequantize_row_q6_K(embd + (size_t)token_id * row_blocks, x, hp->n_embd);
    } else if (model->token_embd_type == GGML_TYPE_Q8_0) {
        size_t row_blocks = hp->n_embd / QK8_0;
        const block_q8_0 *embd = (const block_q8_0 *)model->token_embd;
        dequantize_row_q8_0(embd + (size_t)token_id * row_blocks, x, hp->n_embd);
    } else if (model->token_embd_type == GGML_TYPE_F32) {
        memcpy(x, (float *)model->token_embd + (size_t)token_id * hp->n_embd,
               hp->n_embd * sizeof(float));
    } else if (model->token_embd_type == GGML_TYPE_F16) {
        const uint16_t *embd = (const uint16_t *)model->token_embd;
        for (uint32_t i = 0; i < hp->n_embd; i++)
            x[i] = fp16_to_fp32(embd[(size_t)token_id * hp->n_embd + i]);
    } else if (model->token_embd_type == GGML_TYPE_BF16) {
        const uint16_t *embd = (const uint16_t *)model->token_embd;
        for (uint32_t i = 0; i < hp->n_embd; i++) {
            uint32_t bits = (uint32_t)embd[(size_t)token_id * hp->n_embd + i] << 16;
            memcpy(&x[i], &bits, sizeof(float));
        }
    }

#ifdef QMOE_GPU
    if (model->gpu_ctx) {
        float *logits = gpu_forward(model->gpu_ctx, model, token_id, pos);
        if (logits) return logits;
        fprintf(stderr, "forward: GPU failed, CPU fallback\n");
        gpu_free(model->gpu_ctx);
        model->gpu_ctx = NULL;
    }
#endif

    if (g_debug < 0) {
        const char *denv = getenv("QMOE_DEBUG");
        g_debug = denv ? atoi(denv) : 0;
    }

    if (g_debug > 0) dbg_stats("embedding", x, hp->n_embd, -1);

    // Debug: limit number of layers
    uint32_t max_layers = hp->n_layers;
    const char *layer_limit = getenv("QMOE_MAX_LAYERS");
    if (layer_limit) {
        max_layers = (uint32_t)atoi(layer_limit);
        if (max_layers > hp->n_layers) max_layers = hp->n_layers;
        fprintf(stderr, "DEBUG: running only %u/%u layers\n", max_layers, hp->n_layers);
    }

    // ---- Layer loop ----
    for (uint32_t il = 0; il < max_layers; il++) {
        layer_weights_t *l = &model->layers[il];
        float *h = model->buf_h;

        // --- Pre-attention norm ---
        CHECK_NAN(x, hp->n_embd, "x input");
        rms_norm(h, x, l->attn_norm, hp->n_embd, hp->rms_norm_eps);
        CHECK_NAN(h, hp->n_embd, "after attn_norm");

        if (l->attn_type == LAYER_ATTN_FULL) {
            // ---- Gated full attention ----
            gated_gqa_attention(model, il, pos);
        } else if (getenv("QMOE_SKIP_SSM")) {
            // Debug: bypass SSM, output zero
            memset(model->buf_ffn, 0, hp->n_embd * sizeof(float));
        } else {
            // ---- DeltaNet SSM ----
            deltanet_ssm(model, il);
        }

        CHECK_NAN(model->buf_ffn, hp->n_embd, "attn_output");
        if (g_debug > 0) dbg_stats(l->attn_type == LAYER_ATTN_FULL ? "attn_out" : "ssm_out",
                                     model->buf_ffn, hp->n_embd, il);

        // Residual after attention
        vec_add(x, x, model->buf_ffn, hp->n_embd);
        CHECK_NAN(x, hp->n_embd, "x after attn residual");

        // --- Post-attention / pre-FFN norm ---
        rms_norm(h, x, l->ffn_norm, hp->n_embd, hp->rms_norm_eps);
        CHECK_NAN(h, hp->n_embd, "after ffn_norm");

        // --- MoE FFN ---
        // Router
        mat_vec_mul(model->buf_router, l->router, GGML_TYPE_F32, h,
                    hp->n_experts, hp->n_embd);
        softmax(model->buf_router, hp->n_experts);

        // Top-k expert selection
        int expert_ids[16];
        float expert_scores[16];
        top_k(model->buf_router, hp->n_experts, hp->n_experts_used,
              expert_ids, expert_scores);

        // Renormalize
        float score_sum = 0.0f;
        for (uint32_t i = 0; i < hp->n_experts_used; i++) score_sum += expert_scores[i];
        if (score_sum > 0.0f) {
            float inv = 1.0f / score_sum;
            for (uint32_t i = 0; i < hp->n_experts_used; i++) expert_scores[i] *= inv;
        }

        // Routing prediction experiment: compare previous prediction with actual
        if (g_pred_enabled < 0) g_pred_enabled = getenv("QMOE_PREDICT_ROUTING") != NULL;
        if (g_pred_enabled) {
            if (il == 0) g_pred_n = 0;  // reset at start of each token

            if (g_pred_n > 0) {
                int skip_ov = 0, same_ov = 0;
                for (uint32_t i = 0; i < hp->n_experts_used; i++) {
                    for (int j = 0; j < g_pred_n; j++)
                        if (expert_ids[i] == g_pred_skip_ids[j]) { skip_ov++; break; }
                    for (int j = 0; j < g_pred_n; j++)
                        if (expert_ids[i] == g_pred_same_ids[j]) { same_ov++; break; }
                }
                g_pred_total_skip += skip_ov;
                g_pred_total_same += same_ov;
                g_pred_total_n += hp->n_experts_used;
                fprintf(stderr, "  PREDICT L%02d: skip-attn=%d/%d  same-expert=%d/%d\n",
                        il, skip_ov, hp->n_experts_used, same_ov, hp->n_experts_used);
            }
            memcpy(g_pred_same_ids, expert_ids, hp->n_experts_used * sizeof(int));
        }

        // Load experts from NVMe
        void *expert_buffers[16];
        if (model->nvme_io) {
            if (nvme_io_load_experts(model->nvme_io, il, expert_ids,
                                     hp->n_experts_used, expert_buffers) != 0) {
                fprintf(stderr, "forward: expert load failed layer %u\n", il);
                return NULL;
            }
        } else {
            fprintf(stderr, "forward: no NVMe I/O configured\n");
            return NULL;
        }

        // Compute routed expert FFN
        memset(model->buf_ffn, 0, hp->n_embd * sizeof(float));

#ifdef QMOE_GPU
        if (model->gpu_ctx) {
            int rc = gpu_expert_ffn(model->gpu_ctx, h, expert_buffers,
                                    expert_scores, hp->n_experts_used,
                                    hp->n_embd, hp->expert_intermediate,
                                    hp->expert_gate_size, hp->expert_up_size,
                                    hp->expert_stride,
                                    hp->gate_type, hp->up_type, l->down_type,
                                    model->buf_ffn);
            if (rc == 0) goto moe_done;
        }
#endif

        for (uint32_t ei = 0; ei < hp->n_experts_used; ei++) {
            uint8_t *expert_data = (uint8_t *)expert_buffers[ei];
            void *gate_w = expert_data;
            void *up_w   = expert_data + hp->expert_gate_size;
            void *down_w = expert_data + hp->expert_gate_size + hp->expert_up_size;

            mat_vec_mul(model->buf_gate, gate_w, hp->gate_type, h,
                        hp->expert_intermediate, hp->n_embd);
            silu(model->buf_gate, hp->expert_intermediate);

            mat_vec_mul(model->buf_up, up_w, hp->up_type, h,
                        hp->expert_intermediate, hp->n_embd);

            vec_mul(model->buf_gate, model->buf_gate, model->buf_up,
                    hp->expert_intermediate);

            mat_vec_mul(model->buf_down, down_w, l->down_type,
                        model->buf_gate, hp->n_embd, hp->expert_intermediate);

            vec_scaled_add(model->buf_ffn, model->buf_down, expert_scores[ei], hp->n_embd);
        }

#ifdef QMOE_GPU
        moe_done:
#endif

        // Shared expert (always active)
        if (hp->has_shared_expert && l->shared_gate && l->shared_up && l->shared_down) {
            shared_expert_ffn(model, h, l);

            // Apply shared expert gate: scalar = sigmoid(dot(h, gate_weight))
            if (l->shared_expert_gate) {
                float dot = 0.0f;
                for (uint32_t i = 0; i < hp->n_embd; i++)
                    dot += h[i] * l->shared_expert_gate[i];
                float gate_scalar = 1.0f / (1.0f + expf(-dot));
                for (uint32_t i = 0; i < hp->n_embd; i++)
                    model->buf_down[i] *= gate_scalar;
            }

            vec_add(model->buf_ffn, model->buf_ffn, model->buf_down, hp->n_embd);
        }

        CHECK_NAN(model->buf_ffn, hp->n_embd, "MoE ffn_out");
        if (g_debug > 0) dbg_stats("moe_out", model->buf_ffn, hp->n_embd, il);

        // Debug: skip MoE output
        if (getenv("QMOE_SKIP_MOE"))
            memset(model->buf_ffn, 0, hp->n_embd * sizeof(float));

        // Residual after FFN
        vec_add(x, x, model->buf_ffn, hp->n_embd);
        CHECK_NAN(x, hp->n_embd, "x after MoE residual");
        if (g_debug > 0) dbg_stats("x_residual", x, hp->n_embd, il);

        // Routing prediction: predict next layer's routing by skipping attention
        if (g_pred_enabled > 0 && il + 1 < max_layers) {
            layer_weights_t *next_l = &model->layers[il + 1];
            rms_norm(h, x, next_l->ffn_norm, hp->n_embd, hp->rms_norm_eps);
            mat_vec_mul(model->buf_router, next_l->router, GGML_TYPE_F32, h,
                        hp->n_experts, hp->n_embd);
            softmax(model->buf_router, hp->n_experts);
            float pred_scores[16];
            top_k(model->buf_router, hp->n_experts, hp->n_experts_used,
                  g_pred_skip_ids, pred_scores);
            g_pred_n = hp->n_experts_used;
        }
    }

    // ---- Output ----
    rms_norm(model->buf_h, x, model->output_norm, hp->n_embd, hp->rms_norm_eps);
    if (g_debug > 0) dbg_stats("final_norm", model->buf_h, hp->n_embd, 99);

    // Dump hidden state for comparison
    if (getenv("QMOE_DUMP_HIDDEN")) {
        static int dump_count = 0;
        char fname[256];
        snprintf(fname, sizeof(fname), "/tmp/our_embd_%d.bin", dump_count);
        FILE *f = fopen(fname, "wb");
        if (f) { fwrite(model->buf_h, sizeof(float), hp->n_embd, f); fclose(f); }
        fprintf(stderr, "  Dumped %u hidden floats to %s\n", hp->n_embd, fname);
        fprintf(stderr, "  embd[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                model->buf_h[0], model->buf_h[1], model->buf_h[2], model->buf_h[3],
                model->buf_h[4], model->buf_h[5], model->buf_h[6], model->buf_h[7]);
        dump_count++;
    }

    mat_vec_mul(model->buf_logits, model->output, model->output_type,
                model->buf_h, hp->vocab_size, hp->n_embd);
    if (g_debug > 0) {
        dbg_stats("logits", model->buf_logits, hp->vocab_size, 99);
        fprintf(stderr, "  logits[0..19]:");
        for (int i = 0; i < 20 && i < (int)hp->vocab_size; i++)
            fprintf(stderr, " %.4f", model->buf_logits[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  logit[11751]('Paris')=%.4f logit[264]('a')=%.4f logit[13]('.')=%.4f\n",
                model->buf_logits[11751], model->buf_logits[264], model->buf_logits[13]);
    }

    // Dump logits for comparison
    if (getenv("QMOE_DUMP_HIDDEN")) {
        static int logit_dump_count = 0;
        char fname[256];
        snprintf(fname, sizeof(fname), "/tmp/our_logits_%d.bin", logit_dump_count);
        FILE *f = fopen(fname, "wb");
        if (f) { fwrite(model->buf_logits, sizeof(float), hp->vocab_size, f); fclose(f); }
        logit_dump_count++;
    }

    // Routing prediction summary
    if (g_pred_enabled > 0 && g_pred_total_n > 0) {
        fprintf(stderr, "PREDICT: skip-attn=%d/%d (%.1f%%)  same-expert=%d/%d (%.1f%%)\n",
                g_pred_total_skip, g_pred_total_n,
                100.0 * g_pred_total_skip / g_pred_total_n,
                g_pred_total_same, g_pred_total_n,
                100.0 * g_pred_total_same / g_pred_total_n);
    }

    return model->buf_logits;
}
