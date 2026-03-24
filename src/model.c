#include "model.h"
#include "quant.h"
#ifdef QMOE_GPU
#include "gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper: allocate and read a tensor from GGUF (raw quantized data)
static void *read_tensor_data(gguf_ctx_t *ctx, const char *name, size_t *out_size,
                               enum ggml_dtype *out_type) {
    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) {
        fprintf(stderr, "model: tensor not found: %s\n", name);
        return NULL;
    }
    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    void *data = malloc(ti->size);
    if (!data) {
        fprintf(stderr, "model: OOM for tensor %s (%zu bytes)\n", name, ti->size);
        return NULL;
    }
    if (gguf_read_tensor(ctx, tid, data, ti->size) == 0) { free(data); return NULL; }

    if (ti->type == GGML_TYPE_F16 || ti->type == GGML_TYPE_BF16) {
        int64_t nelements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) nelements *= ti->dims[d];

        float *f32 = malloc((size_t)nelements * sizeof(float));
        if (!f32) {
            free(data);
            fprintf(stderr, "model: OOM converting tensor %s to F32\n", name);
            return NULL;
        }

        const uint16_t *src = (const uint16_t *)data;
        if (ti->type == GGML_TYPE_F16) {
            for (int64_t i = 0; i < nelements; i++) f32[i] = fp16_to_fp32(src[i]);
        } else {
            for (int64_t i = 0; i < nelements; i++) {
                uint32_t bits = (uint32_t)src[i] << 16;
                memcpy(&f32[i], &bits, sizeof(float));
            }
        }

        free(data);
        if (out_size) *out_size = (size_t)nelements * sizeof(float);
        if (out_type) *out_type = GGML_TYPE_F32;
        return f32;
    }

    if (out_size) *out_size = ti->size;
    if (out_type) *out_type = ti->type;
    return data;
}

// Try to read tensor, NULL silently if not found
static void *try_read_tensor_data(gguf_ctx_t *ctx, const char *name, size_t *out_size,
                                   enum ggml_dtype *out_type) {
    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) return NULL;
    return read_tensor_data(ctx, name, out_size, out_type);
}

// Read tensor and convert to F32
static float *read_tensor_f32(gguf_ctx_t *ctx, const char *name) {
    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) {
        fprintf(stderr, "model: tensor not found: %s\n", name);
        return NULL;
    }
    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    void *raw = malloc(ti->size);
    if (!raw) return NULL;
    if (gguf_read_tensor(ctx, tid, raw, ti->size) == 0) { free(raw); return NULL; }

    int64_t nelements = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) nelements *= ti->dims[d];

    if (ti->type == GGML_TYPE_F32) return (float *)raw;

    float *f32 = malloc(nelements * sizeof(float));
    if (!f32) { free(raw); return NULL; }

    if (ti->type == GGML_TYPE_Q4_K) {
        dequantize_row_q4_K((const block_q4_K *)raw, f32, nelements);
    } else if (ti->type == GGML_TYPE_Q6_K) {
        dequantize_row_q6_K((const block_q6_K *)raw, f32, nelements);
    } else if (ti->type == GGML_TYPE_Q5_K) {
        dequantize_row_q5_K((const block_q5_K *)raw, f32, nelements);
    } else if (ti->type == GGML_TYPE_Q8_0) {
        dequantize_row_q8_0((const block_q8_0 *)raw, f32, nelements);
    } else if (ti->type == GGML_TYPE_MXFP4) {
        dequantize_row_mxfp4((const block_mxfp4 *)raw, f32, nelements);
    } else if (ti->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (int64_t i = 0; i < nelements; i++) f32[i] = fp16_to_fp32(src[i]);
    } else if (ti->type == GGML_TYPE_BF16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (int64_t i = 0; i < nelements; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &bits, sizeof(float));
        }
    } else {
        fprintf(stderr, "model: unsupported type %s (code=%d) for F32 conversion: %s\n",
                ggml_type_name(ti->type), (int)ti->type, name);
        free(f32); free(raw); return NULL;
    }
    free(raw);
    return f32;
}

static float *try_read_tensor_f32(gguf_ctx_t *ctx, const char *name) {
    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) return NULL;
    return read_tensor_f32(ctx, name);
}

static uint32_t get_u32_or(gguf_ctx_t *ctx, const char *key, uint32_t fallback) {
    int64_t ki = gguf_find_key(ctx, key);
    if (ki < 0) return fallback;
    return ctx->kv[ki].value.u32;
}

static float get_f32_or(gguf_ctx_t *ctx, const char *key, float fallback) {
    int64_t ki = gguf_find_key(ctx, key);
    if (ki < 0) return fallback;
    return ctx->kv[ki].value.f32;
}

model_t *model_load(const char *gguf_path, const char **store_paths, int n_stores) {
    fprintf(stderr, "Loading model from %s...\n", gguf_path);

    gguf_ctx_t *ctx = gguf_open(gguf_path);
    if (!ctx) return NULL;

    model_t *m = calloc(1, sizeof(model_t));
    if (!m) { gguf_close(ctx); return NULL; }
    m->gguf = ctx;

    const char *arch = gguf_get_str(ctx, "general.architecture");
    if (!arch) {
        fprintf(stderr, "model: missing general.architecture\n");
        model_free(m); return NULL;
    }

    char key[256];
    #define GET_U32(field, suffix) do { \
        snprintf(key, sizeof(key), "%s." suffix, arch); \
        m->hparams.field = gguf_get_u32(ctx, key); \
    } while(0)
    #define GET_F32(field, suffix) do { \
        snprintf(key, sizeof(key), "%s." suffix, arch); \
        m->hparams.field = gguf_get_f32(ctx, key); \
    } while(0)
    #define GET_U32_OR(field, suffix, fb) do { \
        snprintf(key, sizeof(key), "%s." suffix, arch); \
        m->hparams.field = get_u32_or(ctx, key, fb); \
    } while(0)
    #define GET_F32_OR(field, suffix, fb) do { \
        snprintf(key, sizeof(key), "%s." suffix, arch); \
        m->hparams.field = get_f32_or(ctx, key, fb); \
    } while(0)

    // Core hyperparams
    GET_U32(n_layers,            "block_count");
    GET_U32(n_embd,              "embedding_length");
    GET_U32(n_heads,             "attention.head_count");
    GET_U32(n_kv_heads,          "attention.head_count_kv");
    GET_U32(n_experts,           "expert_count");
    GET_U32(n_experts_used,      "expert_used_count");
    GET_U32(ffn_intermediate,    "feed_forward_length");
    GET_U32(ctx_len,             "context_length");
    // Allow overriding ctx_len (e.g. for PPL eval with small chunks)
    const char *ctx_override = getenv("QMOE_CTX_SIZE");
    if (ctx_override) {
        uint32_t new_ctx = (uint32_t)atoi(ctx_override);
        if (new_ctx > 0 && new_ctx < m->hparams.ctx_len) {
            fprintf(stderr, "model: ctx_len overridden %u -> %u\n",
                    m->hparams.ctx_len, new_ctx);
            m->hparams.ctx_len = new_ctx;
        }
    }
    GET_F32(rope_freq_base,      "rope.freq_base");
    GET_F32(rms_norm_eps,        "attention.layer_norm_rms_epsilon");

    GET_U32_OR(expert_intermediate, "expert_feed_forward_length", m->hparams.ffn_intermediate);
    GET_U32_OR(shared_expert_intermediate, "expert_shared_feed_forward_length",
               m->hparams.expert_intermediate);

    // Head dim
    snprintf(key, sizeof(key), "%s.attention.key_length", arch);
    int64_t ki = gguf_find_key(ctx, key);
    m->hparams.head_dim = (ki >= 0) ? ctx->kv[ki].value.u32
                                     : m->hparams.n_embd / m->hparams.n_heads;

    // Rope dim
    snprintf(key, sizeof(key), "%s.rope.dimension_count", arch);
    ki = gguf_find_key(ctx, key);
    if (ki >= 0) {
        m->hparams.rope_dim = ctx->kv[ki].value.u32;
        m->hparams.partial_rotary_factor = (float)m->hparams.rope_dim / m->hparams.head_dim;
    } else {
        m->hparams.partial_rotary_factor = 1.0f;
        m->hparams.rope_dim = m->hparams.head_dim;
    }

    // Hybrid attention: full_attention_interval
    // Try both "full_attention_interval" and "attention.full_attention_interval"
    snprintf(key, sizeof(key), "%s.full_attention_interval", arch);
    ki = gguf_find_key(ctx, key);
    if (ki >= 0) {
        m->hparams.full_attn_interval = ctx->kv[ki].value.u32;
    } else {
        GET_U32_OR(full_attn_interval, "attention.full_attention_interval", 0);
    }

    // SSM/DeltaNet config
    GET_U32_OR(ssm_inner_size,     "ssm.inner_size",      8192);
    GET_U32_OR(ssm_state_size,     "ssm.state_size",       128);
    GET_U32_OR(ssm_group_count,    "ssm.group_count",       16);
    GET_U32_OR(ssm_time_step_rank, "ssm.time_step_rank",    64);
    GET_U32_OR(ssm_conv_kernel,    "ssm.conv_kernel",        4);

    // Derived SSM dimensions
    m->hparams.ssm_value_dim = m->hparams.ssm_inner_size / m->hparams.ssm_time_step_rank;
    m->hparams.ssm_qkv_dim = m->hparams.ssm_inner_size +
                              2 * m->hparams.ssm_group_count * m->hparams.ssm_state_size;
    m->hparams.full_q_dim = 2 * m->hparams.n_heads * m->hparams.head_dim;

    // Feature detection
    m->hparams.has_shared_expert = (gguf_find_tensor(ctx, "blk.0.ffn_gate_shexp.weight") >= 0);

    // Vocab size from embedding tensor
    int64_t embd_tid = gguf_find_tensor(ctx, "token_embd.weight");
    if (embd_tid >= 0) m->hparams.vocab_size = ctx->tensors[embd_tid].dims[1];

    #undef GET_U32
    #undef GET_F32
    #undef GET_U32_OR
    #undef GET_F32_OR

    const model_hparams_t *hp = &m->hparams;
    fprintf(stderr, "Model: %s\n", arch);
    fprintf(stderr, "  layers=%u, embd=%u, heads=%u/%u, head_dim=%u\n",
            hp->n_layers, hp->n_embd, hp->n_heads, hp->n_kv_heads, hp->head_dim);
    fprintf(stderr, "  experts=%u (used=%u), expert_ffn=%u, vocab=%u\n",
            hp->n_experts, hp->n_experts_used, hp->expert_intermediate, hp->vocab_size);
    fprintf(stderr, "  rope_base=%.0f, rms_eps=%g, rope_dim=%u (%.0f%%)\n",
            hp->rope_freq_base, hp->rms_norm_eps, hp->rope_dim,
            hp->partial_rotary_factor * 100.0f);
    if (hp->full_attn_interval > 0) {
        fprintf(stderr, "  hybrid: full attn every %u layers, DeltaNet/SSM otherwise\n",
                hp->full_attn_interval);
        fprintf(stderr, "  SSM: inner=%u, state=%u, groups=%u, heads=%u, conv=%u\n",
                hp->ssm_inner_size, hp->ssm_state_size, hp->ssm_group_count,
                hp->ssm_time_step_rank, hp->ssm_conv_kernel);
        fprintf(stderr, "  SSM derived: value_dim=%u, qkv_dim=%u\n",
                hp->ssm_value_dim, hp->ssm_qkv_dim);
    }
    if (hp->has_shared_expert)
        fprintf(stderr, "  shared expert: intermediate=%u\n", hp->shared_expert_intermediate);

    // ---- Load global weights ----
    fprintf(stderr, "Loading global weights...\n");
    m->token_embd = read_tensor_data(ctx, "token_embd.weight", &m->token_embd_size, &m->token_embd_type);
    m->output_norm = read_tensor_f32(ctx, "output_norm.weight");
    m->output = read_tensor_data(ctx, "output.weight", &m->output_size, &m->output_type);
    if (!m->token_embd || !m->output_norm || !m->output) {
        fprintf(stderr, "model: failed to load global weights\n");
        model_free(m); return NULL;
    }

    // ---- Load per-layer weights ----
    fprintf(stderr, "Loading layer weights...\n");
    m->layers = calloc(hp->n_layers, sizeof(layer_weights_t));
    if (!m->layers) { model_free(m); return NULL; }

    int n_full = 0, n_linear = 0;

    for (uint32_t i = 0; i < hp->n_layers; i++) {
        layer_weights_t *l = &m->layers[i];
        char name[128];

        // Determine attention type
        if (hp->full_attn_interval > 0 && ((i + 1) % hp->full_attn_interval == 0)) {
            l->attn_type = LAYER_ATTN_FULL;
            n_full++;
        } else if (hp->full_attn_interval > 0) {
            l->attn_type = LAYER_ATTN_LINEAR;
            n_linear++;
        } else {
            l->attn_type = LAYER_ATTN_FULL;
            n_full++;
        }

        // Common norms
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        l->attn_norm = read_tensor_f32(ctx, name);

        // post_attention_norm (GGUF name) = our ffn_norm
        snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
        l->ffn_norm = try_read_tensor_f32(ctx, name);
        if (!l->ffn_norm) {
            // Fallback to old name
            snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);
            l->ffn_norm = read_tensor_f32(ctx, name);
        }

        if (!l->attn_norm || !l->ffn_norm) {
            fprintf(stderr, "model: failed loading norms for layer %u\n", i);
            model_free(m); return NULL;
        }

        if (l->attn_type == LAYER_ATTN_FULL) {
            // ---- Full attention: Q (with gate), K, V, O ----
            snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
            l->wq = read_tensor_data(ctx, name, NULL, &l->wq_type);

            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            l->wk = read_tensor_data(ctx, name, NULL, &l->wk_type);

            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            l->wv = read_tensor_data(ctx, name, NULL, &l->wv_type);

            snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
            l->wo = read_tensor_data(ctx, name, NULL, &l->wo_type);

            snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
            l->q_norm = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
            l->k_norm = try_read_tensor_f32(ctx, name);

            if (!l->wq || !l->wk || !l->wv || !l->wo) {
                fprintf(stderr, "model: failed loading full attn for layer %u\n", i);
                model_free(m); return NULL;
            }
        } else {
            // ---- DeltaNet/SSM: fused QKV, gate, SSM weights ----
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", i);
            l->attn_qkv = read_tensor_data(ctx, name, NULL, &l->attn_qkv_type);

            snprintf(name, sizeof(name), "blk.%u.attn_gate.weight", i);
            l->attn_gate = try_read_tensor_data(ctx, name, NULL, &l->attn_gate_type);

            snprintf(name, sizeof(name), "blk.%u.ssm_a", i);
            l->ssm_a = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.ssm_alpha.weight", i);
            l->ssm_alpha = try_read_tensor_data(ctx, name, NULL, &l->ssm_alpha_type);

            snprintf(name, sizeof(name), "blk.%u.ssm_beta.weight", i);
            l->ssm_beta = try_read_tensor_data(ctx, name, NULL, &l->ssm_beta_type);

            snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", i);
            l->ssm_conv1d = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", i);
            l->ssm_dt_bias = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", i);
            l->ssm_norm = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.ssm_out.weight", i);
            l->ssm_out = try_read_tensor_data(ctx, name, NULL, &l->ssm_out_type);

            if (!l->attn_qkv) {
                fprintf(stderr, "model: failed loading SSM attn_qkv for layer %u\n", i);
                model_free(m); return NULL;
            }
        }

        // MoE router
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", i);
        l->router = read_tensor_f32(ctx, name);
        if (!l->router) {
            fprintf(stderr, "model: failed loading router for layer %u\n", i);
            model_free(m); return NULL;
        }

        // Shared expert weights
        if (hp->has_shared_expert) {
            // Shared expert gate scalar (sigmoid activation)
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp_shexp.weight", i);
            l->shared_expert_gate = try_read_tensor_f32(ctx, name);

            snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", i);
            l->shared_gate = try_read_tensor_data(ctx, name, NULL, &l->shared_gate_type);

            snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", i);
            l->shared_up = try_read_tensor_data(ctx, name, NULL, &l->shared_up_type);

            snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", i);
            l->shared_down = try_read_tensor_data(ctx, name, NULL, &l->shared_down_type);
        }

        if ((i + 1) % 8 == 0 || i == hp->n_layers - 1) {
            fprintf(stderr, "  Loaded layer %u/%u (%s)\n", i + 1, hp->n_layers,
                    l->attn_type == LAYER_ATTN_FULL ? "full" : "ssm");
        }
    }

    fprintf(stderr, "  Layer types: %d full attention, %d SSM/DeltaNet\n", n_full, n_linear);

    // ---- NVMe expert I/O ----
    if (n_stores > 0 && store_paths) {
        m->nvme_io = nvme_io_init(store_paths, n_stores);
        if (!m->nvme_io) {
            fprintf(stderr, "model: failed to initialize NVMe I/O\n");
            model_free(m); return NULL;
        }

        char name[128];
        snprintf(name, sizeof(name), "blk.0.ffn_gate_exps.weight");
        int64_t tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) m->hparams.gate_type = ctx->tensors[tid].type;

        snprintf(name, sizeof(name), "blk.0.ffn_up_exps.weight");
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) m->hparams.up_type = ctx->tensors[tid].type;

        expert_store_t *es = m->nvme_io->drives[0].store;
        m->hparams.expert_gate_size = es->header.expert_gate_size;
        m->hparams.expert_up_size   = es->header.expert_up_size;
        m->hparams.expert_stride    = es->header.expert_stride;

        for (uint32_t i = 0; i < hp->n_layers && i < es->header.n_moe_layers; i++) {
            m->layers[i].down_type = (enum ggml_dtype)es->layer_index[i].down_type;
            m->layers[i].expert_down_size = es->layer_index[i].down_size;
        }
    }

#ifdef QMOE_GPU
    m->gpu_ctx = gpu_init();
    if (m->gpu_ctx && m->nvme_io) {
        gpu_register_buffers(m->gpu_ctx, m->nvme_io->buffers,
                             m->nvme_io->n_buffers, m->nvme_io->buffer_size);
    }
    if (m->gpu_ctx) {
        const char *rc_env = getenv("QMOE_RAM_CACHE_MB");
        if (rc_env) gpu_set_ram_cache_mb(m->gpu_ctx, atoi(rc_env));
        const char *vc_env = getenv("QMOE_VRAM_CACHE_MB");
        if (vc_env) gpu_set_vram_cache_mb(m->gpu_ctx, atoi(vc_env));
        if (gpu_upload_model(m->gpu_ctx, m) != 0)
            fprintf(stderr, "model: warning: GPU upload failed, CPU fallback\n");

        // Seed VRAM cache from frequency profile if specified
        const char *freq_path = getenv("QMOE_FREQ_PROFILE");
        if (freq_path && freq_path[0]) {
            int seeded = gpu_seed_vram_cache(m->gpu_ctx, m, freq_path);
            if (seeded < 0)
                fprintf(stderr, "model: warning: VRAM cache seeding failed\n");
        }
    }
#endif

    // ---- KV cache (full attention layers) ----
    fprintf(stderr, "Allocating KV cache (ctx=%u, full=%d)...\n", hp->ctx_len, n_full);
    m->kv_cache = calloc(hp->n_layers, sizeof(kv_cache_t));
    if (!m->kv_cache) { model_free(m); return NULL; }

    uint32_t kv_dim = hp->n_kv_heads * hp->head_dim;
    for (uint32_t i = 0; i < hp->n_layers; i++) {
        if (m->layers[i].attn_type != LAYER_ATTN_FULL) continue;
        m->kv_cache[i].k = calloc(hp->ctx_len * kv_dim, sizeof(float));
        m->kv_cache[i].v = calloc(hp->ctx_len * kv_dim, sizeof(float));
        if (!m->kv_cache[i].k || !m->kv_cache[i].v) {
            fprintf(stderr, "model: KV cache alloc failed layer %u\n", i);
            model_free(m); return NULL;
        }
    }
    m->kv_pos = 0;

    // ---- SSM state (DeltaNet layers) ----
    if (n_linear > 0) {
        fprintf(stderr, "Allocating SSM state (%d layers, %u heads, state=%ux%u)...\n",
                n_linear, hp->ssm_time_step_rank, hp->ssm_state_size, hp->ssm_value_dim);
        m->ssm_state = calloc(hp->n_layers, sizeof(ssm_state_t));
        if (!m->ssm_state) { model_free(m); return NULL; }

        uint32_t state_size = hp->ssm_time_step_rank * hp->ssm_state_size * hp->ssm_value_dim;
        uint32_t conv_buf_size = (hp->ssm_conv_kernel - 1) * hp->ssm_qkv_dim;
        for (uint32_t i = 0; i < hp->n_layers; i++) {
            if (m->layers[i].attn_type != LAYER_ATTN_LINEAR) continue;
            m->ssm_state[i].state = calloc(state_size, sizeof(float));
            m->ssm_state[i].conv_buf = calloc(conv_buf_size, sizeof(float));
            m->ssm_state[i].conv_pos = 0;
            if (!m->ssm_state[i].state || !m->ssm_state[i].conv_buf) {
                fprintf(stderr, "model: SSM state alloc failed layer %u\n", i);
                model_free(m); return NULL;
            }
        }
    }

    // ---- Scratch buffers ----
    uint32_t max_qkv = hp->ssm_qkv_dim;
    if (hp->full_q_dim > max_qkv) max_qkv = hp->full_q_dim;

    uint32_t max_ffn = hp->expert_intermediate;
    if (hp->shared_expert_intermediate > max_ffn) max_ffn = hp->shared_expert_intermediate;

    m->buf_x         = calloc(hp->n_embd, sizeof(float));
    m->buf_h         = calloc(hp->n_embd, sizeof(float));
    m->buf_qkv       = calloc(max_qkv, sizeof(float));
    m->buf_k         = calloc(hp->n_kv_heads * hp->head_dim, sizeof(float));
    m->buf_v         = calloc(hp->n_kv_heads * hp->head_dim, sizeof(float));
    m->buf_attn      = calloc(hp->ssm_inner_size, sizeof(float));
    m->buf_attn_gate = calloc(hp->ssm_inner_size, sizeof(float));
    m->buf_ssm_dt    = calloc(hp->ssm_time_step_rank > 0 ? hp->ssm_time_step_rank : 1, sizeof(float));
    m->buf_ssm_beta  = calloc(hp->ssm_time_step_rank > 0 ? hp->ssm_time_step_rank : 1, sizeof(float));
    m->buf_ffn       = calloc(hp->n_embd, sizeof(float));
    m->buf_router    = calloc(hp->n_experts, sizeof(float));
    m->buf_gate      = calloc(max_ffn, sizeof(float));
    m->buf_up        = calloc(max_ffn, sizeof(float));
    m->buf_down      = calloc(hp->n_embd, sizeof(float));
    m->buf_logits    = calloc(hp->vocab_size, sizeof(float));

    fprintf(stderr, "Model loaded successfully.\n");
    return m;
}

void model_reset_state(model_t *m) {
    if (!m) return;
    const model_hparams_t *hp = &m->hparams;

#ifdef QMOE_GPU
    if (m->gpu_ctx) {
        gpu_reset_state(m->gpu_ctx);
        return;
    }
#endif

    // Reset KV cache (full attention layers)
    if (m->kv_cache) {
        uint32_t kv_dim = hp->n_kv_heads * hp->head_dim;
        size_t kv_bytes = (size_t)hp->ctx_len * kv_dim * sizeof(float);
        for (uint32_t i = 0; i < hp->n_layers; i++) {
            if (m->kv_cache[i].k) memset(m->kv_cache[i].k, 0, kv_bytes);
            if (m->kv_cache[i].v) memset(m->kv_cache[i].v, 0, kv_bytes);
        }
    }
    m->kv_pos = 0;

    // Reset SSM state (DeltaNet layers)
    if (m->ssm_state) {
        uint32_t state_size = hp->ssm_time_step_rank * hp->ssm_state_size * hp->ssm_value_dim;
        uint32_t conv_buf_size = (hp->ssm_conv_kernel - 1) * hp->ssm_qkv_dim;
        for (uint32_t i = 0; i < hp->n_layers; i++) {
            if (m->ssm_state[i].state)
                memset(m->ssm_state[i].state, 0, state_size * sizeof(float));
            if (m->ssm_state[i].conv_buf)
                memset(m->ssm_state[i].conv_buf, 0, conv_buf_size * sizeof(float));
            m->ssm_state[i].conv_pos = 0;
        }
    }
}

void model_free(model_t *m) {
    if (!m) return;

    free(m->token_embd);
    free(m->output_norm);
    free(m->output);

    if (m->layers) {
        for (uint32_t i = 0; i < m->hparams.n_layers; i++) {
            layer_weights_t *l = &m->layers[i];
            // Full attention
            free(l->wq); free(l->wk); free(l->wv); free(l->wo);
            free(l->q_norm); free(l->k_norm);
            // SSM/DeltaNet
            free(l->attn_qkv); free(l->attn_gate);
            free(l->ssm_a); free(l->ssm_alpha); free(l->ssm_beta);
            free(l->ssm_conv1d); free(l->ssm_dt_bias);
            free(l->ssm_norm); free(l->ssm_out);
            // Common
            free(l->attn_norm); free(l->ffn_norm);
            free(l->router);
            free(l->shared_expert_gate);
            free(l->shared_gate); free(l->shared_up); free(l->shared_down);
        }
        free(m->layers);
    }

    if (m->kv_cache) {
        for (uint32_t i = 0; i < m->hparams.n_layers; i++) {
            free(m->kv_cache[i].k);
            free(m->kv_cache[i].v);
        }
        free(m->kv_cache);
    }

    if (m->ssm_state) {
        for (uint32_t i = 0; i < m->hparams.n_layers; i++) {
            free(m->ssm_state[i].state);
            free(m->ssm_state[i].conv_buf);
        }
        free(m->ssm_state);
    }

#ifdef QMOE_GPU
    if (m->gpu_ctx && m->nvme_io)
        gpu_unregister_buffers(m->gpu_ctx, m->nvme_io->buffers, m->nvme_io->n_buffers);
    gpu_free(m->gpu_ctx);
#endif

    nvme_io_free(m->nvme_io);

    free(m->buf_x); free(m->buf_h);
    free(m->buf_qkv); free(m->buf_k); free(m->buf_v);
    free(m->buf_attn); free(m->buf_attn_gate);
    free(m->buf_ssm_dt); free(m->buf_ssm_beta);
    free(m->buf_ffn); free(m->buf_router);
    free(m->buf_gate); free(m->buf_up); free(m->buf_down);
    free(m->buf_logits);

    gguf_close(m->gguf);
    free(m);
}
