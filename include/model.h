#ifndef QMOE_MODEL_H
#define QMOE_MODEL_H

#include "gguf.h"
#include "tensor.h"
#include "nvme_io.h"
#include <stdint.h>

#ifdef QMOE_GPU
struct gpu_ctx;
#endif

// Layer attention types for hybrid architecture
enum layer_attn_type {
    LAYER_ATTN_FULL   = 0,   // standard GQA (every 4th layer in Qwen3.5)
    LAYER_ATTN_LINEAR = 1,   // DeltaNet SSM linear attention
};

// Qwen3.5MoE model hyperparameters
typedef struct {
    uint32_t n_layers;
    uint32_t n_embd;              // hidden dimension (3072)
    uint32_t n_heads;             // query attention heads (32)
    uint32_t n_kv_heads;          // key/value heads for GQA (2)
    uint32_t head_dim;            // per-head dimension (256)
    uint32_t n_experts;           // total experts per MoE layer (256)
    uint32_t n_experts_used;      // active routed experts per token (8)
    uint32_t expert_intermediate; // MoE FFN intermediate size (1024)
    uint32_t shared_expert_intermediate; // shared expert intermediate size (1024)
    uint32_t ffn_intermediate;    // dense FFN intermediate size
    uint32_t vocab_size;
    uint32_t ctx_len;
    float    rope_freq_base;
    float    rms_norm_eps;
    float    partial_rotary_factor;
    uint32_t rope_dim;            // head_dim * partial_rotary_factor (64)

    // Hybrid attention config
    uint32_t full_attn_interval;  // every N-th layer is full attention (4)
    bool     has_shared_expert;

    // SSM/DeltaNet config (from qwen35moe.ssm.* keys)
    uint32_t ssm_inner_size;      // total value dim (8192)
    uint32_t ssm_state_size;      // key/state dim per group (128)
    uint32_t ssm_group_count;     // number of key groups (16)
    uint32_t ssm_time_step_rank;  // number of SSM heads (64)
    uint32_t ssm_conv_kernel;     // conv1d kernel size (4)

    // Derived SSM dimensions
    uint32_t ssm_value_dim;       // inner_size / time_step_rank (128)
    uint32_t ssm_qkv_dim;        // inner_size + 2 * group_count * state_size (12288)
    uint32_t full_q_dim;          // 2 * n_heads * head_dim (16384, includes gate)

    // Expert weight types
    enum ggml_dtype gate_type;
    enum ggml_dtype up_type;

    // Expert store sizes
    uint64_t expert_gate_size;
    uint64_t expert_up_size;
    uint64_t expert_stride;
} model_hparams_t;

// Per-layer weights (non-expert, kept in RAM)
typedef struct {
    enum layer_attn_type attn_type;

    // ---- Full attention weights (LAYER_ATTN_FULL only) ----
    void *wq;           // [n_embd, 2*n_heads*head_dim] (Q + output gate fused)
    void *wk;           // [n_embd, n_kv_heads*head_dim]
    void *wv;           // [n_embd, n_kv_heads*head_dim]
    void *wo;           // [n_heads*head_dim, n_embd]
    enum ggml_dtype wq_type, wk_type, wv_type, wo_type;

    float *q_norm;      // [head_dim]
    float *k_norm;      // [head_dim]

    // ---- DeltaNet/SSM weights (LAYER_ATTN_LINEAR only) ----
    void *attn_qkv;             // [n_embd, ssm_qkv_dim] fused x+B+C projection
    enum ggml_dtype attn_qkv_type;

    void *attn_gate;            // [n_embd, ssm_inner_size] output gate matrix
    enum ggml_dtype attn_gate_type;

    float *ssm_a;               // [time_step_rank] base decay (f32)
    void  *ssm_alpha;           // [n_embd, time_step_rank] decay projection
    enum ggml_dtype ssm_alpha_type;
    void  *ssm_beta;            // [n_embd, time_step_rank] update gate projection
    enum ggml_dtype ssm_beta_type;
    float *ssm_conv1d;          // [conv_kernel, ssm_qkv_dim] (f32)
    float *ssm_dt_bias;         // [time_step_rank] (f32)
    float *ssm_norm;            // [ssm_state_size] per-head RMS norm (f32)
    void  *ssm_out;             // [ssm_inner_size, n_embd] output projection
    enum ggml_dtype ssm_out_type;

    // ---- Common weights ----
    float *attn_norm;   // [n_embd] pre-attention RMS norm
    float *ffn_norm;    // [n_embd] post-attention / pre-FFN RMS norm

    // MoE router
    float *router;      // [n_embd, n_experts] F32

    // Shared expert
    float *shared_expert_gate; // [n_embd] sigmoid gate scalar (ffn_gate_inp_shexp)
    void *shared_gate;  // [n_embd, shared_expert_intermediate]
    void *shared_up;    // [n_embd, shared_expert_intermediate]
    void *shared_down;  // [shared_expert_intermediate, n_embd]
    enum ggml_dtype shared_gate_type, shared_up_type, shared_down_type;

    // Per-layer expert down projection info
    enum ggml_dtype down_type;
    uint64_t expert_down_size;
} layer_weights_t;

// KV cache for one layer (full attention layers only)
typedef struct {
    float *k;  // [ctx_len, n_kv_heads * head_dim]
    float *v;  // [ctx_len, n_kv_heads * head_dim]
} kv_cache_t;

// SSM recurrent state for one layer (DeltaNet layers only)
typedef struct {
    // State: [time_step_rank, state_size, value_dim] = [64, 128, 128]
    float *state;
    // Conv buffer: last (conv_kernel-1) steps of qkv output [conv_kernel-1, ssm_qkv_dim]
    float *conv_buf;
    int    conv_pos;  // circular buffer position
} ssm_state_t;

// Full model context
typedef struct {
    model_hparams_t  hparams;
    gguf_ctx_t      *gguf;

    // Global weights
    void           *token_embd;
    enum ggml_dtype  token_embd_type;
    size_t           token_embd_size;
    float           *output_norm;
    void            *output;
    enum ggml_dtype  output_type;
    size_t           output_size;

    // Per-layer weights
    layer_weights_t *layers;

    // KV cache (full attention layers only, NULL entries for SSM layers)
    kv_cache_t      *kv_cache;
    int              kv_pos;

    // SSM state (DeltaNet layers only, NULL entries for full attn layers)
    ssm_state_t     *ssm_state;

    // NVMe expert I/O
    nvme_io_t       *nvme_io;

#ifdef QMOE_GPU
    struct gpu_ctx  *gpu_ctx;
#endif

    // Scratch buffers for inference
    float *buf_x;         // [n_embd]
    float *buf_h;         // [n_embd]
    float *buf_qkv;       // [max(ssm_qkv_dim, full_q_dim)]
    float *buf_k;         // [n_kv_heads * head_dim] (full attn only)
    float *buf_v;         // [n_kv_heads * head_dim] (full attn only)
    float *buf_attn;      // [ssm_inner_size] = [n_heads * head_dim] = [8192]
    float *buf_attn_gate; // [ssm_inner_size] = [8192]
    float *buf_ssm_dt;    // [time_step_rank] = [64]
    float *buf_ssm_beta;  // [time_step_rank] = [64]
    float *buf_ffn;       // [n_embd]
    float *buf_router;    // [n_experts]
    float *buf_gate;      // [max(expert_intermediate, shared_expert_intermediate)]
    float *buf_up;        // [max(expert_intermediate, shared_expert_intermediate)]
    float *buf_down;      // [n_embd]
    float *buf_logits;    // [vocab_size]
} model_t;

// Load model from GGUF + expert stores
model_t *model_load(const char *gguf_path, const char **store_paths, int n_stores);

// Free model
void model_free(model_t *model);

// Reset KV cache and SSM state (for chunked perplexity evaluation)
void model_reset_state(model_t *model);

#endif // QMOE_MODEL_H
