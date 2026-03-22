#include "gpu.h"

// Map FR_CUDA (set by Makefile) to QMOE_CUDA (used in kernel guards)
#ifdef FR_CUDA
#define QMOE_CUDA 1
#endif

// We can't include model.h directly because it transitively includes
// <liburing.h> which uses C++ templates incompatible with extern "C".
// Instead, declare the C functions and types we need.
extern "C" {

// From gguf.h
enum ggml_dtype {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_MXFP4 = 39,
};

// From quant.h
#define QK_K 256
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

#define QK8_0 32
typedef struct {
    uint16_t d;          // scale (FP16)
    int8_t   qs[QK8_0];  // quantized values
} block_q8_0;

#define QK_MXFP4 32
typedef struct {
    uint8_t s;                   // shared exponent (E8M0)
    uint8_t qs[QK_MXFP4 / 2];   // FP4 E2M1 values (16 bytes, 32 nibbles)
} block_mxfp4;

// Model type declarations — must match include/model.h layout EXACTLY.
// We can't #include model.h because it transitively pulls in <liburing.h>
// which conflicts with HIP's extern "C" block.

// Layer attention types (must match model.h enum)
enum layer_attn_type {
    LAYER_ATTN_FULL   = 0,
    LAYER_ATTN_LINEAR = 1,
};

typedef struct {
    uint32_t n_layers;
    uint32_t n_embd;              // 3072
    uint32_t n_heads;             // 32
    uint32_t n_kv_heads;          // 2
    uint32_t head_dim;            // 256
    uint32_t n_experts;           // 256
    uint32_t n_experts_used;      // 8
    uint32_t expert_intermediate; // 1024
    uint32_t shared_expert_intermediate; // 1024
    uint32_t ffn_intermediate;
    uint32_t vocab_size;
    uint32_t ctx_len;
    float    rope_freq_base;
    float    rms_norm_eps;
    float    partial_rotary_factor;
    uint32_t rope_dim;            // 64

    // Hybrid attention config
    uint32_t full_attn_interval;  // 4
    bool     has_shared_expert;

    // SSM/DeltaNet config
    uint32_t ssm_inner_size;      // 8192
    uint32_t ssm_state_size;      // 128
    uint32_t ssm_group_count;     // 16
    uint32_t ssm_time_step_rank;  // 64
    uint32_t ssm_conv_kernel;     // 4

    // Derived SSM dimensions
    uint32_t ssm_value_dim;       // 128
    uint32_t ssm_qkv_dim;        // 12288
    uint32_t full_q_dim;          // 16384

    // Expert weight types
    enum ggml_dtype gate_type;
    enum ggml_dtype up_type;

    // Expert store sizes
    uint64_t expert_gate_size;
    uint64_t expert_up_size;
    uint64_t expert_stride;
} model_hparams_t;

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
    void *attn_qkv;             // [n_embd, ssm_qkv_dim]
    enum ggml_dtype attn_qkv_type;

    void *attn_gate;            // [n_embd, ssm_inner_size]
    enum ggml_dtype attn_gate_type;

    float *ssm_a;               // [time_step_rank] (f32)
    void  *ssm_alpha;           // [n_embd, time_step_rank]
    enum ggml_dtype ssm_alpha_type;
    void  *ssm_beta;            // [n_embd, time_step_rank]
    enum ggml_dtype ssm_beta_type;
    float *ssm_conv1d;          // [conv_kernel, ssm_qkv_dim] (f32)
    float *ssm_dt_bias;         // [time_step_rank] (f32)
    float *ssm_norm;            // [ssm_state_size] (f32)
    void  *ssm_out;             // [ssm_inner_size, n_embd]
    enum ggml_dtype ssm_out_type;

    // ---- Common weights ----
    float *attn_norm;   // [n_embd]
    float *ffn_norm;    // [n_embd]

    // MoE router
    float *router;      // [n_embd, n_experts] F32

    // Shared expert
    float *shared_expert_gate; // [n_embd]
    void *shared_gate;  // [n_embd, shared_expert_intermediate]
    void *shared_up;    // [n_embd, shared_expert_intermediate]
    void *shared_down;  // [shared_expert_intermediate, n_embd]
    enum ggml_dtype shared_gate_type, shared_up_type, shared_down_type;

    // Per-layer expert down projection info
    enum ggml_dtype down_type;
    uint64_t expert_down_size;
} layer_weights_t;

typedef struct { float *k; float *v; } kv_cache_t;

// SSM recurrent state (must match model.h)
typedef struct {
    float *state;
    float *conv_buf;
    int    conv_pos;
} ssm_state_t;

// Opaque nvme_io_t - we only call nvme_io_load_experts through it
typedef struct nvme_io_t_tag nvme_io_t;

// Blocking pread of a single expert into caller-supplied buffer.
// Avoids io_uring entirely — safe to call from any thread without contention.
// Defined in nvme_io.c.
int nvme_io_pread_expert(nvme_io_t *io, int layer, int expert_id, void *buf);
int nvme_io_load_experts(nvme_io_t *io, int layer, const int *expert_ids,
                         int n_experts, void **out_buffers);
int nvme_io_load_experts_at(nvme_io_t *io, int layer, const int *expert_ids,
                             int n_experts, int buffer_offset, void **out_buffers);
typedef void (*nvme_expert_cb)(int buffer_idx, void *buffer, void *user_data);
int nvme_io_load_experts_at_cb(nvme_io_t *io, int layer, const int *expert_ids,
                                int n_experts, int buffer_offset, void **out_buffers,
                                nvme_expert_cb cb, void *cb_data);
int nvme_io_load_experts_prefetch(nvme_io_t *io, int layer, const int *expert_ids,
                                   int n_experts, int buffer_offset, void **out_buffers);
// From tensor.h — CPU-side math for spec routing
void rms_norm(float *out, const float *x, const float *weight, int64_t n, float eps);
void softmax(float *x, int64_t n);

// Forward-declared gguf context (opaque)
typedef struct gguf_ctx gguf_ctx_t;

typedef struct {
    model_hparams_t  hparams;
    gguf_ctx_t      *gguf;

    // Global weights
    void            *token_embd;
    enum ggml_dtype  token_embd_type;
    float           *output_norm;
    void            *output;
    enum ggml_dtype  output_type;

    // Per-layer weights
    layer_weights_t *layers;

    // KV cache
    kv_cache_t      *kv_cache;
    int              kv_pos;

    // SSM state
    ssm_state_t     *ssm_state;

    // NVMe expert I/O
    nvme_io_t       *nvme_io;

    // GPU context
    struct gpu_ctx  *gpu_ctx;

    // Scratch buffers
    float *buf_x;
    float *buf_h;
    float *buf_qkv;       // [max(ssm_qkv_dim, full_q_dim)]
    float *buf_k;
    float *buf_v;
    float *buf_attn;
    float *buf_attn_gate;
    float *buf_ssm_dt;
    float *buf_ssm_beta;
    float *buf_ffn;
    float *buf_router;
    float *buf_gate;
    float *buf_up;
    float *buf_down;
    float *buf_logits;
} model_t;

// From freq_profile.h
typedef struct {
    uint16_t layer;
    uint16_t expert_id;
    float    frequency;
} freq_entry_t;

typedef struct freq_profile {
    int           n_layers;
    int           n_entries;
    freq_entry_t *entries;
} freq_profile_t;

freq_profile_t *freq_profile_load(const char *path);
void freq_profile_free(freq_profile_t *fp);
int freq_profile_top_for_layer(const freq_profile_t *fp, int layer,
                               freq_entry_t *out, int max_out);

// From expert_cache.h
typedef struct {
    void     *d_buf;
    int       max_slots;
    int       slots_per_layer;
    int      *map;
    int      *slot_layer;
    int      *slot_expert;
    uint64_t *slot_ts;
    uint64_t  ts;
    int       n_layers;
    int       n_experts;
    uint64_t  expert_stride;
    uint64_t  hits;
    uint64_t  misses;
} vram_cache_t;

typedef struct {
    void     *buf;
    bool      pinned;
    int       max_slots;
    int       slots_per_layer;
    int      *map;
    int      *slot_layer;
    int      *slot_expert;
    uint64_t *slot_ts;
    uint64_t  ts;
    int       n_layers;
    int       n_experts;
    uint64_t  expert_stride;
    uint64_t  hits;
    uint64_t  misses;
} ram_cache_t;

int vram_cache_init(vram_cache_t *c, int n_layers, int n_experts,
                    int max_slots, uint64_t expert_stride);
void vram_cache_free(vram_cache_t *c);
int vram_cache_alloc_slot(vram_cache_t *c, int layer, int expert_id);

int ram_cache_init(ram_cache_t *c, int n_layers, int n_experts,
                   int max_slots, uint64_t expert_stride);
void ram_cache_free(ram_cache_t *c);
int ram_cache_alloc_slot(ram_cache_t *c, int layer, int expert_id);

static inline void vram_cache_touch(vram_cache_t *c, int slot) {
    c->slot_ts[slot] = ++c->ts;
}
static inline void ram_cache_touch(ram_cache_t *c, int slot) {
    c->slot_ts[slot] = ++c->ts;
}

// From car.h
typedef struct {
    float    threshold;
    int      max_subs_per_layer;
    int      n_layers;
    bool     skip_renorm;
    bool     layer_weight;
    int      warmup_tokens;
    float    free_ratio;
    float    budget_pool;
    float    budget_per_layer;
    uint64_t substitutions;
    uint64_t nvme_avoided;
    uint64_t nvme_remaining;
    uint64_t free_subs;
    double   score_ratio_sum;
} car_state_t;

void car_init(car_state_t *car, float threshold);
void car_reset_stats(car_state_t *car);
void car_begin_token(car_state_t *car);

int car_evaluate(
    car_state_t *car,
    int layer,
    int ram_layer,
    int *expert_ids,
    float *expert_scores,
    int n_experts,
    const float *all_scores,
    int n_total_experts,
    const vram_cache_t *vcache,
    const vram_cache_t *vcache1,
    const ram_cache_t *rcache,
    int *uncached_indices,
    int *n_uncached,
    int *vram_sub_indices,
    int *vram_sub_slots,
    int *n_vram_subs,
    int *vram1_sub_indices,
    int *vram1_sub_slots,
    int *n_vram1_subs,
    int *ram_sub_indices,
    int *ram_sub_slots,
    int *n_ram_subs
);

// --- Prefetch module (v2, opaque pointer) ---
struct prefetch_state;
typedef struct prefetch_state prefetch_state_t;

prefetch_state_t *prefetch_create(int nvme_budget, int spec_k,
                                   nvme_io_t *nvme_io, ram_cache_t *rcache,
                                   int nvme_buf_offset);
void prefetch_destroy(prefetch_state_t *ps);
int prefetch_classify_and_submit(prefetch_state_t *ps, int next_layer,
                                  const int *predicted_ids,
                                  const float *predicted_scores,
                                  int n_predicted,
                                  vram_cache_t *vcache, vram_cache_t *vcache1,
                                  ram_cache_t *rcache);
void prefetch_wait_nvme(prefetch_state_t *ps);
bool prefetch_try_complete_nvme(prefetch_state_t *ps);
void prefetch_commit_ram_cache(prefetch_state_t *ps, ram_cache_t *rcache);
void prefetch_reset(prefetch_state_t *ps);
int prefetch_get_ram_transfers(const prefetch_state_t *ps,
                                int *vram_slots, int *ram_slots, int max_out);
int prefetch_get_nvme_transfers(const prefetch_state_t *ps,
                                 int *vram_slots, int *ram_slots, int max_out);
void prefetch_mark_all_h2d_done(prefetch_state_t *ps);
int prefetch_vram_transfer_count(const prefetch_state_t *ps);

// From cpu_expert.h — CPU-side expert FFN compute
typedef struct cpu_expert_ctx cpu_expert_ctx_t;
typedef struct {
    const void *expert_data;
    float score;
    int gate_type, up_type, down_type;
    uint64_t expert_gate_size, expert_up_size;
} cpu_expert_work_t;
cpu_expert_ctx_t *cpu_expert_create(int n_embd, int expert_intermediate);
void cpu_expert_free(cpu_expert_ctx_t *ctx);
void cpu_expert_submit_async(cpu_expert_ctx_t *ctx, const float *input,
                             const cpu_expert_work_t *work, int n_work);
void cpu_expert_wait(cpu_expert_ctx_t *ctx);
float *cpu_expert_result(cpu_expert_ctx_t *ctx);
void cpu_expert_get_timing(cpu_expert_ctx_t *ctx, double *avg_wakeup_us,
                           double *avg_compute_us, int *samples);

} // extern "C"

#include "gpu_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/sysinfo.h>

// ---- Constants ----
#define K_SCALE_SIZE 12

// GGML type IDs (matching the enum above)
#define GGML_TYPE_F32_ID  0
#define GGML_TYPE_F16_ID  1
#define GGML_TYPE_Q8_0_ID 8
#define GGML_TYPE_Q4_K_ID 12
#define GGML_TYPE_Q5_K_ID 13
#define GGML_TYPE_Q6_K_ID 14
#define GGML_TYPE_MXFP4_ID 39

// Max model dimensions (supports up to Qwen3.5-397B: n_embd=4096, 512 experts, 10 used)
#define MAX_EXPERTS 10
#define MAX_N_EMBD 4096
#define MAX_INTERMEDIATE 2048
#define MAX_EXPERT_STRIDE (12 * 1024 * 1024)  // 12 MB
#define MAX_VOCAB 252000
#define MAX_KV_HEADS 64
#define MAX_HEADS 64
#define MAX_HEAD_DIM 256
#define MAX_CTX_LEN 4096
#define MAX_N_EXPERTS_TOTAL 512
#define MAX_LAYERS 256

// ---- Routing trace format (QMOE_TRACE_FILE) ----
#define TRACE_MAGIC   0x54524345  // "TRCE"
#define TRACE_VERSION 1

enum {
    TRACE_SRC_VRAM    = 0,
    TRACE_SRC_RAM     = 1,
    TRACE_SRC_NVME    = 2,
    TRACE_SRC_CAR     = 3,
    TRACE_SRC_DROPPED = 4,
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint16_t n_layers;
    uint16_t n_experts_total;
    uint16_t top_k;
    uint16_t reserved;
    uint64_t reserved2;
    uint64_t reserved3;
} trace_header_t;  // 32 bytes

typedef struct __attribute__((packed)) {
    uint32_t token_id;
    uint16_t layer_id;
    uint8_t  n_used;
    uint8_t  n_substitutions;
    int16_t  orig_ids[MAX_EXPERTS];
    float    orig_scores[MAX_EXPERTS];
    int16_t  final_ids[MAX_EXPERTS];
    float    final_scores[MAX_EXPERTS];
    uint8_t  source[MAX_EXPERTS];
    uint8_t  pad[6];
} trace_record_t;  // 144 bytes (top_k=10)

// ---- Per-layer GPU weight storage ----

struct gpu_layer {
    int attn_type;  // LAYER_ATTN_FULL or LAYER_ATTN_LINEAR

    // ---- Full attention weights (LAYER_ATTN_FULL only) ----
    void *d_wq, *d_wk, *d_wv, *d_wo;
    size_t wq_bytes, wk_bytes, wv_bytes, wo_bytes;
    int wq_type, wk_type, wv_type, wo_type;
    float *d_q_norm, *d_k_norm;          // [head_dim]

    // ---- DeltaNet/SSM weights (LAYER_ATTN_LINEAR only) ----
    void  *d_attn_qkv;    int attn_qkv_type;   size_t attn_qkv_bytes;
    void  *d_attn_gate;   int attn_gate_type;   size_t attn_gate_bytes;
    float *d_ssm_a;               // [time_step_rank] f32
    void  *d_ssm_alpha;  int ssm_alpha_type;   size_t ssm_alpha_bytes;
    void  *d_ssm_beta;   int ssm_beta_type;    size_t ssm_beta_bytes;
    float *d_ssm_conv1d;          // [conv_kernel * qkv_dim] f32
    float *d_ssm_dt_bias;         // [time_step_rank] f32
    float *d_ssm_norm;            // [ssm_state_size] f32
    void  *d_ssm_out;    int ssm_out_type;     size_t ssm_out_bytes;

    // ---- Common weights ----
    float *d_attn_norm, *d_ffn_norm;     // [n_embd]
    float *d_router;                     // [n_experts * n_embd] F32

    // ---- Shared expert weights ----
    float *d_shared_expert_gate;         // [n_embd] f32
    void  *d_shared_gate;  int shared_gate_type;  size_t shared_gate_bytes;
    void  *d_shared_up;    int shared_up_type;    size_t shared_up_bytes;
    void  *d_shared_down;  int shared_down_type;  size_t shared_down_bytes;
};

// ---- Ping-pong dual-GPU per-device resources ----

struct gpu_dev {
    int device_id;
    hipStream_t stream, transfer_stream;
    hipEvent_t batch_event[2];
    hipEvent_t spec_event;
    hipEvent_t prefetch_event;
    bool prefetch_pending;  // true if prefetch_event needs sync before using transfer_stream

    // Scratch buffers
    float *d_x, *d_norm, *d_q, *d_k, *d_v;
    float *d_attn_out, *d_attn_scores, *d_attn_gate_buf;
    float *d_router_out, *d_ffn_buf;
    float *d_logits;  // only on GPU with last layer
    int *d_top_k_ids; float *d_top_k_vals;
    float *d_ssm_dt, *d_ssm_beta_buf;
    int8_t *d_input_q8; float *d_input_q8_scales;
    float *d_shared_gate_buf, *d_shared_up_buf, *d_shared_down_buf;
    float *d_spec_router_out;
    int *d_spec_top_k_ids; float *d_spec_top_k_vals;

    // Expert FFN
    void *d_expert_staging;
    float *d_gate, *d_up, *d_down, *d_scores, *d_ffn_out;
    void **d_expert_ptrs;
    void *h_expert_ptrs[MAX_EXPERTS];

    // VRAM cache (this device's layers only)
    vram_cache_t ecache;

    // KV cache (this device's full-attention layers)
    __half *d_kv_k, *d_kv_v;
    int n_full_attn;
    int *full_attn_layer_ids; // maps local kv_index -> global layer_id

    // SSM state (this device's DeltaNet layers)
    float *d_ssm_states, *d_ssm_conv_bufs;
    int *d_ssm_conv_pos, *h_ssm_conv_pos;
    int n_ssm_layers;
    int *ssm_layer_ids; // maps local ssm_index -> global layer_id

    // LM head (only on GPU with last layer)
    void *d_output;
    float *d_output_norm;
    int output_type;
    size_t output_row_bytes;

    // Pinned host staging
    float *h_router_out;    // [n_experts]
    int *h_top_k_ids;       // [MAX_EXPERTS]
    float *h_top_k_vals;    // [MAX_EXPERTS]
    float *h_all_scores;    // [n_experts]
    int *h_spec_top_k_ids;  // [MAX_EXPERTS]
    float *h_norm;          // [n_embd]
    float *h_logits;        // [vocab_size] only on last-layer GPU

    // Per-device profiling events
    hipEvent_t prof_e[12];

    // Layer count for this device
    int n_layers;            // 30 for dual, 60 for single
};

typedef struct gpu_dev gpu_dev_t;

// ---- GPU context ----

struct gpu_ctx {
    hipStream_t stream;
    hipStream_t transfer_stream;   // H2D transfer stream for pipelining
    hipStream_t shared_stream;     // SharedFFN compute stream (overlaps with H2D)
    hipEvent_t  batch_event[2];    // sync events between streams
    hipEvent_t  shared_event;      // SharedFFN completion sync

    // --- Expert FFN buffers (existing) ---
    void  *d_expert_staging;   // 2 * MAX_EXPERTS * MAX_EXPERT_STRIDE (double-buffered)
    float *d_gate;             // [MAX_EXPERTS * MAX_INTERMEDIATE]
    float *d_up;               // [MAX_EXPERTS * MAX_INTERMEDIATE]
    float *d_down;             // [MAX_EXPERTS * MAX_N_EMBD]
    float *d_scores;           // [MAX_EXPERTS]
    float *d_ffn_out;          // [MAX_N_EMBD]
    // Per-expert pointer array for zero-copy VRAM cache access
    void  **d_expert_ptrs;     // [MAX_EXPERTS] device pointers
    void  *h_expert_ptrs[MAX_EXPERTS]; // host-side staging for upload

    // Registered buffer tracking
    void **registered_bufs;
    int    n_registered;

    // --- Full GPU inference state ---
    bool model_uploaded;

    // Per-layer weights on device
    struct gpu_layer *layers;
    int n_layers;

    // Global weights on device
    void  *d_token_embd;           // quantized embedding table
    void  *d_output;               // quantized LM head
    float *d_output_norm;          // [n_embd]
    int    token_embd_type, output_type;
    size_t token_embd_row_bytes;   // bytes per vocab row in quantized format
    size_t output_row_bytes;

    // KV cache on device (full attention layers only) — stored as FP16
    __half *d_kv_k;  // [n_full_attn * ctx_len * n_kv_heads * head_dim]
    __half *d_kv_v;  // same
    int n_full_attn;           // number of full attention layers
    int *full_attn_layer_ids;  // maps kv_index → layer_id

    // SSM state on device (DeltaNet layers only)
    float *d_ssm_states;     // [n_ssm * nh * sd * vd] contiguous
    float *d_ssm_conv_bufs;  // [n_ssm * (conv_k-1) * qkv_dim] contiguous
    int   *d_ssm_conv_pos;   // [n_ssm] conv buffer positions
    int   *h_ssm_conv_pos;   // [n_ssm] host copy (pinned)
    int    n_ssm_layers;     // number of SSM layers
    int   *ssm_layer_ids;    // maps ssm_index → layer_id

    // Scratch buffers on device
    float *d_x;              // [n_embd] residual stream
    float *d_norm;           // [n_embd] normalized hidden state
    float *d_q;              // [full_q_dim] (Q + gate for full attn, or QKV for SSM)
    float *d_k, *d_v;        // [n_kv_heads * head_dim]
    float *d_attn_out;       // [max(n_heads*head_dim, ssm_inner_size)]
    float *d_attn_scores;    // [n_heads * ctx_len]
    float *d_attn_gate_buf;  // [max(n_heads*head_dim, ssm_inner_size)] gate buffer
    float *d_router_out;     // [n_experts]
    float *d_ffn_buf;        // [n_embd]
    float *d_logits;         // [vocab_size]
    int   *d_top_k_ids;     // [MAX_EXPERTS] on device
    float *d_top_k_vals;    // [MAX_EXPERTS] on device
    float *d_ssm_dt;        // [time_step_rank] decay values
    float *d_ssm_beta_buf;  // [time_step_rank] beta values

    // Pre-quantized input for dp4a matvec (Q8_0 × Q8 input)
    int8_t *d_input_q8;          // [max_input_dim] quantized input
    float  *d_input_q8_scales;   // [max_input_dim/32] per-block scales

    // Shared expert scratch buffers
    float *d_shared_gate_buf;  // [shared_expert_intermediate]
    float *d_shared_up_buf;    // [shared_expert_intermediate]
    float *d_shared_down_buf;  // [n_embd]

    // Speculative look-ahead router buffers
    float *d_spec_router_out;  // [n_experts] speculative router output
    int   *d_spec_top_k_ids;   // [MAX_EXPERTS] speculative top-k IDs
    float *d_spec_top_k_vals;  // [MAX_EXPERTS] speculative top-k values
    int   *h_spec_top_k_ids;   // [MAX_EXPERTS] pinned host copy
    int    spec_ids[MAX_LAYERS][MAX_EXPERTS]; // saved speculative predictions per layer
    bool   spec_valid[MAX_LAYERS];            // whether prediction exists for this layer
    int    spec_n[MAX_LAYERS];               // number of predicted experts per layer
    int    spec_nvme_ids[MAX_LAYERS][8];     // predicted experts needing NVMe (from spec worker)
    int    spec_nvme_n[MAX_LAYERS];          // count per layer
    hipEvent_t spec_event;                    // GPU event for speculation completion

    // L+2 speculative routing: chain through L+1's shared expert
    int   *h_spec2_top_k_ids;  // [MAX_EXPERTS] pinned host copy for L+2 predictions
    int    spec2_ids[MAX_LAYERS][MAX_EXPERTS]; // saved L+2 predictions per layer
    bool   spec2_valid[MAX_LAYERS];
    int    spec2_n[MAX_LAYERS];
    hipEvent_t spec2_event;
    hipEvent_t vram_prefetch_event;            // GPU event for VRAM prefetch H2D completion
    bool       vram_prefetch_active;           // whether VRAM prefetch H2D was queued

    // Pinned host staging (for sync points)
    float *h_router_out;    // [n_experts] pinned for CPU softmax+top_k
    int   *h_top_k_ids;     // [MAX_EXPERTS] pinned
    float *h_top_k_vals;    // [MAX_EXPERTS] pinned
    float *h_all_scores;    // [n_experts] pinned — full softmax for cache-aware routing
    float *h_logits;         // [vocab_size] pinned

    // CPU expert compute: hidden state D2H + result staging
    float *h_norm;           // [n_embd] pinned — D2H of hidden state for CPU expert FFN
    float *h_cpu_partial;    // [n_embd] pinned — CPU partial weighted sum result
    float *d_cpu_partial;    // [n_embd] device — GPU reads CPU partial sum
    void  *cpu_expert;       // cpu_expert_ctx_t* (opaque, allocated after model upload)

    // Cached model params
    int n_embd, n_heads, n_kv_heads, head_dim, n_experts_total, n_experts_used;
    int expert_intermediate, shared_expert_intermediate, vocab_size, ctx_len;
    int rope_dim;            // partial RoPE dimension (64)
    float rope_freq_base, rms_norm_eps;
    uint64_t expert_gate_size, expert_up_size, expert_stride;
    int gate_type, up_type;

    // SSM cached params
    int ssm_inner_size, ssm_state_size, ssm_group_count;
    int ssm_time_step_rank, ssm_conv_kernel;
    int ssm_value_dim, ssm_qkv_dim, full_q_dim;

    // Expert VRAM cache (v2 module: layer-partitioned LRU)
    vram_cache_t ecache;

    // Expert RAM cache (v2 module: layer-partitioned LRU)
    ram_cache_t rcache;
    bool rcache_registered;   // true if hipHostRegister'd (needs hipHostUnregister)

    // Cache-aware routing (v2 module)
    car_state_t car;
    void *h2d_bounce;           // pinned bounce buffer for unpinned RAM cache → GPU copies
    int h2d_bounce_slots;       // number of bounce buffer slots allocated

    // Bounce buffer memcpy worker (parallel memcpy from RAM cache to bounce)
    struct {
        pthread_t thread;
        pthread_mutex_t mutex;
        pthread_cond_t cond_work;
        pthread_cond_t cond_done;
        bool work_ready;
        bool work_done;
        bool shutdown;
        bool active;

        // Work: copy from scattered RAM cache slots to contiguous bounce buffer
        void    *bounce;        // destination base (h2d_bounce)
        void    *cache;         // source base (rcache.buf)
        int      slots[16];     // source slot indices
        int      n;             // number of items
        uint64_t stride;
    } bounce_worker;

    int ram_cache_mb;           // configured size: -1=auto, 0=disabled, >0=MB
    int vram_cache_mb;          // configured size: -1=auto, 0=disabled, >0=MB

    // Async RAM cache population worker
    struct {
        pthread_t thread;
        pthread_mutex_t mutex;
        pthread_cond_t cond_work;
        pthread_cond_t cond_done;
        bool work_ready;
        bool work_done;
        bool shutdown;
        bool active;

        // Current batch of copies
        void    *srcs[16];      // source pointers (NVMe buffers or prefetch bufs)
        int      slots[16];     // destination slot indices in rcache
        int      n;             // number of items to copy
        void    *cache_buf;     // rcache.buf
        uint64_t stride;        // expert stride
    } rcache_pop;

    int car_protect_layers;  // first/last N layers exempt from CAR

    // Speculative expert prefetch (skip-attention routing prediction)
    // GPU predicts next layer's experts during Phase 1 FFN, loads uncached
    // predicted experts into RAM cache after e[7] (sync, capped at 3 experts).
    struct {
        int prev_experts[MAX_LAYERS][MAX_EXPERTS]; // previous token's expert IDs per layer
        bool has_prev;                              // false on first token
        bool active;                                // spec routing enabled
    } prefetch;

    // NVMe background loader (overlaps NVMe I/O with RAM cache memcpy)
    struct {
        pthread_t thread;
        pthread_mutex_t mutex;
        pthread_cond_t cond_work;
        pthread_cond_t cond_done;
        bool work_ready;
        bool work_done;
        bool shutdown;
        bool active;

        // Work params (set before signaling)
        nvme_io_t *io;
        int layer;
        int expert_ids[16];
        int n_experts;
        int buffer_offset;
        void *out_buffers[16];
        int result;

        // Per-expert H2D params (callback queues H2D as each read completes)
        hipStream_t h2d_stream;
        uint8_t *d_staging;
        size_t stg_base;
        int nv_start;
        size_t stride;
        bool h2d_enabled;
    } nvme_bg;

    // Speculative NVMe prefetch (v2 module: classify + async NVMe + VRAM transfer)
    prefetch_state_t *pf;
    bool pf_pending;          // cross-layer: async NVMe prefetch in flight
    bool no_cold_fetch;       // never block on NVMe — CAR handles all misses

    // Profiling
    struct {
        hipEvent_t e[12];
        double embed_ms;
        double attn_ms;
        double router_ms;
        double nvme_ms;
        double moe_pipe_ms;
        double output_ms;
        double rcwait_ms;     // rcache_pop_wait blocking time
        double hipsync_ms;    // hipStreamSynchronize blocking time (router sync)
        double classify_ms;   // expert classification + H2D submit time
        double gpu_ffn_ms;    // Phase 2 FFN (staged experts, after H2D)
        double h2d_wait_ms;   // GPU compute stream blocking on transfer stream
        double phase1_ffn_ms; // Phase 1 FFN (VRAM-cached experts, overlaps H2D)
        double nvme_wait_ms;  // CPU blocking on nvme_bg_wait()
        double shared_ffn_ms; // e[5]→e[11]: shared expert FFN + CPU classify overhead
        double post_ffn_ms;   // e[10]→e[7]: weighted_sum + shared_add + residual
        double attn_full_ms;  // Full attention layers only
        double attn_ssm_ms;   // SSM/DeltaNet layers only
        double ecache_d2d_ms; // VRAM cache D2D copy overhead
        double spec_pf_ms;   // CPU blocking on spec_pf_wait() (async NVMe prefetch)
        int token_count;
        // Expert overlap tracking (NVMe misses predictable by prev-token)
        int overlap_total;
        int overlap_hits;
        // Speculative routing accuracy
        int spec_total;       // total actual experts across all layers
        int spec_hits;        // correctly predicted by skip-attention router
        int spec_prefetched;  // experts loaded into RAM cache via speculative prefetch
        int vram_prefetched;  // experts prefetched from RAM→VRAM via spec routing
        // Cache-aware routing stats
        int car_substitutions;   // number of expert substitutions made
        int car_nvme_avoided;    // NVMe reads avoided by substitution
        int car_nvme_remaining;  // NVMe reads still needed
        int car_free_subs;       // subs that were "free" (ratio >= free_ratio)
        double car_score_ratio_sum; // sum of (substitute_score / original_score)
        // L+2 speculative routing accuracy
        int spec2_total;      // total actual experts at L+2 layers
        int spec2_hits;       // correctly predicted by chained L+1→L+2 routing
        int cold_fetch_dropped;  // experts dropped in no-cold-fetch mode
        // CPU expert compute stats
        double cpu_expert_ms;       // total CPU expert FFN compute time
        int cpu_expert_count;       // number of experts computed on CPU
        // GPU1 expert engine stats
        int gpu1_expert_count;      // experts computed on GPU1
        double gpu1_wait_ms;        // CPU blocking on GPU1 event sync
    } prof;

    // Expert frequency counting (for profile generation)
    uint32_t *expert_freq;    // [n_layers * n_experts_total] or NULL if disabled
    bool freq_enabled;

    // ---- Multi-GPU: expert-split TP ----
    // GPU0 ("main") uses existing fields above.
    // GPU1 ("expert engine") resources below — allocated only when n_devices >= 2.
    int n_devices;             // 1 or 2

    // ---- Ping-pong dual-GPU pipeline ----
    // Even layers on dev[0], odd layers on dev[1].
    // d_x relayed between GPUs via pinned staging after each layer.
    gpu_dev_t dev[2];
    bool pingpong;             // true when ping-pong mode active
    int  debug_level;          // 0=silent, 1=per-token, 2=per-layer (from QMOE_DEBUG)

    struct {
        int device_id;
        hipStream_t stream;
        hipStream_t transfer_stream;
        hipEvent_t  batch_event[2];

        // Expert FFN buffers (same layout as GPU0)
        void  *d_expert_staging;    // allocated in gpu_upload_model
        float *d_gate;              // [MAX_EXPERTS * MAX_INTERMEDIATE]
        float *d_up;                // [MAX_EXPERTS * MAX_INTERMEDIATE]
        float *d_down;              // [MAX_EXPERTS * MAX_N_EMBD]
        float *d_scores;            // [MAX_EXPERTS]
        float *d_ffn_out;           // [MAX_N_EMBD]
        void  **d_expert_ptrs;      // [MAX_EXPERTS] device pointers
        void  *h_expert_ptrs[MAX_EXPERTS]; // host staging

        // VRAM expert cache (layer-partitioned LRU, all 60 layers)
        vram_cache_t ecache;

        // Sync: GPU1 cache population (transfer_stream) → GPU1 FFN (stream)
        hipEvent_t cache_event;

        // Input from GPU0: hidden state for expert FFN
        float *d_norm;              // [n_embd] on GPU1 — allocated in gpu_upload_model

        // Output: partial weighted sum of GPU1's experts
        float *d_partial;           // [n_embd] on GPU1 — allocated in gpu_upload_model
    } gpu1;

    // Pinned host staging for inter-GPU transfer (allocated in gpu_upload_model)
    float *h_gpu1_norm;         // [n_embd] pinned — relay d_norm GPU0→GPU1
    float *h_gpu1_partial;      // [n_embd] pinned — relay partial sum GPU1→GPU0
    float *d_gpu1_partial;      // [n_embd] on GPU0 — receives GPU1's partial sum

    // Ping-pong spec routing worker (background thread for CPU-side router matvec)
    struct {
        pthread_t thread;
        pthread_mutex_t mutex;
        pthread_cond_t cond;
        bool active;         // worker thread is running
        bool work_ready;     // main thread signals: new work available
        bool shutdown;       // main thread signals: terminate

        // Input (set by main thread before signaling)
        hipEvent_t gpu_event;   // event to sync before CPU work
        float *h_input;         // pinned host buffer with D2H result (h_norm)
        const float *ffn_norm;  // host-side next layer's ffn_norm weight
        const float *router;    // host-side next layer's router weight [n_experts × n_embd]
        int n_embd;
        int n_experts;
        int n_used;
        float rms_eps;
        int next_il;
        int next_d;
        gpu_dev_t *next_dc;
        vram_cache_t *next_ec;
        ram_cache_t *rcache;
        uint64_t expert_stride;
        nvme_io_t *nvme_io;     // for NVMe reads on worker thread

        // Output (written by worker, read by main thread after worker finishes)
        int n_prefetched;
        int pred_ids[16];
        int n_pred;

        // Deferred prefetch: worker saves RAM-hit experts here instead of
        // doing VRAM alloc + DMA (not thread-safe). Main thread does the
        // actual VRAM alloc + hipMemcpyAsync at the prefetch sync point.
        int pf_eids[16];       // expert IDs to prefetch
        void *pf_srcs[16];     // source pointers (rcache.buf + slot * stride)
        int n_pf;              // number of deferred prefetches
        volatile bool work_done; // true when worker has finished current work
    } spec_worker;

    // ---- Background NVMe backfill for CAR-substituted experts ----
    // When CAR substitutes expert E with cached expert C, E is never loaded.
    // The backfill worker loads E from NVMe into RAM cache during idle cycles
    // so that future tokens find E in RAM instead of substituting again.
    struct {
        pthread_t       thread;
        pthread_mutex_t mutex;
        pthread_cond_t  cond_work;
        pthread_cond_t  cond_done;
        bool            active;
        bool            work_ready;
        bool            work_done;
        bool            shutdown;

        // Priority queue of pending backfill requests (capacity = 512)
        // Sorted by score_ratio ascending: worst substitutions backfilled first.
        // Set no_priority=true for FIFO ordering (QMOE_NO_PRIORITY_BACKFILL=1).
        int queue_layer[512];
        int queue_eid[512];
        float queue_ratio[512];  // CAR score ratio (lower = worse sub = higher priority)
        int queue_count;
        bool no_priority;

        // Dedup bitmap: 1 bit per (layer, expert) pair.
        // Prevents the same expert from filling the queue on repeated subs.
        // Main-thread only, no locking needed.
        uint64_t *dedup;       // allocated: ceil(n_layers * n_experts / 64) uint64s
        int       dedup_n_experts;  // n_experts_total, for indexing

        // Work batch (dequeued for current background cycle)
        int batch_layer[16];
        int batch_eid[16];
        int batch_n;
        int batch_max;  // max reads per dispatch (default 4, env QMOE_BACKFILL_N)
        bool sync_mode; // deterministic mode: once-per-token, blocking (env QMOE_BACKFILL_SYNC)

        // Resources (set at init, read-only after)
        nvme_io_t   *nvme_io;
        ram_cache_t *rcache;
        uint64_t     expert_stride;
        void        *buffers[16];  // page-aligned NVMe read buffers

        // Stats
        uint64_t total_backfilled;
        uint64_t total_skipped;
        uint64_t total_queued;
        uint64_t total_dropped;
        uint64_t total_deduped;
    } backfill;

    // Routing trace (QMOE_TRACE_FILE)
    struct {
        FILE    *fp;
        uint8_t *buf;
        size_t   buf_used;
        size_t   buf_cap;
    } trace;
};

// Forward declaration for spec routing worker thread
static void *pp_spec_worker_fn(void *arg);

// ---- GPU error checking ----

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        return nullptr; \
    } \
} while(0)

#define HIP_CHECK_V(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        return; \
    } \
} while(0)

#define HIP_CHECK_I(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ---- FP16 -> FP32 device helper ----

static __device__ float device_fp16_to_fp32(uint16_t h) {
    __half hv;
    memcpy(&hv, &h, sizeof(hv));
    return __half2float(hv);
}

// ---- Q4_K scale/min extraction (device) ----

static __device__ void device_get_scale_min_k4(int j, const uint8_t *q,
                                                uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// ============================================================
// KERNEL 1: Q4_K mat-vec (shared input, batched experts)
// Warp-per-row: 32 threads cooperate per row for coalesced access
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q4k_matvec_shared(
    void *const *expert_ptrs, const float *input, float *output,
    int M, int K, uint64_t weight_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q4_K *mat = (const block_q4_K *)(expert_data + weight_offset);
    int nb = K / QK_K;
    const block_q4_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        const float *yp = input + i * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            partial += (d1 * (qs[lane] & 0xF) - m1) * yp[j + lane];
            partial += (d2 * (qs[lane] >> 4)  - m2) * yp[j + 32 + lane];
            qs += 32;
            is += 2;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[expert * M + row] = partial;
}

// ============================================================
// KERNEL 2: Q4_K mat-vec (per-expert input, batched experts)
// Warp-per-row: 32 threads cooperate per row for coalesced access
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q4k_matvec_per_expert(
    void *const *expert_ptrs, const float *per_expert_input, float *output,
    int M, int K, uint64_t weight_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q4_K *mat = (const block_q4_K *)(expert_data + weight_offset);
    const float *input = per_expert_input + expert * K;
    int nb = K / QK_K;
    const block_q4_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        const float *yp = input + i * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            partial += (d1 * (qs[lane] & 0xF) - m1) * yp[j + lane];
            partial += (d2 * (qs[lane] >> 4)  - m2) * yp[j + 32 + lane];
            qs += 32;
            is += 2;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[expert * M + row] = partial;
}

// ============================================================
// KERNEL 3: Q6_K mat-vec (shared input, batched experts)
// Warp-per-row: 32 threads cooperate per row for coalesced access
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q6k_matvec_shared(
    void *const *expert_ptrs, const float *input, float *output,
    int M, int K, uint64_t weight_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q6_K *mat = (const block_q6_K *)(expert_data + weight_offset);
    int nb = K / QK_K;
    const block_q6_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = device_fp16_to_fp32(row_blocks[i].d);
        const uint8_t *ql = row_blocks[i].ql;
        const uint8_t *qh = row_blocks[i].qh;
        const int8_t  *sc = row_blocks[i].scales;
        const float *yp = input + i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            int is = lane / 16;
            int8_t q1 = (int8_t)((ql[lane +  0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[lane +  0] >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
            partial += d * sc[is + 0] * q1 * yp[lane +  0];
            partial += d * sc[is + 2] * q2 * yp[lane + 32];
            partial += d * sc[is + 4] * q3 * yp[lane + 64];
            partial += d * sc[is + 6] * q4 * yp[lane + 96];
            yp += 128; ql += 64; qh += 32; sc += 8;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[expert * M + row] = partial;
}

// ============================================================
// KERNEL 4: Q6_K mat-vec (per-expert input, batched experts)
// Warp-per-row: 32 threads cooperate per row for coalesced access
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q5k_matvec_per_expert(
    void *const *expert_ptrs, const float *per_expert_input, float *output,
    int M, int K, uint64_t weight_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q5_K *mat = (const block_q5_K *)(expert_data + weight_offset);
    const float *input = per_expert_input + expert * K;
    int nb = K / QK_K;
    const block_q5_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        const uint8_t *qh = row_blocks[i].qh;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        const float *yp = input + i * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            partial += (d1 * ((qs[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0)) - m1) * yp[j + lane];
            partial += (d2 * ((qs[lane] >> 4)  + ((qh[lane] & u2) ? 16 : 0)) - m2) * yp[j + 32 + lane];
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[expert * M + row] = partial;
}

// ============================================================
// KERNEL 4c: Q6_K mat-vec per expert (pointer array)
// ============================================================

__global__ void kernel_q6k_matvec_per_expert(
    void *const *expert_ptrs, const float *per_expert_input, float *output,
    int M, int K, uint64_t weight_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q6_K *mat = (const block_q6_K *)(expert_data + weight_offset);
    const float *input = per_expert_input + expert * K;
    int nb = K / QK_K;
    const block_q6_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = device_fp16_to_fp32(row_blocks[i].d);
        const uint8_t *ql = row_blocks[i].ql;
        const uint8_t *qh = row_blocks[i].qh;
        const int8_t  *sc = row_blocks[i].scales;
        const float *yp = input + i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            int is = lane / 16;
            int8_t q1 = (int8_t)((ql[lane +  0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[lane +  0] >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
            partial += d * sc[is + 0] * q1 * yp[lane +  0];
            partial += d * sc[is + 2] * q2 * yp[lane + 32];
            partial += d * sc[is + 4] * q3 * yp[lane + 64];
            partial += d * sc[is + 6] * q4 * yp[lane + 96];
            yp += 128; ql += 64; qh += 32; sc += 8;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[expert * M + row] = partial;
}

// ============================================================
// KERNEL 5: SiLU(gate) * up (element-wise, batched)
// ============================================================

__global__ void kernel_silu_mul(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    gate[i] = s * up[i];
}

// ============================================================
// KERNEL 5b: Fused Gate+Up+SiLU for Q4_K (shared input, batched experts)
// Reads input once, computes gate and up matvecs, outputs SiLU(gate)*up
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q4k_fused_gate_up_silu(
    void *const *expert_ptrs, const float *input, float *output,
    int M, int K, uint64_t gate_offset, uint64_t up_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q4_K *gate_mat = (const block_q4_K *)(expert_data + gate_offset);
    const block_q4_K *up_mat   = (const block_q4_K *)(expert_data + up_offset);
    int nb = K / QK_K;
    const block_q4_K *gate_row = gate_mat + (int64_t)row * nb;
    const block_q4_K *up_row   = up_mat   + (int64_t)row * nb;

    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float *yp = input + i * QK_K;

        // Gate block
        float gd    = device_fp16_to_fp32(gate_row[i].d);
        float gdmin = device_fp16_to_fp32(gate_row[i].dmin);
        const uint8_t *gqs = gate_row[i].qs;
        // Up block
        float ud    = device_fp16_to_fp32(up_row[i].d);
        float udmin = device_fp16_to_fp32(up_row[i].dmin);
        const uint8_t *uqs = up_row[i].qs;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, gate_row[i].scales, &sc, &m);
            float gd1 = gd * sc, gm1 = gdmin * m;
            device_get_scale_min_k4(is + 1, gate_row[i].scales, &sc, &m);
            float gd2 = gd * sc, gm2 = gdmin * m;
            device_get_scale_min_k4(is + 0, up_row[i].scales, &sc, &m);
            float ud1 = ud * sc, um1 = udmin * m;
            device_get_scale_min_k4(is + 1, up_row[i].scales, &sc, &m);
            float ud2 = ud * sc, um2 = udmin * m;

            float y0 = yp[j + lane];
            float y1 = yp[j + 32 + lane];
            gate_partial += (gd1 * (gqs[lane] & 0xF) - gm1) * y0;
            gate_partial += (gd2 * (gqs[lane] >> 4)  - gm2) * y1;
            up_partial   += (ud1 * (uqs[lane] & 0xF) - um1) * y0;
            up_partial   += (ud2 * (uqs[lane] >> 4)  - um2) * y1;
            gqs += 32;
            uqs += 32;
            is += 2;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_partial += __shfl_down(gate_partial, offset);
        up_partial   += __shfl_down(up_partial, offset);
    }
    if (lane == 0) {
        float g = gate_partial;
        float s = g / (1.0f + expf(-g));  // SiLU
        output[expert * M + row] = s * up_partial;
    }
}

// ============================================================
// KERNEL 5c: Fused Gate+Up+SiLU for Q6_K (shared input, batched experts)
// Grid: (ceil(M/RPB), n_experts), Block: (32, RPB)
// ============================================================

__global__ void kernel_q6k_fused_gate_up_silu(
    void *const *expert_ptrs, const float *input, float *output,
    int M, int K, uint64_t gate_offset, uint64_t up_offset)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int expert = blockIdx.y;
    int lane = threadIdx.x;  // 0-31
    if (row >= M) return;

    const uint8_t *expert_data = (const uint8_t *)expert_ptrs[expert];
    const block_q6_K *gate_mat = (const block_q6_K *)(expert_data + gate_offset);
    const block_q6_K *up_mat   = (const block_q6_K *)(expert_data + up_offset);
    int nb = K / QK_K;
    const block_q6_K *gate_row = gate_mat + (int64_t)row * nb;
    const block_q6_K *up_row   = up_mat   + (int64_t)row * nb;

    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float *yp = input + i * QK_K;

        float gd = device_fp16_to_fp32(gate_row[i].d);
        const uint8_t *gql = gate_row[i].ql;
        const uint8_t *gqh = gate_row[i].qh;
        const int8_t  *gsc = gate_row[i].scales;

        float ud = device_fp16_to_fp32(up_row[i].d);
        const uint8_t *uql = up_row[i].ql;
        const uint8_t *uqh = up_row[i].qh;
        const int8_t  *usc = up_row[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            int is = lane / 16;
            // Gate Q6_K dequant
            int8_t gq1 = (int8_t)((gql[lane +  0] & 0xF) | (((gqh[lane] >> 0) & 3) << 4)) - 32;
            int8_t gq2 = (int8_t)((gql[lane + 32] & 0xF) | (((gqh[lane] >> 2) & 3) << 4)) - 32;
            int8_t gq3 = (int8_t)((gql[lane +  0] >> 4)  | (((gqh[lane] >> 4) & 3) << 4)) - 32;
            int8_t gq4 = (int8_t)((gql[lane + 32] >> 4)  | (((gqh[lane] >> 6) & 3) << 4)) - 32;
            // Up Q6_K dequant
            int8_t uq1 = (int8_t)((uql[lane +  0] & 0xF) | (((uqh[lane] >> 0) & 3) << 4)) - 32;
            int8_t uq2 = (int8_t)((uql[lane + 32] & 0xF) | (((uqh[lane] >> 2) & 3) << 4)) - 32;
            int8_t uq3 = (int8_t)((uql[lane +  0] >> 4)  | (((uqh[lane] >> 4) & 3) << 4)) - 32;
            int8_t uq4 = (int8_t)((uql[lane + 32] >> 4)  | (((uqh[lane] >> 6) & 3) << 4)) - 32;

            float y0 = yp[lane +  0];
            float y1 = yp[lane + 32];
            float y2 = yp[lane + 64];
            float y3 = yp[lane + 96];

            gate_partial += gd * gsc[is + 0] * gq1 * y0;
            gate_partial += gd * gsc[is + 2] * gq2 * y1;
            gate_partial += gd * gsc[is + 4] * gq3 * y2;
            gate_partial += gd * gsc[is + 6] * gq4 * y3;

            up_partial += ud * usc[is + 0] * uq1 * y0;
            up_partial += ud * usc[is + 2] * uq2 * y1;
            up_partial += ud * usc[is + 4] * uq3 * y2;
            up_partial += ud * usc[is + 6] * uq4 * y3;

            yp += 128; gql += 64; gqh += 32; gsc += 8;
            uql += 64; uqh += 32; usc += 8;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_partial += __shfl_down(gate_partial, offset);
        up_partial   += __shfl_down(up_partial, offset);
    }
    if (lane == 0) {
        float g = gate_partial;
        float s = g / (1.0f + expf(-g));  // SiLU
        output[expert * M + row] = s * up_partial;
    }
}

// ============================================================
// KERNEL 5d: Multi-warp fused Gate+Up+SiLU for SharedFFN (Q4_K)
// Multiple warps per row for better SM occupancy on small M.
// Block: (32 * nwarps, RPB), Grid: (ceil(M/RPB))
// Shared mem: RPB * nwarps * 2 floats (gate + up partials)
// ============================================================

__global__ void kernel_q4k_fused_gus_mw(
    const void *gate_weight, const void *up_weight,
    const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int nw = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    const block_q4_K *gate_mat = (const block_q4_K *)gate_weight;
    const block_q4_K *up_mat = (const block_q4_K *)up_weight;
    int nb = K / QK_K;

    float gate_partial = 0.0f, up_partial = 0.0f;

    if (row < M) {
        const block_q4_K *gate_row = gate_mat + (int64_t)row * nb;
        const block_q4_K *up_row = up_mat + (int64_t)row * nb;

        // Each warp handles a chunk of Q4_K blocks along K
        int bpw = (nb + nw - 1) / nw;
        int start = warp_id * bpw;
        int end = start + bpw;
        if (end > nb) end = nb;

        for (int i = start; i < end; i++) {
            const float *yp = input + i * QK_K;

            float gd    = device_fp16_to_fp32(gate_row[i].d);
            float gdmin = device_fp16_to_fp32(gate_row[i].dmin);
            const uint8_t *gqs = gate_row[i].qs;

            float ud    = device_fp16_to_fp32(up_row[i].d);
            float udmin = device_fp16_to_fp32(up_row[i].dmin);
            const uint8_t *uqs = up_row[i].qs;

            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                device_get_scale_min_k4(is + 0, gate_row[i].scales, &sc, &m);
                float gd1 = gd * sc, gm1 = gdmin * m;
                device_get_scale_min_k4(is + 1, gate_row[i].scales, &sc, &m);
                float gd2 = gd * sc, gm2 = gdmin * m;
                device_get_scale_min_k4(is + 0, up_row[i].scales, &sc, &m);
                float ud1 = ud * sc, um1 = udmin * m;
                device_get_scale_min_k4(is + 1, up_row[i].scales, &sc, &m);
                float ud2 = ud * sc, um2 = udmin * m;

                float y0 = yp[j + lane];
                float y1 = yp[j + 32 + lane];
                gate_partial += (gd1 * (gqs[lane] & 0xF) - gm1) * y0;
                gate_partial += (gd2 * (gqs[lane] >> 4)  - gm2) * y1;
                up_partial   += (ud1 * (uqs[lane] & 0xF) - um1) * y0;
                up_partial   += (ud2 * (uqs[lane] >> 4)  - um2) * y1;
                gqs += 32;
                uqs += 32;
                is += 2;
            }
        }

        // Intra-warp reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_partial += __shfl_down(gate_partial, offset);
            up_partial   += __shfl_down(up_partial, offset);
        }
    }

    // Inter-warp reduction via shared memory
    extern __shared__ float smem[];
    int base = threadIdx.y * nw * 2;
    if (lane == 0) {
        smem[base + warp_id] = gate_partial;
        smem[base + nw + warp_id] = up_partial;
    }
    __syncthreads();

    if (row < M && warp_id == 0 && lane == 0) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        for (int w = 0; w < nw; w++) {
            gate_sum += smem[base + w];
            up_sum   += smem[base + nw + w];
        }
        float s = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = s * up_sum;
    }
}

// ============================================================
// KERNEL 5e: Multi-warp Q4_K mat-vec (single matrix)
// Multiple warps per row, inter-warp reduction via shared memory.
// Block: (32 * nwarps, RPB), Grid: (ceil(M/RPB))
// Shared mem: RPB * nwarps floats
// ============================================================

__global__ void kernel_q4k_matvec_mw(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int nw = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    const block_q4_K *mat = (const block_q4_K *)weight;
    int nb = K / QK_K;

    float partial = 0.0f;

    if (row < M) {
        const block_q4_K *row_blocks = mat + (int64_t)row * nb;

        int bpw = (nb + nw - 1) / nw;
        int start = warp_id * bpw;
        int end = start + bpw;
        if (end > nb) end = nb;

        for (int i = start; i < end; i++) {
            const uint8_t *qs = row_blocks[i].qs;
            float d    = device_fp16_to_fp32(row_blocks[i].d);
            float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
            const float *yp = input + i * QK_K;
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
                float d1 = d * sc, m1 = dmin * m;
                device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
                float d2 = d * sc, m2 = dmin * m;
                partial += (d1 * (qs[lane] & 0xF) - m1) * yp[j + lane];
                partial += (d2 * (qs[lane] >> 4)  - m2) * yp[j + 32 + lane];
                qs += 32;
                is += 2;
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down(partial, offset);
    }

    extern __shared__ float smem[];
    int base = threadIdx.y * nw;
    if (lane == 0) smem[base + warp_id] = partial;
    __syncthreads();

    if (row < M && warp_id == 0 && lane == 0) {
        float sum = 0.0f;
        for (int w = 0; w < nw; w++)
            sum += smem[base + w];
        output[row] = sum;
    }
}

// ============================================================
// KERNEL 6: Weighted sum across experts
// ============================================================

__global__ void kernel_weighted_sum(const float *down, const float *scores,
                                     float *ffn_out, int n_embd, int n_experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_experts; e++)
        acc += scores[e] * down[e * n_embd + i];
    ffn_out[i] = acc;
}

// ============================================================
// KERNEL 6b: Fused MoE output + shared expert + residual
// Combines: weighted_sum + vec_add(shared) + vec_add(residual)
// output[i] = residual[i] + shared[i] + sum_e(scores[e] * down[e*n+i])
// ============================================================

__global__ void kernel_moe_output_residual(
    const float *down, const float *scores,
    const float *shared_out, const float *residual,
    float *output, int n_embd, int n_experts, bool has_shared)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_experts; e++)
        acc += scores[e] * down[e * n_embd + i];
    if (has_shared) acc += shared_out[i];
    output[i] = residual[i] + acc;
}

// ============================================================
// KERNEL 7: Q4_K mat-vec (single matrix, direct pointer)
// Grid: (ceil(M/256)), Block: (256)
// For attention Q/K/V/O projections and LM head
// ============================================================

__global__ void kernel_q4k_matvec_single(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const block_q4_K *mat = (const block_q4_K *)weight;
    int nb = K / QK_K;
    const block_q4_K *row_blocks = mat + (int64_t)row * nb;

    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        float d   = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            int base = i * QK_K + j;
            for (int l = 0; l < 32; ++l)
                sum += (d1 * (qs[l] & 0xF) - m1) * input[base + l];
            for (int l = 0; l < 32; ++l)
                sum += (d2 * (qs[l] >> 4) - m2) * input[base + 32 + l];
            qs += 32;
            is += 2;
        }
    }
    output[row] = sum;
}

// ============================================================
// KERNEL 8: Q6_K mat-vec (single matrix, direct pointer)
// ============================================================

__global__ void kernel_q6k_matvec_single(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const block_q6_K *mat = (const block_q6_K *)weight;
    int nb = K / QK_K;
    const block_q6_K *row_blocks = mat + (int64_t)row * nb;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = device_fp16_to_fp32(row_blocks[i].d);
        const uint8_t *ql = row_blocks[i].ql;
        const uint8_t *qh = row_blocks[i].qh;
        const int8_t  *sc = row_blocks[i].scales;
        const float *yp = input + i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                sumf += d * sc[is + 0] * q1 * yp[l +  0];
                sumf += d * sc[is + 2] * q2 * yp[l + 32];
                sumf += d * sc[is + 4] * q3 * yp[l + 64];
                sumf += d * sc[is + 6] * q4 * yp[l + 96];
            }
            yp += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    output[row] = sumf;
}

// ============================================================
// KERNEL: Q6_K mat-vec cooperative (32 threads per row, warp shuffle)
// Grid: (ceil(M/ROWS_PER_BLOCK)), Block: (32, ROWS_PER_BLOCK)
// Much faster than 1-thread-per-row for large M (e.g. LM head)
// ============================================================
__global__ void kernel_q6k_matvec_coop(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int lane = threadIdx.x;  // 0-31

    const block_q6_K *mat = (const block_q6_K *)weight;
    int nb = K / QK_K;
    const block_q6_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = device_fp16_to_fp32(row_blocks[i].d);
        const uint8_t *ql = row_blocks[i].ql;
        const uint8_t *qh = row_blocks[i].qh;
        const int8_t  *sc = row_blocks[i].scales;
        const float *yp = input + i * QK_K;

        // Sub-block 0: elements 0-127 (lane maps to l=0..31)
        {
            int is = lane / 16;
            int8_t q1 = (int8_t)((ql[lane +  0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[lane +  0] >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
            partial += d * sc[is + 0] * q1 * yp[lane +  0];
            partial += d * sc[is + 2] * q2 * yp[lane + 32];
            partial += d * sc[is + 4] * q3 * yp[lane + 64];
            partial += d * sc[is + 6] * q4 * yp[lane + 96];
        }

        // Sub-block 1: elements 128-255
        {
            const uint8_t *ql2 = ql + 64;
            const uint8_t *qh2 = qh + 32;
            const int8_t  *sc2 = sc + 8;
            const float *yp2 = yp + 128;

            int is = lane / 16;
            int8_t q1 = (int8_t)((ql2[lane +  0] & 0xF) | (((qh2[lane] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql2[lane + 32] & 0xF) | (((qh2[lane] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql2[lane +  0] >> 4)  | (((qh2[lane] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql2[lane + 32] >> 4)  | (((qh2[lane] >> 6) & 3) << 4)) - 32;
            partial += d * sc2[is + 0] * q1 * yp2[lane +  0];
            partial += d * sc2[is + 2] * q2 * yp2[lane + 32];
            partial += d * sc2[is + 4] * q3 * yp2[lane + 64];
            partial += d * sc2[is + 6] * q4 * yp2[lane + 96];
        }
    }

    // Warp reduction (32-wide wavefront on RDNA)
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down(partial, offset);
    }

    if (lane == 0) {
        output[row] = partial;
    }
}

// ============================================================
// KERNEL: Q5_K mat-vec (single matrix, direct pointer)
// Grid: (ceil(M/256)), Block: (256)
// ============================================================

__global__ void kernel_q5k_matvec_single(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const block_q5_K *mat = (const block_q5_K *)weight;
    int nb = K / QK_K;
    const block_q5_K *row_blocks = mat + (int64_t)row * nb;

    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        const uint8_t *qh = row_blocks[i].qh;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            int base = i * QK_K + j;
            for (int l = 0; l < 32; ++l)
                sum += (d1 * ((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * input[base + l];
            for (int l = 0; l < 32; ++l)
                sum += (d2 * ((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2) * input[base + 32 + l];
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    output[row] = sum;
}

// ============================================================
// KERNEL: Q4_K mat-vec cooperative (32 threads per row, warp shuffle)
// Grid: (ceil(M/ROWS_PER_BLOCK)), Block: (32, ROWS_PER_BLOCK)
// ============================================================
__global__ void kernel_q4k_matvec_coop(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int lane = threadIdx.x;  // 0-31

    const block_q4_K *mat = (const block_q4_K *)weight;
    int nb = K / QK_K;
    const block_q4_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        const float *yp = input + i * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            partial += (d1 * (qs[lane] & 0xF) - m1) * yp[j + lane];
            partial += (d2 * (qs[lane] >> 4)  - m2) * yp[j + 32 + lane];
            qs += 32;
            is += 2;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down(partial, offset);
    }
    if (lane == 0) output[row] = partial;
}

// ============================================================
// KERNEL: Q8_0 mat-vec cooperative (32 threads per row, warp shuffle)
// Grid: (ceil(M/ROWS_PER_BLOCK)), Block: (32, ROWS_PER_BLOCK)
// ============================================================
__global__ void kernel_q8_0_matvec_coop(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int lane = threadIdx.x;  // 0-31

    const block_q8_0 *mat = (const block_q8_0 *)weight;
    int nb = K / QK8_0;
    const block_q8_0 *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = device_fp16_to_fp32(row_blocks[i].d);
        partial += d * (float)row_blocks[i].qs[lane] * input[i * QK8_0 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[row] = partial;
}

// ============================================================
// dp4a: packed int8x4 dot product (RDNA 3/4 sudot4 instruction)
// Computes c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
// where a,b are packed int8x4 in int32
// ============================================================
static __device__ __forceinline__ int dp4a_s8(int a, int b, int c) {
#ifdef QMOE_CUDA
    return __dp4a(a, b, c);
#else
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#endif
}

// ============================================================
// KERNEL: Quantize float vector to Q8 (int8 + per-block scale)
// Grid: (K/32), Block: (32)
// ============================================================
__global__ void kernel_quantize_f32_q8(
    const float *input, int8_t *output_qs, float *output_scales, int K)
{
    int bid = blockIdx.x;
    int lane = threadIdx.x;
    int idx = bid * 32 + lane;

    float val = (idx < K) ? input[idx] : 0.0f;
    float abs_val = fabsf(val);

    // Warp max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, offset));

    float max_val = abs_val;
    float d = max_val / 127.0f;
    float id = (max_val != 0.0f) ? 127.0f / max_val : 0.0f;

    output_qs[idx] = (int8_t)rintf(val * id);
    if (lane == 0) output_scales[bid] = d;
}

// ============================================================
// KERNEL: Q8_0 mat-vec with dp4a (pre-quantized Q8 input)
// Each lane processes full Q8_0 blocks at stride-32, doing 8 dp4a per block.
// Grid: (ceil(M/RPB)), Block: (32, RPB)
// ============================================================
__global__ void kernel_q8_0_matvec_dp4a(
    const void *weight, const int8_t *input_qs, const float *input_scales,
    float *output, int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    int lane = threadIdx.x;  // 0-31

    const block_q8_0 *mat = (const block_q8_0 *)weight;
    int nb = K / QK8_0;
    const block_q8_0 *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float w_d = device_fp16_to_fp32(row_blocks[b].d);
        float x_d = input_scales[b];

        // Q8_0 blocks are 34 bytes (2+32), so qs is at offset 2 —
        // not 4-byte aligned. Use memcpy to avoid misaligned int reads.
        int w_i32[8];
        memcpy(w_i32, row_blocks[b].qs, 32);
        const int *x_i32 = (const int *)(input_qs + b * 32);

        int isum = 0;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            isum = dp4a_s8(w_i32[p], x_i32[p], isum);
        }
        partial += w_d * x_d * (float)isum;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[row] = partial;
}

// ============================================================
// KERNEL: Q5_K mat-vec cooperative (32 threads per row, warp shuffle)
// Grid: (ceil(M/ROWS_PER_BLOCK)), Block: (32, ROWS_PER_BLOCK)
// ============================================================
__global__ void kernel_q5k_matvec_coop(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int lane = threadIdx.x;  // 0-31

    const block_q5_K *mat = (const block_q5_K *)weight;
    int nb = K / QK_K;
    const block_q5_K *row_blocks = mat + (int64_t)row * nb;

    float partial = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = row_blocks[i].qs;
        const uint8_t *qh = row_blocks[i].qh;
        float d    = device_fp16_to_fp32(row_blocks[i].d);
        float dmin = device_fp16_to_fp32(row_blocks[i].dmin);
        const float *yp = input + i * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            device_get_scale_min_k4(is + 0, row_blocks[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            device_get_scale_min_k4(is + 1, row_blocks[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            partial += (d1 * ((qs[lane] & 0xF) + ((qh[lane] & u1) ? 16 : 0)) - m1) * yp[j + lane];
            partial += (d2 * ((qs[lane] >> 4)  + ((qh[lane] & u2) ? 16 : 0)) - m2) * yp[j + 32 + lane];
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down(partial, offset);
    }
    if (lane == 0) output[row] = partial;
}

// ============================================================
// KERNEL 8b: MXFP4 mat-vec (single matrix, direct pointer)
// Grid: (ceil(M/256)), Block: (256)
// For shared expert gate/up projections
// ============================================================

__constant__ float d_mxfp4_table[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__global__ void kernel_mxfp4_matvec_single(
    const void *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const block_mxfp4 *mat = (const block_mxfp4 *)weight;
    int nb = K / QK_MXFP4;
    const block_mxfp4 *row_blocks = mat + (int64_t)row * nb;

    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float scale;
        uint8_t s = row_blocks[i].s;
        if (s == 0xFF) {
            scale = 0.0f;
        } else {
            scale = ldexpf(1.0f, (int)s - 127);
        }
        float block_sum = 0.0f;
        int base = i * QK_MXFP4;
        // GGML layout: low nibbles at positions 0..15, high nibbles at 16..31
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            uint8_t byte = row_blocks[i].qs[j];
            block_sum += d_mxfp4_table[byte & 0x0F] * input[base + j];
            block_sum += d_mxfp4_table[byte >> 4]   * input[base + j + QK_MXFP4 / 2];
        }
        sum += scale * block_sum;
    }
    output[row] = sum;
}

// ============================================================
// KERNEL 9: F32 mat-vec (single matrix)
// For router weights
// ============================================================

__global__ void kernel_f32_matvec(
    const float *weight, const float *input, float *output,
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float *row_data = weight + (int64_t)row * K;
    float sum = 0.0f;
    for (int i = 0; i < K; i++)
        sum += row_data[i] * input[i];
    output[row] = sum;
}

// ============================================================
// KERNEL 9b: F32 mat-vec cooperative (32 threads per row, warp shuffle)
// Grid: (ceil(M/RPB)), Block: (32, RPB)
// ============================================================

__global__ void kernel_f32_matvec_coop(
    const float *weight, const float *input, float *output,
    int M, int K)
{
    extern __shared__ float s_input[];
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;  // 0-31
    int tid = threadIdx.y * 32 + lane;
    int block_threads = blockDim.y * 32;

    // Collaborative load: cache input vector in shared memory
    for (int i = tid; i < K; i += block_threads)
        s_input[i] = input[i];
    __syncthreads();

    if (row >= M) return;

    // float4 vectorized dot product — 4x fewer loop iterations
    const float4 *row4 = (const float4 *)(weight + (int64_t)row * K);
    const float4 *inp4 = (const float4 *)s_input;
    int K4 = K >> 2;

    float partial = 0.0f;
    for (int i = lane; i < K4; i += 32) {
        float4 w = row4[i];
        float4 v = inp4[i];
        partial += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);
    if (lane == 0) output[row] = partial;
}

// ============================================================
// KERNEL 10: RMS Norm
// Block: (256), Grid: (1)
// Two-pass: sum of squares -> normalize * weight
// ============================================================

__global__ void kernel_rms_norm(const float *x, const float *weight,
                                 float *output, int n, float eps) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Phase 1: sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += 256)
        local_sum += x[i] * x[i];
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)n + eps);

    // Phase 2: normalize and scale
    for (int i = tid; i < n; i += 256)
        output[i] = x[i] * rms * weight[i];
}

// ============================================================
// KERNEL 11: Per-head RMS Norm (Qwen3-specific)
// Block: (128), Grid: (n_heads)
// Each block normalizes one head of head_dim elements
// ============================================================

__global__ void kernel_per_head_rms_norm(float *data, const float *weight,
                                          int head_dim, float eps) {
    __shared__ float sdata[256];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    float *head_data = data + head * head_dim;

    // Sum of squares (one element per thread, blockDim.x >= head_dim)
    float val = (tid < head_dim) ? head_data[tid] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)head_dim + eps);

    if (tid < head_dim)
        head_data[tid] = val * rms * weight[tid];
}

// ============================================================
// KERNEL 12: RoPE (Rotary Position Embeddings)
// Block: (256), Grid: (1)
// Processes all Q and K heads in a single kernel
// ============================================================

// Partial NEOX-style RoPE: rotates pairs (j, j+half) for j in [0, half).
// Only the first rope_dim dimensions are rotated; remaining pass through.
// Frequency: 1 / freq_base^(2*j / rope_dim)
__global__ void kernel_rope(float *q, float *k, int pos,
                             int head_dim, int rope_dim, int n_heads, int n_kv_heads,
                             float freq_base) {
    int tid = threadIdx.x;
    int half = rope_dim / 2;
    int total_pairs = (n_heads + n_kv_heads) * half;

    for (int idx = tid; idx < total_pairs; idx += blockDim.x) {
        int h = idx / half;
        int j = idx % half;

        float *vec;
        if (h < n_heads)
            vec = q + h * head_dim;
        else
            vec = k + (h - n_heads) * head_dim;

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

// ============================================================
// KERNEL: Deinterleave Q+gate from [Q_h0, Gate_h0, Q_h1, Gate_h1, ...]
// into [Q_all, Gate_all] contiguous buffers
// Grid: (1), Block: (256)
// ============================================================
__global__ void kernel_deinterleave_q_gate(float *qg, float *gate_out,
                                            int n_heads, int head_dim) {
    int inner_dim = n_heads * head_dim;
    // qg has layout: [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...]
    // Deinterleave in-place: first inner_dim = Q, second inner_dim = gate
    // We need a temporary approach since we read and write the same buffer.
    // Strategy: copy gate values to gate_out, shift Q values down.
    // But both reads and writes overlap, so we process per-head.
    // Since we're on GPU with shared memory, use shared mem for staging.
    extern __shared__ float smem[];
    int total = 2 * inner_dim;

    // First: copy entire buffer to shared memory
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        smem[i] = qg[i];
    }
    __syncthreads();

    // Now write Q contiguous and gate contiguous
    for (int i = threadIdx.x; i < inner_dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        qg[i] = smem[h * 2 * head_dim + d];           // Q
        gate_out[i] = smem[h * 2 * head_dim + head_dim + d]; // Gate
    }
}

// ============================================================
// KERNEL: Apply sigmoid gate to attention output: out[i] *= sigmoid(gate[i])
// Grid: (ceil(n/256)), Block: (256)
// ============================================================
__global__ void kernel_sigmoid_gate(float *out, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        out[i] *= g;
    }
}

// ============================================================
// KERNEL 13: KV cache update
// Block: (256), Grid: (1)
// Copy d_k and d_v into KV cache at the given position
// ============================================================

__global__ void kernel_kv_update(__half *kv_k, __half *kv_v,
                                  const float *k_in, const float *v_in,
                                  int kv_dim, int layer, int pos,
                                  int ctx_len) {
    int tid = threadIdx.x;
    // kv_k layout: [n_layers * ctx_len * kv_dim] in FP16
    int64_t base = (int64_t)layer * ctx_len * kv_dim + (int64_t)pos * kv_dim;
    for (int i = tid; i < kv_dim; i += 256) {
        kv_k[base + i] = __float2half(k_in[i]);
        kv_v[base + i] = __float2half(v_in[i]);
    }
}

// ============================================================
// KERNEL 14: GQA Attention
// Block: (256), Grid: (n_heads)
// Each block processes one query head
// ============================================================

// GQA Attention with FP16 KV cache for 2× bandwidth reduction.
// 3-phase approach: (1) Q·K scores, (2) softmax, (3) weighted V accumulation.
// Block: (256), Grid: (n_heads)
__global__ void kernel_gqa_attention(
    const float *d_q,          // [n_heads * head_dim]
    const __half *d_kv_k,      // [n_layers * ctx_len * kv_dim] FP16
    const __half *d_kv_v,      // same
    float *d_attn_out,         // [n_heads * head_dim]
    int layer, int pos,
    int head_dim, int n_heads, int n_kv_heads, int ctx_len)
{
    extern __shared__ float smem[];  // [seq_len] floats for scores
    int h = blockIdx.x;       // query head index
    int tid = threadIdx.x;
    int kv_h = h / (n_heads / n_kv_heads);  // which KV head
    int seq_len = pos + 1;
    int kv_dim = n_kv_heads * head_dim;

    float *scores = smem;  // [seq_len]

    const float *q_head = d_q + h * head_dim;
    float scale = rsqrtf((float)head_dim);

    // Phase 1: Compute Q @ K^T scores (FP16 K reads)
    for (int t = tid; t < seq_len; t += 256) {
        int64_t k_base = (int64_t)layer * ctx_len * kv_dim + (int64_t)t * kv_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_head[d] * __half2float(d_kv_k[k_base + d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores
    __shared__ float smax[256];
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += 256)
        local_max = fmaxf(local_max, scores[t]);
    smax[tid] = local_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }
    float max_val = smax[0];

    __shared__ float ssum[256];
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += 256) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    for (int t = tid; t < seq_len; t += 256)
        scores[t] *= inv_sum;
    __syncthreads();

    // Phase 3: Weighted V accumulation (FP16 V reads)
    for (int d = tid; d < head_dim; d += 256) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            int64_t v_base = (int64_t)layer * ctx_len * kv_dim + (int64_t)t * kv_dim + kv_h * head_dim;
            acc += scores[t] * __half2float(d_kv_v[v_base + d]);
        }
        d_attn_out[h * head_dim + d] = acc;
    }
}

// ============================================================
// KERNEL 15: Softmax (small N, e.g. 128 experts)
// Block: (256), Grid: (1)
// ============================================================

__global__ void kernel_softmax(float *x, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Max
    float local_max = -1e30f;
    for (int i = tid; i < n; i += 256)
        local_max = fmaxf(local_max, x[i]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += 256) {
        float e = expf(x[i] - max_val);
        x[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // Normalize
    for (int i = tid; i < n; i += 256)
        x[i] *= inv_sum;
}

// ============================================================
// KERNEL 16: Top-K selection + renormalization
// Single thread, k is small (8)
// ============================================================

__global__ void kernel_top_k_renorm(const float *x, int n, int k,
                                     int *out_ids, float *out_vals) {
    // Simple linear scan for top-k (k=8 is tiny)
    for (int j = 0; j < k; j++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int i = 0; i < n; i++) {
            // Check if already selected
            bool skip = false;
            for (int p = 0; p < j; p++) {
                if (out_ids[p] == i) { skip = true; break; }
            }
            if (!skip && x[i] > best) {
                best = x[i];
                best_idx = i;
            }
        }
        out_ids[j] = best_idx;
        out_vals[j] = best;
    }

    // Renormalize
    float sum = 0.0f;
    for (int j = 0; j < k; j++) sum += out_vals[j];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int j = 0; j < k; j++) out_vals[j] *= inv;
    }
}

// ============================================================
// KERNEL 16b: Parallel warp-based top-k + renormalize
// Grid: (1), Block: (32) — single warp
// Each lane handles n/32 elements with strided access
// ============================================================

__global__ void kernel_top_k_parallel(const float *x, int n, int k,
                                       int *out_ids, float *out_vals) {
    int lane = threadIdx.x;

    // Strided load into registers (coalesced access)
    float vals[16];
    int n_local = 0;
    for (int i = lane; i < n; i += 32)
        vals[n_local++] = x[i];

    for (int j = 0; j < k; j++) {
        // Each lane finds its local max
        float local_max = -1e30f;
        int local_best = -1;
        for (int p = 0; p < n_local; p++) {
            if (vals[p] > local_max) {
                local_max = vals[p];
                local_best = p;
            }
        }
        int local_idx = (local_best >= 0) ? (local_best * 32 + lane) : -1;

        // Warp max-reduce to find global max
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down(local_max, offset);
            int other_idx = __shfl_down(local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Lane 0 stores the winner
        if (lane == 0) {
            out_ids[j] = local_idx;
            out_vals[j] = local_max;
        }

        // Broadcast winner and mask it out
        int winner = __shfl(local_idx, 0);
        int winner_lane = winner % 32;
        int winner_slot = winner / 32;
        if (lane == winner_lane)
            vals[winner_slot] = -1e30f;
    }

    // Renormalize (lane 0 only)
    if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += out_vals[j];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int j = 0; j < k; j++) out_vals[j] *= inv;
        }
    }
}

// ============================================================
// KERNEL 17: Vector add (residual connection)
// ============================================================

__global__ void kernel_vec_add(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

// ============================================================
// KERNEL 17b: Fused residual add + RMS norm
// x[i] += residual[i], then output[i] = rms_norm(x) * weight
// Block: (256), Grid: (1) — single block reduction
// ============================================================

__global__ void kernel_residual_rms_norm(float *x, const float *residual,
                                          const float *weight, float *output,
                                          int n, float eps) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Phase 1: add residual + compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += 256) {
        float val = x[i] + residual[i];
        x[i] = val;
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)n + eps);

    // Phase 2: normalize and scale
    for (int i = tid; i < n; i += 256)
        output[i] = x[i] * rms * weight[i];
}

// ============================================================
// KERNEL 17c: Fused MoE output + residual
// out = weighted_sum(down, scores, n_experts) + shared + residual_x
// Block: (256), Grid: (ceil(n_embd/256))
// ============================================================

__global__ void kernel_moe_output_residual(float *x, const float *down,
                                            const float *scores,
                                            const float *shared_out,
                                            int n_embd, int n_experts,
                                            bool has_shared) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_experts; e++)
        acc += scores[e] * down[e * n_embd + i];
    if (has_shared) acc += shared_out[i];
    x[i] += acc;
}

// ============================================================
// KERNEL 18: Embedding lookup (Q4_K)
// Dequantize one row from quantized embedding table
// ============================================================

__global__ void kernel_embed_lookup_q4k(
    const void *embd_table, float *output,
    int token_id, int n_embd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n_embd / QK_K;  // blocks per row

    // Each thread dequantizes one Q4_K block
    if (tid >= nb) return;

    const block_q4_K *row = (const block_q4_K *)embd_table + (int64_t)token_id * nb;
    const block_q4_K *blk = &row[tid];

    float d    = device_fp16_to_fp32(blk->d);
    float dmin = device_fp16_to_fp32(blk->dmin);
    const uint8_t *qs = blk->qs;
    float *out = output + tid * QK_K;

    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        device_get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        device_get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;

        for (int l = 0; l < 32; ++l)
            out[j + l] = d1 * (qs[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l)
            out[j + 32 + l] = d2 * (qs[l] >> 4) - m2;
        qs += 32;
        is += 2;
    }
}

__global__ void kernel_embed_lookup_q8_0(
    const void *embd_table, float *output,
    int token_id, int n_embd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n_embd / QK8_0;  // blocks per row

    if (tid >= nb) return;

    const block_q8_0 *row = (const block_q8_0 *)embd_table + (int64_t)token_id * nb;
    const block_q8_0 *blk = &row[tid];

    float d = device_fp16_to_fp32(blk->d);
    float *out = output + tid * QK8_0;
    for (int i = 0; i < QK8_0; i++)
        out[i] = d * (float)blk->qs[i];
}

// ============================================================
// KERNEL 19: Bias add
// ============================================================

__global__ void kernel_bias_add(float *x, const float *bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += bias[i];
}

// ============================================================
// Helper: dispatch quantized single-matrix matvec by type
// ============================================================

static void dispatch_matvec(hipStream_t s, const void *weight, int type,
                             const float *input, float *output, int M, int K) {
    // Use cooperative kernels (32 threads/row) for all Q4_K/Q5_K/Q6_K — better memory coalescing
    if (M >= 8) {
        constexpr int RPB = 8;  // rows per block
        dim3 cblock(32, RPB);
        dim3 cgrid((M + RPB - 1) / RPB);
        if (type == GGML_TYPE_Q4_K_ID) {
            kernel_q4k_matvec_coop<<<cgrid, cblock, 0, s>>>(weight, input, output, M, K);
            return;
        } else if (type == GGML_TYPE_Q5_K_ID) {
            kernel_q5k_matvec_coop<<<cgrid, cblock, 0, s>>>(weight, input, output, M, K);
            return;
        } else if (type == GGML_TYPE_Q6_K_ID) {
            kernel_q6k_matvec_coop<<<cgrid, cblock, 0, s>>>(weight, input, output, M, K);
            return;
        } else if (type == GGML_TYPE_Q8_0_ID) {
            kernel_q8_0_matvec_coop<<<cgrid, cblock, 0, s>>>(weight, input, output, M, K);
            return;
        } else if (type == GGML_TYPE_F32_ID) {
            kernel_f32_matvec_coop<<<cgrid, cblock, K * sizeof(float), s>>>(
                (const float *)weight, input, output, M, K);
            return;
        }
    }
    dim3 block(256);
    dim3 grid((M + 255) / 256);
    if (type == GGML_TYPE_Q4_K_ID) {
        kernel_q4k_matvec_single<<<grid, block, 0, s>>>(weight, input, output, M, K);
    } else if (type == GGML_TYPE_Q5_K_ID) {
        kernel_q5k_matvec_single<<<grid, block, 0, s>>>(weight, input, output, M, K);
    } else if (type == GGML_TYPE_Q6_K_ID) {
        kernel_q6k_matvec_single<<<grid, block, 0, s>>>(weight, input, output, M, K);
    } else if (type == GGML_TYPE_Q8_0_ID) {
        kernel_q8_0_matvec_coop<<<(M + 7) / 8, dim3(32, 8), 0, s>>>(weight, input, output, M, K);
    } else if (type == GGML_TYPE_MXFP4_ID) {
        kernel_mxfp4_matvec_single<<<grid, block, 0, s>>>(weight, input, output, M, K);
    } else if (type == GGML_TYPE_F32_ID) {
        kernel_f32_matvec<<<grid, block, 0, s>>>(
            (const float *)weight, input, output, M, K);
    }
}

// Quantize a float input vector to Q8 format for dp4a matvecs.
// Call once, then use dispatch_matvec_q8 for all matvecs sharing this input.
static void quantize_input_q8(hipStream_t s, const float *input, int K,
                                int8_t *out_qs, float *out_scales) {
    int nblocks = K / 32;
    kernel_quantize_f32_q8<<<nblocks, 32, 0, s>>>(input, out_qs, out_scales, K);
}

// Dispatch Q8_0 matvec using pre-quantized Q8 input (dp4a path).
// For non-Q8_0 types, falls back to regular dispatch_matvec (needs float input).
static void dispatch_matvec_q8(hipStream_t s, const void *weight, int type,
                                 const float *input_f32,
                                 const int8_t *input_qs, const float *input_scales,
                                 float *output, int M, int K) {
    if (type == GGML_TYPE_Q8_0_ID && M >= 8) {
        constexpr int RPB = 8;
        dim3 block(32, RPB);
        dim3 grid((M + RPB - 1) / RPB);
        kernel_q8_0_matvec_dp4a<<<grid, block, 0, s>>>(
            weight, input_qs, input_scales, output, M, K);
    } else {
        dispatch_matvec(s, weight, type, input_f32, output, M, K);
    }
}

// ============================================================
// Async RAM cache population: background memcpy thread
// ============================================================

static void *rcache_pop_worker(void *arg) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)arg;

    while (true) {
        pthread_mutex_lock(&ctx->rcache_pop.mutex);
        while (!ctx->rcache_pop.work_ready && !ctx->rcache_pop.shutdown) {
            pthread_cond_wait(&ctx->rcache_pop.cond_work, &ctx->rcache_pop.mutex);
        }
        if (ctx->rcache_pop.shutdown) {
            pthread_mutex_unlock(&ctx->rcache_pop.mutex);
            break;
        }

        int n = ctx->rcache_pop.n;
        void *srcs[16];
        int slots[16];
        memcpy(srcs, ctx->rcache_pop.srcs, n * sizeof(void *));
        memcpy(slots, ctx->rcache_pop.slots, n * sizeof(int));
        void *cache_buf = ctx->rcache_pop.cache_buf;
        uint64_t stride = ctx->rcache_pop.stride;
        ctx->rcache_pop.work_ready = false;
        pthread_mutex_unlock(&ctx->rcache_pop.mutex);

        // Bulk memcpy: NVMe/prefetch buffers → pinned RAM cache slots
        for (int i = 0; i < n; i++) {
            memcpy((uint8_t *)cache_buf + (size_t)slots[i] * stride,
                   srcs[i], stride);
        }

        pthread_mutex_lock(&ctx->rcache_pop.mutex);
        ctx->rcache_pop.work_done = true;
        pthread_cond_signal(&ctx->rcache_pop.cond_done);
        pthread_mutex_unlock(&ctx->rcache_pop.mutex);
    }
    return NULL;
}

static void rcache_pop_start(gpu_ctx_t *ctx, void **srcs, int *slots, int n) {
    if (!ctx->rcache_pop.active || n <= 0) return;
    pthread_mutex_lock(&ctx->rcache_pop.mutex);
    ctx->rcache_pop.n = n;
    memcpy(ctx->rcache_pop.srcs, srcs, n * sizeof(void *));
    memcpy(ctx->rcache_pop.slots, slots, n * sizeof(int));
    ctx->rcache_pop.work_ready = true;
    ctx->rcache_pop.work_done = false;
    pthread_cond_signal(&ctx->rcache_pop.cond_work);
    pthread_mutex_unlock(&ctx->rcache_pop.mutex);
}

static void rcache_pop_wait(gpu_ctx_t *ctx) {
    if (!ctx->rcache_pop.active) return;
    pthread_mutex_lock(&ctx->rcache_pop.mutex);
    while (!ctx->rcache_pop.work_done) {
        pthread_cond_wait(&ctx->rcache_pop.cond_done, &ctx->rcache_pop.mutex);
    }
    pthread_mutex_unlock(&ctx->rcache_pop.mutex);
}

// ---- CAR backfill worker ----
// Loads CAR-substituted experts from NVMe into RAM cache during idle cycles.
// The worker only does NVMe reads; the main thread commits to RAM cache.

static void *backfill_worker_fn(void *arg) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)arg;
    auto &bf = ctx->backfill;

    while (true) {
        pthread_mutex_lock(&bf.mutex);
        while (!bf.work_ready && !bf.shutdown)
            pthread_cond_wait(&bf.cond_work, &bf.mutex);
        if (bf.shutdown) { pthread_mutex_unlock(&bf.mutex); break; }

        int n = bf.batch_n;
        int b_layer[16], b_eid[16];
        memcpy(b_layer, bf.batch_layer, n * sizeof(int));
        memcpy(b_eid, bf.batch_eid, n * sizeof(int));
        bf.work_ready = false;
        pthread_mutex_unlock(&bf.mutex);

        // Read experts from NVMe via blocking pread into our own buffers.
        // We avoid io_uring entirely to prevent contention with the main
        // pipeline's ring1/ring2 submissions.
        int completed = 0;
        for (int i = 0; i < n; i++) {
            // Re-check RAM cache (may have been loaded since enqueue)
            if (bf.rcache->map[b_layer[i] * bf.rcache->n_experts + b_eid[i]] >= 0) {
                bf.total_skipped++;
                continue;
            }
            int rc = nvme_io_pread_expert(bf.nvme_io, b_layer[i], b_eid[i],
                                           bf.buffers[completed]);
            if (rc == 0) {
                bf.batch_layer[completed] = b_layer[i];
                bf.batch_eid[completed] = b_eid[i];
                completed++;
            }
        }

        pthread_mutex_lock(&bf.mutex);
        bf.batch_n = completed;
        bf.work_done = true;
        pthread_cond_signal(&bf.cond_done);
        pthread_mutex_unlock(&bf.mutex);
    }
    return NULL;
}

// Dedup bitmap helpers (main thread only — no locking)
static inline bool backfill_dedup_test(gpu_ctx_t *ctx, int layer, int eid) {
    int idx = layer * ctx->backfill.dedup_n_experts + eid;
    return (ctx->backfill.dedup[idx >> 6] >> (idx & 63)) & 1;
}
static inline void backfill_dedup_set(gpu_ctx_t *ctx, int layer, int eid) {
    int idx = layer * ctx->backfill.dedup_n_experts + eid;
    ctx->backfill.dedup[idx >> 6] |= 1ULL << (idx & 63);
}
static inline void backfill_dedup_clear(gpu_ctx_t *ctx, int layer, int eid) {
    int idx = layer * ctx->backfill.dedup_n_experts + eid;
    ctx->backfill.dedup[idx >> 6] &= ~(1ULL << (idx & 63));
}

// Enqueue a substituted expert for background backfill (main thread only).
// Maintains a priority queue sorted by score_ratio ascending (worst subs first).
static void backfill_enqueue(gpu_ctx_t *ctx, int layer, int expert_id, float score_ratio) {
    auto &bf = ctx->backfill;
    if (!bf.active) return;
    // Skip if already in RAM cache
    if (bf.rcache->map[layer * bf.rcache->n_experts + expert_id] >= 0) return;
    // Skip if already queued (dedup)
    if (backfill_dedup_test(ctx, layer, expert_id)) {
        bf.total_deduped++;
        return;
    }
    // If queue full, only insert if worse than the current worst (last entry)
    if (bf.queue_count >= 512) {
        if (score_ratio >= bf.queue_ratio[bf.queue_count - 1]) {
            bf.total_dropped++;
            return;
        }
        // Evict the least-priority entry (last in sorted order)
        backfill_dedup_clear(ctx, bf.queue_layer[bf.queue_count - 1],
                             bf.queue_eid[bf.queue_count - 1]);
        bf.queue_count--;
    }
    // Insertion sort: find position for this ratio (ascending order)
    int pos = bf.queue_count;
    while (pos > 0 && bf.queue_ratio[pos - 1] > score_ratio) {
        bf.queue_layer[pos] = bf.queue_layer[pos - 1];
        bf.queue_eid[pos] = bf.queue_eid[pos - 1];
        bf.queue_ratio[pos] = bf.queue_ratio[pos - 1];
        pos--;
    }
    bf.queue_layer[pos] = layer;
    bf.queue_eid[pos] = expert_id;
    bf.queue_ratio[pos] = score_ratio;
    bf.queue_count++;
    bf.total_queued++;
    backfill_dedup_set(ctx, layer, expert_id);
}

// Dispatch a batch of backfill reads to the worker (main thread, non-blocking)
static void backfill_dispatch(gpu_ctx_t *ctx) {
    auto &bf = ctx->backfill;
    if (!bf.active || bf.queue_count == 0) return;

    pthread_mutex_lock(&bf.mutex);
    if (!bf.work_done) {
        // Worker still busy from last dispatch, skip this cycle
        pthread_mutex_unlock(&bf.mutex);
        return;
    }

    // Dequeue from front (lowest ratio = worst substitutions = highest priority)
    bf.batch_n = 0;
    int removed = 0;
    for (int i = 0; i < bf.queue_count && bf.batch_n < bf.batch_max; ) {
        int layer = bf.queue_layer[i];
        int eid = bf.queue_eid[i];
        backfill_dedup_clear(ctx, layer, eid);
        removed++;
        if (bf.rcache->map[layer * bf.rcache->n_experts + eid] >= 0) {
            bf.total_skipped++;
            i++;
            continue;
        }
        bf.batch_layer[bf.batch_n] = layer;
        bf.batch_eid[bf.batch_n] = eid;
        bf.batch_n++;
        i++;
    }
    // Compact: shift remaining entries down
    if (removed > 0 && removed < bf.queue_count) {
        memmove(bf.queue_layer, bf.queue_layer + removed, (bf.queue_count - removed) * sizeof(int));
        memmove(bf.queue_eid, bf.queue_eid + removed, (bf.queue_count - removed) * sizeof(int));
        memmove(bf.queue_ratio, bf.queue_ratio + removed, (bf.queue_count - removed) * sizeof(float));
    }
    bf.queue_count -= removed;

    if (bf.batch_n > 0) {
        bf.work_ready = true;
        bf.work_done = false;
        pthread_cond_signal(&bf.cond_work);
    }
    pthread_mutex_unlock(&bf.mutex);
}

// Commit completed backfill reads to RAM cache (main thread only — safe)
static void backfill_try_complete(gpu_ctx_t *ctx) {
    auto &bf = ctx->backfill;
    if (!bf.active) return;

    pthread_mutex_lock(&bf.mutex);
    bool done = bf.work_done;
    int n = bf.batch_n;
    pthread_mutex_unlock(&bf.mutex);
    if (!done || n == 0) return;

    for (int i = 0; i < n; i++) {
        int layer = bf.batch_layer[i];
        int eid = bf.batch_eid[i];
        // Skip if loaded by main pipeline since read started
        if (bf.rcache->map[layer * bf.rcache->n_experts + eid] >= 0) {
            bf.total_skipped++;
            continue;
        }
        int rslot = ram_cache_alloc_slot(bf.rcache, layer, eid);
        memcpy((uint8_t *)bf.rcache->buf + (size_t)rslot * bf.expert_stride,
               bf.buffers[i], bf.expert_stride);
        bf.total_backfilled++;
    }

    pthread_mutex_lock(&bf.mutex);
    bf.batch_n = 0;
    pthread_mutex_unlock(&bf.mutex);
}

// Deterministic backfill: dispatch exactly batch_max experts and block until done.
// Called once per token (after layer loop), ensures identical cache evolution
// regardless of NVMe bandwidth.
static void backfill_dispatch_sync(gpu_ctx_t *ctx) {
    auto &bf = ctx->backfill;
    if (!bf.active || !bf.sync_mode) return;

    // Wait for any in-flight work from previous call to complete
    pthread_mutex_lock(&bf.mutex);
    while (!bf.work_done && !bf.shutdown)
        pthread_cond_wait(&bf.cond_done, &bf.mutex);
    pthread_mutex_unlock(&bf.mutex);

    // Commit previous batch results to RAM cache
    backfill_try_complete(ctx);

    if (bf.queue_count == 0) return;

    // Dequeue up to batch_max entries
    bf.batch_n = 0;
    int removed = 0;
    for (int i = 0; i < bf.queue_count && bf.batch_n < bf.batch_max; ) {
        int layer = bf.queue_layer[i];
        int eid = bf.queue_eid[i];
        backfill_dedup_clear(ctx, layer, eid);
        removed++;
        if (bf.rcache->map[layer * bf.rcache->n_experts + eid] >= 0) {
            bf.total_skipped++;
            i++;
            continue;
        }
        bf.batch_layer[bf.batch_n] = layer;
        bf.batch_eid[bf.batch_n] = eid;
        bf.batch_n++;
        i++;
    }
    if (removed > 0 && removed < bf.queue_count) {
        memmove(bf.queue_layer, bf.queue_layer + removed, (bf.queue_count - removed) * sizeof(int));
        memmove(bf.queue_eid, bf.queue_eid + removed, (bf.queue_count - removed) * sizeof(int));
        memmove(bf.queue_ratio, bf.queue_ratio + removed, (bf.queue_count - removed) * sizeof(float));
    }
    bf.queue_count -= removed;

    if (bf.batch_n == 0) return;

    // Signal worker and block until all reads complete
    pthread_mutex_lock(&bf.mutex);
    bf.work_ready = true;
    bf.work_done = false;
    pthread_cond_signal(&bf.cond_work);
    while (!bf.work_done && !bf.shutdown)
        pthread_cond_wait(&bf.cond_done, &bf.mutex);
    pthread_mutex_unlock(&bf.mutex);

    // Commit results to RAM cache immediately
    backfill_try_complete(ctx);
}

// ---- Bounce buffer memcpy worker ----
// Copies scattered RAM cache slots to contiguous bounce buffer in background.
// Main thread can do hipEventSync/rcache_pop_wait while this runs.

static void *bounce_memcpy_worker(void *arg) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)arg;

    while (true) {
        pthread_mutex_lock(&ctx->bounce_worker.mutex);
        while (!ctx->bounce_worker.work_ready && !ctx->bounce_worker.shutdown) {
            pthread_cond_wait(&ctx->bounce_worker.cond_work, &ctx->bounce_worker.mutex);
        }
        if (ctx->bounce_worker.shutdown) {
            pthread_mutex_unlock(&ctx->bounce_worker.mutex);
            break;
        }

        int n = ctx->bounce_worker.n;
        int slots[16];
        memcpy(slots, ctx->bounce_worker.slots, n * sizeof(int));
        void *bounce = ctx->bounce_worker.bounce;
        void *cache = ctx->bounce_worker.cache;
        uint64_t stride = ctx->bounce_worker.stride;
        ctx->bounce_worker.work_ready = false;
        pthread_mutex_unlock(&ctx->bounce_worker.mutex);

        for (int i = 0; i < n; i++) {
            memcpy((uint8_t *)bounce + (size_t)i * stride,
                   (uint8_t *)cache + (size_t)slots[i] * stride,
                   stride);
        }

        pthread_mutex_lock(&ctx->bounce_worker.mutex);
        ctx->bounce_worker.work_done = true;
        pthread_cond_signal(&ctx->bounce_worker.cond_done);
        pthread_mutex_unlock(&ctx->bounce_worker.mutex);
    }
    return NULL;
}

static void bounce_worker_start(gpu_ctx_t *ctx, const int *slots, int n) {
    if (!ctx->bounce_worker.active || n <= 0) return;
    pthread_mutex_lock(&ctx->bounce_worker.mutex);
    ctx->bounce_worker.n = n;
    memcpy(ctx->bounce_worker.slots, slots, n * sizeof(int));
    ctx->bounce_worker.work_done = false;
    ctx->bounce_worker.work_ready = true;
    pthread_cond_signal(&ctx->bounce_worker.cond_work);
    pthread_mutex_unlock(&ctx->bounce_worker.mutex);
}

static void bounce_worker_wait(gpu_ctx_t *ctx) {
    if (!ctx->bounce_worker.active) return;
    pthread_mutex_lock(&ctx->bounce_worker.mutex);
    while (!ctx->bounce_worker.work_done) {
        pthread_cond_wait(&ctx->bounce_worker.cond_done, &ctx->bounce_worker.mutex);
    }
    pthread_mutex_unlock(&ctx->bounce_worker.mutex);
}

// ---- NVMe background loader ----
// Runs nvme_io_load_experts_at on a dedicated thread so the main thread
// can do RAM cache memcpy + hipMemcpyAsync in parallel with NVMe I/O.
// With H2D callback: each expert's H2D is queued on the transfer stream
// immediately when its NVMe read completes, eliminating the idle gap.

// Callback fired from drive I/O threads — queues H2D per expert
static void nvme_h2d_callback(int buffer_idx, void *buffer, void *user_data) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)user_data;
    size_t stride = ctx->nvme_bg.stride;
    int pos = ctx->nvme_bg.nv_start + buffer_idx;
    uint8_t *dst = ctx->nvme_bg.d_staging + ctx->nvme_bg.stg_base + (size_t)pos * stride;
    (void)hipMemcpyAsync(dst, buffer, stride, hipMemcpyHostToDevice, ctx->nvme_bg.h2d_stream);
}

static void *nvme_bg_worker(void *arg) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)arg;

    while (true) {
        pthread_mutex_lock(&ctx->nvme_bg.mutex);
        while (!ctx->nvme_bg.work_ready && !ctx->nvme_bg.shutdown) {
            pthread_cond_wait(&ctx->nvme_bg.cond_work, &ctx->nvme_bg.mutex);
        }
        if (ctx->nvme_bg.shutdown) {
            pthread_mutex_unlock(&ctx->nvme_bg.mutex);
            break;
        }

        // Snapshot work params
        nvme_io_t *io = ctx->nvme_bg.io;
        int layer = ctx->nvme_bg.layer;
        int expert_ids[16];
        int n = ctx->nvme_bg.n_experts;
        int buf_off = ctx->nvme_bg.buffer_offset;
        bool h2d = ctx->nvme_bg.h2d_enabled;
        memcpy(expert_ids, ctx->nvme_bg.expert_ids, n * sizeof(int));
        ctx->nvme_bg.work_ready = false;
        pthread_mutex_unlock(&ctx->nvme_bg.mutex);

        // Blocking NVMe load with optional per-expert H2D callback
        void *out_bufs[16];
        int rc;
        if (h2d) {
            rc = nvme_io_load_experts_at_cb(io, layer, expert_ids, n, buf_off, out_bufs,
                                             nvme_h2d_callback, ctx);
        } else {
            rc = nvme_io_load_experts_at(io, layer, expert_ids, n, buf_off, out_bufs);
        }

        pthread_mutex_lock(&ctx->nvme_bg.mutex);
        ctx->nvme_bg.result = rc;
        memcpy(ctx->nvme_bg.out_buffers, out_bufs, n * sizeof(void *));
        ctx->nvme_bg.work_done = true;
        pthread_cond_signal(&ctx->nvme_bg.cond_done);
        pthread_mutex_unlock(&ctx->nvme_bg.mutex);
    }
    return NULL;
}

static void nvme_bg_start(gpu_ctx_t *ctx, nvme_io_t *io, int layer,
                           const int *ids, int n, int buf_off) {
    if (!ctx->nvme_bg.active || n <= 0) return;
    pthread_mutex_lock(&ctx->nvme_bg.mutex);
    ctx->nvme_bg.io = io;
    ctx->nvme_bg.layer = layer;
    ctx->nvme_bg.n_experts = n;
    ctx->nvme_bg.buffer_offset = buf_off;
    memcpy(ctx->nvme_bg.expert_ids, ids, n * sizeof(int));
    ctx->nvme_bg.work_done = false;
    ctx->nvme_bg.work_ready = true;
    pthread_cond_signal(&ctx->nvme_bg.cond_work);
    pthread_mutex_unlock(&ctx->nvme_bg.mutex);
}

static void nvme_bg_wait(gpu_ctx_t *ctx) {
    if (!ctx->nvme_bg.active) return;
    pthread_mutex_lock(&ctx->nvme_bg.mutex);
    while (!ctx->nvme_bg.work_done) {
        pthread_cond_wait(&ctx->nvme_bg.cond_done, &ctx->nvme_bg.mutex);
    }
    pthread_mutex_unlock(&ctx->nvme_bg.mutex);
}

// ============================================================
// DeltaNet/SSM GPU Kernels
// ============================================================

// Conv1d + SiLU: causal conv1d on QKV using circular buffer, then SiLU
// Grid: (ceil(qkv_dim/256)), Block: (256)
__global__ void kernel_ssm_conv1d_silu(
    float *qkv,           // [qkv_dim] input (current step), modified in-place
    float *conv_buf,       // [(conv_k-1) * qkv_dim] circular buffer
    const float *conv_w,   // [qkv_dim * conv_k] weights (channel-major: w[ch*conv_k + t])
    int conv_pos,          // current position in circular buffer
    int qkv_dim,           // 12288
    int conv_k             // 4
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qkv_dim) return;

    // Conv1d: newest input uses weight[conv_k-1], past inputs use earlier weights
    float val = conv_w[j * conv_k + (conv_k - 1)] * qkv[j];
    for (int t = 1; t < conv_k; t++) {
        int past_idx = conv_pos - t;
        if (past_idx < 0) continue;
        past_idx = past_idx % (conv_k - 1);
        float past_val = conv_buf[past_idx * qkv_dim + j];
        int w_idx = conv_k - 1 - t;
        val += conv_w[j * conv_k + w_idx] * past_val;
    }

    // Store current input in circular buffer BEFORE overwriting qkv
    // (we need to save the original qkv[j], so store it first)
    int buf_pos = conv_pos % (conv_k - 1);
    conv_buf[buf_pos * qkv_dim + j] = qkv[j];

    // SiLU activation
    val = val / (1.0f + expf(-val));
    qkv[j] = val;
}

// L2-normalize Q and K per group
// Grid: (ng), Block: (128) — one block per group, sd=128 elements
__global__ void kernel_l2_norm_groups(float *Q, float *K, int sd) {
    __shared__ float sq[128], sk[128];
    int g = blockIdx.x;
    int tid = threadIdx.x;
    float *q_g = Q + g * sd;
    float *k_g = K + g * sd;

    // Load and compute squares
    float qv = (tid < sd) ? q_g[tid] : 0.0f;
    float kv = (tid < sd) ? k_g[tid] : 0.0f;
    sq[tid] = qv * qv;
    sk[tid] = kv * kv;
    __syncthreads();

    // Parallel reduction
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            sq[tid] += sq[tid + s];
            sk[tid] += sk[tid + s];
        }
        __syncthreads();
    }

    if (tid < sd) {
        float q_scale = rsqrtf(sq[0] + 1e-6f);
        float k_scale = rsqrtf(sk[0] + 1e-6f);
        q_g[tid] = qv * q_scale;
        k_g[tid] = kv * k_scale;
    }
}

// Compute decay and beta values from alpha/beta projections
// Grid: (1), Block: (64 or nh)
__global__ void kernel_decay_beta(
    float *dt,         // [nh] alpha projection output (modified in-place → decay)
    float *beta,       // [nh] beta projection output (modified in-place → sigmoid)
    const float *ssm_a,      // [nh] base decay values
    const float *dt_bias,    // [nh] bias (may be NULL → passed as separate flag)
    int nh,
    bool has_bias
) {
    int i = threadIdx.x;
    if (i >= nh) return;

    float d = dt[i];
    if (has_bias) d += dt_bias[i];

    // softplus
    d = logf(1.0f + expf(d));
    // decay = exp(A * softplus(alpha + bias))
    float A = ssm_a[i];
    dt[i] = expf(A * d);

    // beta = sigmoid
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));
}

// DeltaNet state update + output — the core kernel
// Grid: (nh=64), Block: (128 = vd)
// Each block handles one head's [sd, vd] = [128, 128] state
__global__ void kernel_deltanet_state_update(
    float *state,       // [nh * sd * vd] contiguous state
    const float *Q,     // [ng * sd] L2-normalized query (grouped)
    const float *K,     // [ng * sd] L2-normalized key (grouped)
    const float *V,     // [inner] value
    const float *decay, // [nh] per-head decay
    const float *beta,  // [nh] per-head beta
    float *y,           // [inner] output
    int sd,             // state dim (128)
    int vd,             // value dim (128)
    int ng,             // number of K groups (16)
    float q_scale       // 1/sqrt(sd)
) {
    int head = blockIdx.x;
    int vi = threadIdx.x;  // value dimension index [0, vd)
    if (vi >= vd) return;

    // Load K and Q into shared memory (all threads read same values)
    __shared__ float s_k[128], s_q[128];
    int g = head % ng;
    if (vi < sd) {
        s_k[vi] = K[g * sd + vi];
        s_q[vi] = Q[g * sd + vi];
    }
    __syncthreads();

    float *S = state + (size_t)head * sd * vd;  // [sd, vd]
    float d = decay[head];
    float b = beta[head];
    float v_val = V[head * vd + vi];

    // Pass 1: decay + compute kv_mem = k^T @ S[:, vi]
    float kv_mem = 0.0f;
    for (int si = 0; si < sd; si++) {
        S[si * vd + vi] *= d;
        kv_mem += s_k[si] * S[si * vd + vi];
    }

    // Delta = beta * (v - kv_mem)
    float delta = b * (v_val - kv_mem);

    // Pass 2 (fused): outer product update + output query
    float out = 0.0f;
    for (int si = 0; si < sd; si++) {
        S[si * vd + vi] += s_k[si] * delta;
        out += S[si * vd + vi] * s_q[si];
    }
    y[head * vd + vi] = out * q_scale;
}

// Per-head RMS norm for SSM output (uses ssm_norm weights, no +1 bias)
// Grid: (nh), Block: (128 = vd)
__global__ void kernel_ssm_rms_norm(float *data, const float *weight,
                                     int vd, float eps) {
    __shared__ float sdata[128];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    float *hd = data + head * vd;

    float val = (tid < vd) ? hd[tid] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)vd + eps);
    if (tid < vd)
        hd[tid] = val * rms * weight[tid];
}

// SiLU gate: y[i] *= SiLU(gate[i]) where SiLU(x) = x * sigmoid(x)
// Grid: (ceil(n/256)), Block: (256)
__global__ void kernel_silu_gate(float *y, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = gate[i];
        float g = z / (1.0f + expf(-z));  // SiLU
        y[i] *= g;
    }
}

// Dot product + sigmoid gate scalar: buf[i] *= sigmoid(dot(h, gate_w))
// Grid: (1), Block: (256) — single-block reduction for dot product
__global__ void kernel_dot_sigmoid_scale(float *buf, const float *h,
                                          const float *gate_w, int n, int buf_n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Compute dot product in parallel
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += h[i] * gate_w[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 has the dot product → sigmoid
    __shared__ float gate_scalar;
    if (tid == 0) {
        gate_scalar = 1.0f / (1.0f + expf(-sdata[0]));
    }
    __syncthreads();

    // Scale buffer
    for (int i = tid; i < buf_n; i += blockDim.x) {
        buf[i] *= gate_scalar;
    }
}

// GPU DeltaNet SSM forward pass for a single layer
static void gpu_deltanet_ssm(gpu_ctx_t *ctx, struct gpu_layer *gl, int il,
                               int ssm_idx, hipStream_t s) {
    int n_embd = ctx->n_embd;
    int qkv_dim = ctx->ssm_qkv_dim;
    int inner = ctx->ssm_inner_size;
    int sd = ctx->ssm_state_size;
    int ng = ctx->ssm_group_count;
    int nh = ctx->ssm_time_step_rank;
    int vd = ctx->ssm_value_dim;
    int conv_k = ctx->ssm_conv_kernel;
    int kd = ng * sd;  // 2048

    // Quantize d_norm once for all Q8_0 matvecs sharing this input
    quantize_input_q8(s, ctx->d_norm, n_embd,
                      ctx->d_input_q8, ctx->d_input_q8_scales);

    // Step 1: QKV projection
    dispatch_matvec_q8(s, gl->d_attn_qkv, gl->attn_qkv_type, ctx->d_norm,
                       ctx->d_input_q8, ctx->d_input_q8_scales,
                       ctx->d_q, qkv_dim, n_embd);

    // Step 2: Conv1d + SiLU
    // SSM state offsets for this layer (ssm_idx is the index among SSM layers)
    int conv_steps = conv_k - 1;
    float *conv_buf = ctx->d_ssm_conv_bufs +
                       (size_t)ssm_idx * conv_steps * qkv_dim;

    // We need conv_pos from host — use the cached value
    int conv_pos = ctx->h_ssm_conv_pos[ssm_idx];

    kernel_ssm_conv1d_silu<<<(qkv_dim + 255) / 256, 256, 0, s>>>(
        ctx->d_q, conv_buf, gl->d_ssm_conv1d, conv_pos, qkv_dim, conv_k);

    // Update conv_pos
    ctx->h_ssm_conv_pos[ssm_idx] = conv_pos + 1;

    // Step 3: Split QKV and L2-normalize Q, K
    // Layout: [Q(kd), K(kd), V(inner)] contiguous in d_q
    float *d_Q = ctx->d_q;            // [kd = 2048]
    float *d_K = ctx->d_q + kd;       // [kd = 2048]
    float *d_V = ctx->d_q + kd + kd;  // [inner = 8192]

    kernel_l2_norm_groups<<<ng, 128, 0, s>>>(d_Q, d_K, sd);

    // Step 4: Decay and beta (reuse quantized d_norm)
    dispatch_matvec_q8(s, gl->d_ssm_alpha, gl->ssm_alpha_type, ctx->d_norm,
                       ctx->d_input_q8, ctx->d_input_q8_scales,
                       ctx->d_ssm_dt, nh, n_embd);
    dispatch_matvec_q8(s, gl->d_ssm_beta, gl->ssm_beta_type, ctx->d_norm,
                       ctx->d_input_q8, ctx->d_input_q8_scales,
                       ctx->d_ssm_beta_buf, nh, n_embd);

    bool has_bias = (gl->d_ssm_dt_bias != nullptr);
    kernel_decay_beta<<<1, nh, 0, s>>>(
        ctx->d_ssm_dt, ctx->d_ssm_beta_buf,
        gl->d_ssm_a, gl->d_ssm_dt_bias, nh, has_bias);

    // Step 5: State update + output
    size_t state_per_layer = (size_t)nh * sd * vd;
    float *d_state = ctx->d_ssm_states + (size_t)ssm_idx * state_per_layer;
    float q_scale = 1.0f / sqrtf((float)sd);

    kernel_deltanet_state_update<<<nh, vd, 0, s>>>(
        d_state, d_Q, d_K, d_V,
        ctx->d_ssm_dt, ctx->d_ssm_beta_buf,
        ctx->d_attn_out, sd, vd, ng, q_scale);

    // Step 6: Per-head RMS norm
    kernel_ssm_rms_norm<<<nh, vd, 0, s>>>(
        ctx->d_attn_out, gl->d_ssm_norm, vd, ctx->rms_norm_eps);

    // Step 7: Output gate (SiLU) — reuse quantized d_norm
    dispatch_matvec_q8(s, gl->d_attn_gate, gl->attn_gate_type, ctx->d_norm,
                       ctx->d_input_q8, ctx->d_input_q8_scales,
                       ctx->d_attn_gate_buf, inner, n_embd);
    kernel_silu_gate<<<(inner + 255) / 256, 256, 0, s>>>(
        ctx->d_attn_out, ctx->d_attn_gate_buf, inner);

    // Step 8: Output projection [inner] → [n_embd]
    // Quantize d_attn_out for this matvec (different input vector)
    quantize_input_q8(s, ctx->d_attn_out, inner,
                      ctx->d_input_q8, ctx->d_input_q8_scales);
    dispatch_matvec_q8(s, gl->d_ssm_out, gl->ssm_out_type, ctx->d_attn_out,
                       ctx->d_input_q8, ctx->d_input_q8_scales,
                       ctx->d_ffn_buf, n_embd, inner);
}

// ============================================================
// PUBLIC API: gpu_init
// ============================================================

extern "C" gpu_ctx_t *gpu_init(void) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        fprintf(stderr, "GPU: no HIP devices found\n");
        return nullptr;
    }

    err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "GPU: failed to set device 0: %s\n", hipGetErrorString(err));
        return nullptr;
    }

    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, 0);
    fprintf(stderr, "GPU: %s (%zu MB VRAM)\n",
            props.name, props.totalGlobalMem / (1024 * 1024));
#ifdef QMOE_CUDA
    fprintf(stderr, "GPU: sm_%d%d, SMs=%d, clock=%d MHz\n",
            props.major, props.minor, props.multiProcessorCount, props.clockRate / 1000);
#else
    fprintf(stderr, "GPU: arch=%s, CUs=%d, clock=%d MHz\n",
            props.gcnArchName, props.multiProcessorCount, props.clockRate / 1000);
#endif

    gpu_ctx_t *ctx = (gpu_ctx_t *)calloc(1, sizeof(gpu_ctx_t));
    if (!ctx) return nullptr;

    HIP_CHECK(hipStreamCreate(&ctx->stream));
    HIP_CHECK(hipStreamCreate(&ctx->transfer_stream));
    HIP_CHECK(hipStreamCreate(&ctx->shared_stream));
    HIP_CHECK(hipEventCreate(&ctx->batch_event[0]));
    HIP_CHECK(hipEventCreate(&ctx->batch_event[1]));
    HIP_CHECK(hipEventCreate(&ctx->shared_event));

    // Allocate expert FFN intermediate buffers (small, use MAX constants)
    // NOTE: d_expert_staging allocated in gpu_upload_model with actual expert_stride
    ctx->d_expert_staging = nullptr;
    HIP_CHECK(hipMalloc(&ctx->d_expert_ptrs, MAX_EXPERTS * sizeof(void *)));
    HIP_CHECK(hipMalloc(&ctx->d_gate, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx->d_up, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx->d_down, (size_t)MAX_EXPERTS * MAX_N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx->d_scores, MAX_EXPERTS * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx->d_ffn_out, MAX_N_EMBD * sizeof(float)));

    ctx->ram_cache_mb = -1;  // auto by default
    ctx->vram_cache_mb = -1; // auto by default
    ctx->n_devices = 1;

    fprintf(stderr, "GPU: expert FFN buffers allocated (GPU 0)\n");

    // ---- GPU1 init (expert engine) ----
    if (device_count >= 2) {
        hipDeviceProp_t props1;
        (void)hipGetDeviceProperties(&props1, 1);

        err = hipSetDevice(1);
        if (err != hipSuccess) {
            fprintf(stderr, "GPU1: failed to set device 1, running single-GPU\n");
        } else {
#ifdef QMOE_CUDA
            fprintf(stderr, "GPU1: %s (%zu MB VRAM), sm_%d%d, SMs=%d\n",
                    props1.name, props1.totalGlobalMem / (1024 * 1024),
                    props1.major, props1.minor, props1.multiProcessorCount);
#else
            fprintf(stderr, "GPU1: %s (%zu MB VRAM), arch=%s, CUs=%d\n",
                    props1.name, props1.totalGlobalMem / (1024 * 1024),
                    props1.gcnArchName, props1.multiProcessorCount);
#endif
            ctx->gpu1.device_id = 1;
            HIP_CHECK(hipStreamCreate(&ctx->gpu1.stream));
            HIP_CHECK(hipStreamCreate(&ctx->gpu1.transfer_stream));
            HIP_CHECK(hipEventCreate(&ctx->gpu1.batch_event[0]));
            HIP_CHECK(hipEventCreate(&ctx->gpu1.batch_event[1]));
            HIP_CHECK(hipEventCreate(&ctx->gpu1.cache_event));

            HIP_CHECK(hipMalloc(&ctx->gpu1.d_expert_ptrs, MAX_EXPERTS * sizeof(void *)));
            HIP_CHECK(hipMalloc(&ctx->gpu1.d_gate, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
            HIP_CHECK(hipMalloc(&ctx->gpu1.d_up, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
            HIP_CHECK(hipMalloc(&ctx->gpu1.d_down, (size_t)MAX_EXPERTS * MAX_N_EMBD * sizeof(float)));
            HIP_CHECK(hipMalloc(&ctx->gpu1.d_scores, MAX_EXPERTS * sizeof(float)));
            HIP_CHECK(hipMalloc(&ctx->gpu1.d_ffn_out, MAX_N_EMBD * sizeof(float)));
            ctx->gpu1.d_expert_staging = nullptr;  // allocated in gpu_upload_model
            ctx->gpu1.d_norm = nullptr;
            ctx->gpu1.d_partial = nullptr;
            memset(&ctx->gpu1.ecache, 0, sizeof(ctx->gpu1.ecache));

            ctx->n_devices = 2;
            fprintf(stderr, "GPU1: expert FFN buffers allocated\n");

            // ---- Ping-pong init: per-device expert FFN buffers ----
            // Check QMOE_PINGPONG env var to enable ping-pong mode
            ctx->pingpong = (getenv("QMOE_PINGPONG") != NULL);
            { const char *d = getenv("QMOE_DEBUG"); ctx->debug_level = d ? atoi(d) : 0; }
            if (ctx->pingpong) {
                fprintf(stderr, "GPU: ping-pong mode enabled (even layers GPU0, odd layers GPU1)\n");

                for (int d = 0; d < 2; d++) {
                    (void)hipSetDevice(d);
                    gpu_dev_t *dc = &ctx->dev[d];
                    memset(dc, 0, sizeof(gpu_dev_t));
                    dc->device_id = d;
                    HIP_CHECK(hipStreamCreate(&dc->stream));
                    HIP_CHECK(hipStreamCreate(&dc->transfer_stream));
                    HIP_CHECK(hipEventCreate(&dc->batch_event[0]));
                    HIP_CHECK(hipEventCreate(&dc->batch_event[1]));
                    HIP_CHECK(hipEventCreate(&dc->spec_event));
                    HIP_CHECK(hipEventCreate(&dc->prefetch_event));
                    dc->prefetch_pending = false;

                    // Expert FFN buffers
                    dc->d_expert_staging = nullptr;
                    HIP_CHECK(hipMalloc(&dc->d_expert_ptrs, MAX_EXPERTS * sizeof(void *)));
                    HIP_CHECK(hipMalloc(&dc->d_gate, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
                    HIP_CHECK(hipMalloc(&dc->d_up, (size_t)MAX_EXPERTS * MAX_INTERMEDIATE * sizeof(float)));
                    HIP_CHECK(hipMalloc(&dc->d_down, (size_t)MAX_EXPERTS * MAX_N_EMBD * sizeof(float)));
                    HIP_CHECK(hipMalloc(&dc->d_scores, MAX_EXPERTS * sizeof(float)));
                    HIP_CHECK(hipMalloc(&dc->d_ffn_out, MAX_N_EMBD * sizeof(float)));

                    // Profiling events
                    for (int i = 0; i < 12; i++) {
                        HIP_CHECK(hipEventCreate(&dc->prof_e[i]));
                    }

                    fprintf(stderr, "GPU: ping-pong dev[%d] init done\n", d);
                }
            }

            // Return to GPU0 as default device
            (void)hipSetDevice(0);
        }
    }

    return ctx;
}

// ============================================================
// PUBLIC API: gpu_set_ram_cache_mb
// ============================================================

extern "C" void gpu_set_ram_cache_mb(gpu_ctx_t *ctx, int mb) {
    if (ctx) ctx->ram_cache_mb = mb;
}

// ============================================================
// PUBLIC API: gpu_set_vram_cache_mb
// ============================================================

extern "C" void gpu_set_vram_cache_mb(gpu_ctx_t *ctx, int mb) {
    if (ctx) ctx->vram_cache_mb = mb;
}

// ============================================================
// PUBLIC API: gpu_upload_model
// Upload all non-expert weights + allocate KV cache + scratch
// ============================================================

extern "C" int gpu_upload_model(gpu_ctx_t *ctx, const void *model_ptr) {
    if (!ctx || !model_ptr) return -1;

    const model_t *m = (const model_t *)model_ptr;
    const model_hparams_t *hp = &m->hparams;

    // Cache params
    ctx->n_embd = hp->n_embd;
    ctx->n_heads = hp->n_heads;
    ctx->n_kv_heads = hp->n_kv_heads;
    ctx->head_dim = hp->head_dim;
    ctx->n_experts_total = hp->n_experts;
    ctx->n_experts_used = hp->n_experts_used;
    ctx->expert_intermediate = hp->expert_intermediate;
    ctx->shared_expert_intermediate = hp->shared_expert_intermediate;
    ctx->vocab_size = hp->vocab_size;
    ctx->ctx_len = (hp->ctx_len > 4096) ? 4096 : hp->ctx_len;
    ctx->rope_dim = hp->rope_dim;
    ctx->rope_freq_base = hp->rope_freq_base;
    ctx->rms_norm_eps = hp->rms_norm_eps;
    ctx->expert_gate_size = hp->expert_gate_size;
    ctx->expert_up_size = hp->expert_up_size;
    ctx->expert_stride = hp->expert_stride;
    ctx->gate_type = hp->gate_type;
    ctx->up_type = hp->up_type;
    ctx->n_layers = hp->n_layers;
    ctx->ssm_inner_size = hp->ssm_inner_size;
    ctx->ssm_state_size = hp->ssm_state_size;
    ctx->ssm_group_count = hp->ssm_group_count;
    ctx->ssm_time_step_rank = hp->ssm_time_step_rank;
    ctx->ssm_conv_kernel = hp->ssm_conv_kernel;
    ctx->ssm_value_dim = hp->ssm_value_dim;
    ctx->ssm_qkv_dim = hp->ssm_qkv_dim;
    ctx->full_q_dim = hp->full_q_dim;

    // --- Allocate expert VRAM cache FIRST (before weight uploads) ---
    // Must be the first large allocation to get a contiguous block before
    // smaller weight uploads fragment the VRAM heap.
    // Skip in ping-pong mode: per-device caches allocated later.
    if (!ctx->pingpong) {
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);

        size_t cache_budget = 0;
        if (ctx->vram_cache_mb == 0) {
            fprintf(stderr, "GPU: VRAM cache: disabled by config\n");
        } else if (ctx->vram_cache_mb > 0) {
            cache_budget = (size_t)ctx->vram_cache_mb * 1024ULL * 1024ULL;
            fprintf(stderr, "GPU: VRAM cache: configured %d MB\n", ctx->vram_cache_mb);
        } else {
            // Reserve VRAM for weights+KV+SSM+staging+headroom, rest goes to cache
            // Estimate weight usage (measured: ~8.8 GB for 397B, ~5.5 GB for 122B)
            size_t weight_estimate = (size_t)hp->n_layers * 170ULL * 1024 * 1024;  // ~170 MB/layer
            size_t kv_estimate = (size_t)hp->n_layers * hp->ctx_len * hp->n_kv_heads *
                                 hp->head_dim * sizeof(__half) * 2;
            if (hp->ctx_len > 4096) kv_estimate = kv_estimate * 4096 / hp->ctx_len;
            size_t ssm_estimate = (size_t)(hp->n_layers * 3 / 4) *
                                  (hp->ssm_inner_size / hp->ssm_group_count * hp->ssm_state_size * sizeof(float) +
                                   hp->ssm_inner_size * hp->ssm_conv_kernel * sizeof(float));
            size_t staging_estimate = (size_t)hp->n_experts_used * 2 * hp->expert_stride;
            size_t overhead = 512ULL * 1024 * 1024;  // scratch buffers + headroom
            size_t reserved = weight_estimate + kv_estimate + ssm_estimate + staging_estimate + overhead;
            if (free_mem > reserved)
                cache_budget = free_mem - reserved;
            fprintf(stderr, "GPU: VRAM budget: %.0f MB free, %.0f MB reserved for weights/KV/SSM, %.0f MB for cache\n",
                    (double)free_mem / (1024.0 * 1024.0),
                    (double)reserved / (1024.0 * 1024.0),
                    (double)cache_budget / (1024.0 * 1024.0));
        }

        int max_slots = (int)(cache_budget / ctx->expert_stride);
        if (max_slots < 0) max_slots = 0;
        int total_experts = ctx->n_layers * ctx->n_experts_total;
        if (max_slots > total_experts) max_slots = total_experts;

        memset(&ctx->ecache, 0, sizeof(ctx->ecache));
        if (max_slots > 0 && ctx->n_layers > 0) {
            if (vram_cache_init(&ctx->ecache, ctx->n_layers, ctx->n_experts_total,
                                max_slots, ctx->expert_stride) == 0) {
                max_slots = ctx->ecache.max_slots;
                size_t alloc_sz = (size_t)max_slots * ctx->expert_stride;
                hipError_t err = hipMalloc(&ctx->ecache.d_buf, alloc_sz);
                if (err != hipSuccess) {
                    // Try progressively smaller if fragmented
                    while (max_slots >= ctx->n_layers) {
                        max_slots = (max_slots * 3 / 4);
                        max_slots -= max_slots % ctx->n_layers;
                        vram_cache_free(&ctx->ecache);
                        if (max_slots >= ctx->n_layers &&
                            vram_cache_init(&ctx->ecache, ctx->n_layers, ctx->n_experts_total,
                                            max_slots, ctx->expert_stride) == 0) {
                            max_slots = ctx->ecache.max_slots;
                            alloc_sz = (size_t)max_slots * ctx->expert_stride;
                            err = hipMalloc(&ctx->ecache.d_buf, alloc_sz);
                            if (err == hipSuccess) break;
                        }
                    }
                }
                if (err != hipSuccess) {
                    fprintf(stderr, "GPU: expert cache alloc failed, disabling\n");
                    vram_cache_free(&ctx->ecache);
                } else {
                    fprintf(stderr, "GPU: expert cache: %d slots (%d/layer, %.1f MB), %.1f%% of %d total experts\n",
                            max_slots, ctx->ecache.slots_per_layer,
                            (double)alloc_sz / (1024.0 * 1024.0),
                            100.0 * max_slots / total_experts, total_experts);
                }
            }
        }
        if (ctx->ecache.max_slots == 0)
            fprintf(stderr, "GPU: expert cache: disabled (insufficient VRAM)\n");
    }

    fprintf(stderr, "GPU: uploading model weights to VRAM...\n");
    fprintf(stderr, "GPU:   %u layers, embd=%u, heads=%u/%u, vocab=%u\n",
            hp->n_layers, hp->n_embd, hp->n_heads, hp->n_kv_heads, hp->vocab_size);

    size_t total_uploaded = 0;

    // --- Upload global weights ---

    // Token embedding — keep on CPU to preserve VRAM for expert cache
    {
        ctx->token_embd_type = m->token_embd_type;
        ctx->d_token_embd = nullptr;
    }

    // Output norm + LM head: skip for pingpong (per-device alloc handles it later)
    if (!ctx->pingpong) {
        // Output norm
        {
            size_t sz = hp->n_embd * sizeof(float);
            HIP_CHECK_I(hipMalloc(&ctx->d_output_norm, sz));
            HIP_CHECK_I(hipMemcpy(ctx->d_output_norm, m->output_norm, sz, hipMemcpyHostToDevice));
            total_uploaded += sz;
        }

        // LM head (output weight)
        {
            // output shape: [n_embd, vocab_size] => vocab_size rows of n_embd
            size_t row_blocks = hp->n_embd / QK_K;
            ctx->output_type = m->output_type;
            if (m->output_type == GGML_TYPE_Q4_K) {
                ctx->output_row_bytes = row_blocks * sizeof(block_q4_K);
            } else if (m->output_type == GGML_TYPE_Q6_K) {
                ctx->output_row_bytes = row_blocks * sizeof(block_q6_K);
            } else {
                ctx->output_row_bytes = hp->n_embd * sizeof(float);
            }
            size_t total_bytes = (size_t)hp->vocab_size * ctx->output_row_bytes;
            HIP_CHECK_I(hipMalloc(&ctx->d_output, total_bytes));
            HIP_CHECK_I(hipMemcpy(ctx->d_output, m->output, total_bytes, hipMemcpyHostToDevice));
            total_uploaded += total_bytes;
        }
    } else {
        // Store type info for pingpong section to use
        ctx->output_type = m->output_type;
        size_t row_blocks = hp->n_embd / QK_K;
        if (m->output_type == GGML_TYPE_Q4_K) {
            ctx->output_row_bytes = row_blocks * sizeof(block_q4_K);
        } else if (m->output_type == GGML_TYPE_Q6_K) {
            ctx->output_row_bytes = row_blocks * sizeof(block_q6_K);
        } else {
            ctx->output_row_bytes = hp->n_embd * sizeof(float);
        }
        ctx->d_output = nullptr;
        ctx->d_output_norm = nullptr;
    }

    // --- Upload per-layer weights ---
    ctx->layers = (struct gpu_layer *)calloc(hp->n_layers, sizeof(struct gpu_layer));
    if (!ctx->layers) return -1;

    // Helper: compute quantized weight size for M rows of K columns
    auto calc_weight_size = [](int type, int M, int K) -> size_t {
        if (type == GGML_TYPE_MXFP4_ID) {
            int nb = K / QK_MXFP4;
            return (size_t)M * nb * sizeof(block_mxfp4);
        }
        if (type == GGML_TYPE_Q8_0) {
            int nb = K / QK8_0;
            return (size_t)M * nb * sizeof(block_q8_0);
        }
        int nb = K / QK_K;
        if (type == GGML_TYPE_Q4_K) return (size_t)M * nb * sizeof(block_q4_K);
        if (type == GGML_TYPE_Q5_K) return (size_t)M * nb * sizeof(block_q5_K);
        if (type == GGML_TYPE_Q6_K) return (size_t)M * nb * sizeof(block_q6_K);
        return (size_t)M * K * sizeof(float);
    };

    // Helper: upload a quantized weight matrix to GPU
    auto upload_weight = [&](void *src, int type, int M, int K,
                              void **d_dst, int *d_type, size_t *d_bytes) -> int {
        *d_type = type;
        *d_bytes = calc_weight_size(type, M, K);
        HIP_CHECK_I(hipMalloc(d_dst, *d_bytes));
        HIP_CHECK_I(hipMemcpy(*d_dst, src, *d_bytes, hipMemcpyHostToDevice));
        return 0;
    };

    // Helper: upload f32 buffer
    auto upload_f32 = [&](const float *src, float **d_dst, size_t n) -> int {
        size_t sz = n * sizeof(float);
        HIP_CHECK_I(hipMalloc(d_dst, sz));
        HIP_CHECK_I(hipMemcpy(*d_dst, src, sz, hipMemcpyHostToDevice));
        return 0;
    };

    int n_full_attn = 0, n_ssm = 0;
    for (uint32_t i = 0; i < hp->n_layers; i++) {
        // Ping-pong: upload directly to correct GPU (even→0, odd→1)
        if (ctx->pingpong) (void)hipSetDevice(i & 1);
        struct gpu_layer *gl = &ctx->layers[i];
        const layer_weights_t *l = &m->layers[i];
        gl->attn_type = l->attn_type;

        size_t norm_sz = hp->n_embd * sizeof(float);

        // Common weights: attn_norm, ffn_norm, router
        HIP_CHECK_I(hipMalloc(&gl->d_attn_norm, norm_sz));
        HIP_CHECK_I(hipMemcpy(gl->d_attn_norm, l->attn_norm, norm_sz, hipMemcpyHostToDevice));
        HIP_CHECK_I(hipMalloc(&gl->d_ffn_norm, norm_sz));
        HIP_CHECK_I(hipMemcpy(gl->d_ffn_norm, l->ffn_norm, norm_sz, hipMemcpyHostToDevice));
        total_uploaded += 2 * norm_sz;

        size_t router_sz = (size_t)hp->n_experts * hp->n_embd * sizeof(float);
        HIP_CHECK_I(hipMalloc(&gl->d_router, router_sz));
        HIP_CHECK_I(hipMemcpy(gl->d_router, l->router, router_sz, hipMemcpyHostToDevice));
        total_uploaded += router_sz;

        // Shared expert weights (all layers have these)
        if (hp->has_shared_expert && l->shared_expert_gate) {
            upload_f32(l->shared_expert_gate, &gl->d_shared_expert_gate, hp->n_embd);
            upload_weight(l->shared_gate, l->shared_gate_type,
                          hp->shared_expert_intermediate, hp->n_embd,
                          &gl->d_shared_gate, &gl->shared_gate_type, &gl->shared_gate_bytes);
            upload_weight(l->shared_up, l->shared_up_type,
                          hp->shared_expert_intermediate, hp->n_embd,
                          &gl->d_shared_up, &gl->shared_up_type, &gl->shared_up_bytes);
            upload_weight(l->shared_down, l->shared_down_type,
                          hp->n_embd, hp->shared_expert_intermediate,
                          &gl->d_shared_down, &gl->shared_down_type, &gl->shared_down_bytes);
            total_uploaded += hp->n_embd * sizeof(float) +
                              gl->shared_gate_bytes + gl->shared_up_bytes + gl->shared_down_bytes;
        }

        if (l->attn_type == LAYER_ATTN_FULL) {
            // Full attention: wq (gated, full_q_dim rows), wk, wv, wo, q_norm, k_norm
            n_full_attn++;
            upload_weight(l->wq, l->wq_type, hp->full_q_dim, hp->n_embd,
                          &gl->d_wq, &gl->wq_type, &gl->wq_bytes);
            upload_weight(l->wk, l->wk_type, hp->n_kv_heads * hp->head_dim, hp->n_embd,
                          &gl->d_wk, &gl->wk_type, &gl->wk_bytes);
            upload_weight(l->wv, l->wv_type, hp->n_kv_heads * hp->head_dim, hp->n_embd,
                          &gl->d_wv, &gl->wv_type, &gl->wv_bytes);
            upload_weight(l->wo, l->wo_type, hp->n_embd, hp->n_heads * hp->head_dim,
                          &gl->d_wo, &gl->wo_type, &gl->wo_bytes);
            total_uploaded += gl->wq_bytes + gl->wk_bytes + gl->wv_bytes + gl->wo_bytes;

            size_t head_norm_sz = hp->head_dim * sizeof(float);
            upload_f32(l->q_norm, &gl->d_q_norm, hp->head_dim);
            upload_f32(l->k_norm, &gl->d_k_norm, hp->head_dim);
            total_uploaded += 2 * head_norm_sz;
        } else {
            // DeltaNet/SSM: attn_qkv, attn_gate, ssm_a, ssm_alpha, ssm_beta,
            //               ssm_conv1d, ssm_dt_bias, ssm_norm, ssm_out
            n_ssm++;
            upload_weight(l->attn_qkv, l->attn_qkv_type, hp->ssm_qkv_dim, hp->n_embd,
                          &gl->d_attn_qkv, &gl->attn_qkv_type, &gl->attn_qkv_bytes);
            upload_weight(l->attn_gate, l->attn_gate_type, hp->ssm_inner_size, hp->n_embd,
                          &gl->d_attn_gate, &gl->attn_gate_type, &gl->attn_gate_bytes);
            total_uploaded += gl->attn_qkv_bytes + gl->attn_gate_bytes;

            upload_f32(l->ssm_a, &gl->d_ssm_a, hp->ssm_time_step_rank);
            upload_weight(l->ssm_alpha, l->ssm_alpha_type, hp->ssm_time_step_rank, hp->n_embd,
                          &gl->d_ssm_alpha, &gl->ssm_alpha_type, &gl->ssm_alpha_bytes);
            upload_weight(l->ssm_beta, l->ssm_beta_type, hp->ssm_time_step_rank, hp->n_embd,
                          &gl->d_ssm_beta, &gl->ssm_beta_type, &gl->ssm_beta_bytes);
            total_uploaded += hp->ssm_time_step_rank * sizeof(float) +
                              gl->ssm_alpha_bytes + gl->ssm_beta_bytes;

            upload_f32(l->ssm_conv1d, &gl->d_ssm_conv1d,
                       (size_t)hp->ssm_conv_kernel * hp->ssm_qkv_dim);
            upload_f32(l->ssm_dt_bias, &gl->d_ssm_dt_bias, hp->ssm_time_step_rank);
            upload_f32(l->ssm_norm, &gl->d_ssm_norm, hp->ssm_state_size);
            upload_weight(l->ssm_out, l->ssm_out_type, hp->n_embd, hp->ssm_inner_size,
                          &gl->d_ssm_out, &gl->ssm_out_type, &gl->ssm_out_bytes);
            total_uploaded += (size_t)hp->ssm_conv_kernel * hp->ssm_qkv_dim * sizeof(float) +
                              hp->ssm_time_step_rank * sizeof(float) +
                              hp->ssm_state_size * sizeof(float) +
                              gl->ssm_out_bytes;
        }

        if ((i + 1) % 16 == 0 || i == hp->n_layers - 1) {
            fprintf(stderr, "GPU:   uploaded layer %u/%u (%d full, %d ssm, %.1f GB total)\n",
                    i + 1, hp->n_layers, n_full_attn, n_ssm,
                    (double)total_uploaded / (1024.0 * 1024.0 * 1024.0));
        }
    }
    if (ctx->pingpong) (void)hipSetDevice(0);  // restore default after pingpong uploads

    // --- Allocate KV cache, SSM state, scratch on device ---
    // Skip for pingpong: per-device allocations are done in the pingpong section below.
    ctx->n_full_attn = n_full_attn;
    ctx->n_ssm_layers = n_ssm;

    if (!ctx->pingpong) {
        // Build mapping: kv_index → layer_id (only full-attn layers get KV slots)
        ctx->full_attn_layer_ids = (int *)calloc(n_full_attn, sizeof(int));
        int kv_idx = 0;
        for (uint32_t i = 0; i < hp->n_layers; i++) {
            if (m->layers[i].attn_type == LAYER_ATTN_FULL) {
                ctx->full_attn_layer_ids[kv_idx++] = i;
            }
        }

        size_t kv_dim = hp->n_kv_heads * hp->head_dim;
        size_t kv_layer_bytes = (size_t)ctx->ctx_len * kv_dim * sizeof(__half);
        size_t kv_total = (size_t)n_full_attn * kv_layer_bytes;
        HIP_CHECK_I(hipMalloc(&ctx->d_kv_k, kv_total));
        HIP_CHECK_I(hipMalloc(&ctx->d_kv_v, kv_total));
        HIP_CHECK_I(hipMemset(ctx->d_kv_k, 0, kv_total));
        HIP_CHECK_I(hipMemset(ctx->d_kv_v, 0, kv_total));
        fprintf(stderr, "GPU:   KV cache: %.1f MB per key/value (%d full-attn layers × %d ctx)\n",
                (double)kv_total / (1024.0 * 1024.0), hp->n_layers, ctx->ctx_len);

        // SSM state
        if (n_ssm > 0) {
            size_t state_per_layer = (size_t)hp->ssm_time_step_rank * hp->ssm_state_size *
                                      hp->ssm_value_dim * sizeof(float);
            size_t state_total = (size_t)n_ssm * state_per_layer;
            HIP_CHECK_I(hipMalloc(&ctx->d_ssm_states, state_total));
            HIP_CHECK_I(hipMemset(ctx->d_ssm_states, 0, state_total));

            int conv_steps = hp->ssm_conv_kernel - 1;
            size_t conv_per_layer = (size_t)conv_steps * hp->ssm_qkv_dim * sizeof(float);
            size_t conv_total = (size_t)n_ssm * conv_per_layer;
            HIP_CHECK_I(hipMalloc(&ctx->d_ssm_conv_bufs, conv_total));
            HIP_CHECK_I(hipMemset(ctx->d_ssm_conv_bufs, 0, conv_total));

            HIP_CHECK_I(hipMalloc(&ctx->d_ssm_conv_pos, n_ssm * sizeof(int)));
            HIP_CHECK_I(hipMemset(ctx->d_ssm_conv_pos, 0, n_ssm * sizeof(int)));
            HIP_CHECK_I(hipHostMalloc(&ctx->h_ssm_conv_pos, n_ssm * sizeof(int)));
            memset(ctx->h_ssm_conv_pos, 0, n_ssm * sizeof(int));

            ctx->ssm_layer_ids = (int *)calloc(n_ssm, sizeof(int));
            int ssm_idx = 0;
            for (uint32_t i = 0; i < hp->n_layers; i++) {
                if (m->layers[i].attn_type == LAYER_ATTN_LINEAR)
                    ctx->ssm_layer_ids[ssm_idx++] = i;
            }

            fprintf(stderr, "GPU:   SSM state: %.1f MB (%d layers × %zu bytes/layer)\n",
                    (double)state_total / (1024.0 * 1024.0), n_ssm, state_per_layer);
            fprintf(stderr, "GPU:   SSM conv:  %.1f MB (%d layers × %zu bytes/layer)\n",
                    (double)conv_total / (1024.0 * 1024.0), n_ssm, conv_per_layer);
        }

        // Scratch buffers
        size_t q_size = hp->full_q_dim;
        if (hp->ssm_qkv_dim > q_size) q_size = hp->ssm_qkv_dim;
        size_t attn_size = hp->n_heads * hp->head_dim;
        if (hp->ssm_inner_size > attn_size) attn_size = hp->ssm_inner_size;

        HIP_CHECK_I(hipMalloc(&ctx->d_x, hp->n_embd * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_norm, hp->n_embd * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_q, q_size * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_k, hp->n_kv_heads * hp->head_dim * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_v, hp->n_kv_heads * hp->head_dim * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_attn_out, attn_size * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_attn_scores, (size_t)hp->n_heads * ctx->ctx_len * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_attn_gate_buf, attn_size * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_router_out, hp->n_experts * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_ffn_buf, hp->n_embd * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_logits, hp->vocab_size * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_top_k_ids, MAX_EXPERTS * sizeof(int)));
        HIP_CHECK_I(hipMalloc(&ctx->d_top_k_vals, MAX_EXPERTS * sizeof(float)));

        HIP_CHECK_I(hipMalloc(&ctx->d_spec_router_out, hp->n_experts * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->d_spec_top_k_ids, MAX_EXPERTS * sizeof(int)));
        HIP_CHECK_I(hipMalloc(&ctx->d_spec_top_k_vals, MAX_EXPERTS * sizeof(float)));

        if (n_ssm > 0) {
            HIP_CHECK_I(hipMalloc(&ctx->d_ssm_dt, hp->ssm_time_step_rank * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&ctx->d_ssm_beta_buf, hp->ssm_time_step_rank * sizeof(float)));
        }

        if (hp->has_shared_expert) {
            HIP_CHECK_I(hipMalloc(&ctx->d_shared_gate_buf, hp->shared_expert_intermediate * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&ctx->d_shared_up_buf, hp->shared_expert_intermediate * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&ctx->d_shared_down_buf, hp->n_embd * sizeof(float)));
        }

        {
            size_t max_input = hp->n_embd;
            if (hp->ssm_inner_size > max_input) max_input = hp->ssm_inner_size;
            if ((size_t)(hp->n_heads * hp->head_dim) > max_input)
                max_input = hp->n_heads * hp->head_dim;
            max_input = ((max_input + 31) / 32) * 32;
            HIP_CHECK_I(hipMalloc(&ctx->d_input_q8, max_input * sizeof(int8_t)));
            HIP_CHECK_I(hipMalloc(&ctx->d_input_q8_scales, (max_input / 32) * sizeof(float)));
        }
    }

    // --- Allocate pinned host buffers ---
    HIP_CHECK_I(hipHostMalloc(&ctx->h_router_out, hp->n_experts * sizeof(float)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_top_k_ids, MAX_EXPERTS * sizeof(int)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_top_k_vals, MAX_EXPERTS * sizeof(float)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_spec_top_k_ids, MAX_EXPERTS * sizeof(int)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_spec2_top_k_ids, MAX_EXPERTS * sizeof(int)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_all_scores, hp->n_experts * sizeof(float)));
    HIP_CHECK_I(hipHostMalloc(&ctx->h_logits, hp->vocab_size * sizeof(float)));

    ctx->model_uploaded = true;
    fprintf(stderr, "GPU: model upload complete (%.2f GB VRAM used for weights)\n",
            (double)total_uploaded / (1024.0 * 1024.0 * 1024.0));

    // --- Allocate expert staging buffer with actual expert_stride ---
    // Deferred from gpu_create_context so we use actual stride, not MAX_EXPERT_STRIDE.
    // Skip in ping-pong: per-device staging allocated later.
    if (!ctx->pingpong) {
        size_t staging_size = (size_t)ctx->n_experts_used * 2 * ctx->expert_stride;
        HIP_CHECK_I(hipMalloc(&ctx->d_expert_staging, staging_size));
        fprintf(stderr, "GPU: expert staging: %.1f MB (2×%d slots × %.2f MB stride)\n",
                (double)staging_size / (1024.0 * 1024.0),
                ctx->n_experts_used,
                (double)ctx->expert_stride / (1024.0 * 1024.0));
    }

    // Expert VRAM cache was allocated early (before weight uploads) to avoid fragmentation

    // --- GPU1 expert engine: allocate staging, VRAM cache, scratch ---
    // Skip in ping-pong: per-device resources allocated in the ping-pong section below.
    // Stage mode uses the same gpu1 struct for GPU_B (MoE expert engine).
    if (ctx->n_devices >= 2 && !ctx->pingpong) {
        (void)hipSetDevice(ctx->gpu1.device_id);

        // Expert staging (same size as GPU0)
        size_t staging_size = (size_t)ctx->n_experts_used * 2 * ctx->expert_stride;
        HIP_CHECK_I(hipMalloc(&ctx->gpu1.d_expert_staging, staging_size));
        fprintf(stderr, "GPU1: expert staging: %.1f MB\n",
                (double)staging_size / (1024.0 * 1024.0));

        // Input/output scratch
        HIP_CHECK_I(hipMalloc(&ctx->gpu1.d_norm, ctx->n_embd * sizeof(float)));
        HIP_CHECK_I(hipMalloc(&ctx->gpu1.d_partial, ctx->n_embd * sizeof(float)));

        // VRAM expert cache — GPU1 has almost no weights, so nearly all VRAM is cache
        {
            size_t free_mem = 0, total_mem = 0;
            hipMemGetInfo(&free_mem, &total_mem);
            size_t overhead = 256ULL * 1024 * 1024;  // staging + FFN buffers + headroom
            size_t cache_budget = (free_mem > overhead) ? free_mem - overhead : 0;

            int max_slots = (int)(cache_budget / ctx->expert_stride);
            if (max_slots < 0) max_slots = 0;
            int total_experts = ctx->n_layers * ctx->n_experts_total;
            if (max_slots > total_experts) max_slots = total_experts;

            memset(&ctx->gpu1.ecache, 0, sizeof(ctx->gpu1.ecache));
            if (max_slots > 0 && ctx->n_layers > 0) {
                if (vram_cache_init(&ctx->gpu1.ecache, ctx->n_layers, ctx->n_experts_total,
                                    max_slots, ctx->expert_stride) == 0) {
                    max_slots = ctx->gpu1.ecache.max_slots;
                    size_t alloc_sz = (size_t)max_slots * ctx->expert_stride;
                    hipError_t err = hipMalloc(&ctx->gpu1.ecache.d_buf, alloc_sz);
                    if (err != hipSuccess) {
                        // Retry smaller
                        while (max_slots >= ctx->n_layers) {
                            max_slots = (max_slots * 3 / 4);
                            max_slots -= max_slots % ctx->n_layers;
                            vram_cache_free(&ctx->gpu1.ecache);
                            if (max_slots >= ctx->n_layers &&
                                vram_cache_init(&ctx->gpu1.ecache, ctx->n_layers, ctx->n_experts_total,
                                                max_slots, ctx->expert_stride) == 0) {
                                max_slots = ctx->gpu1.ecache.max_slots;
                                alloc_sz = (size_t)max_slots * ctx->expert_stride;
                                err = hipMalloc(&ctx->gpu1.ecache.d_buf, alloc_sz);
                                if (err == hipSuccess) break;
                            }
                        }
                    }
                    if (err != hipSuccess) {
                        fprintf(stderr, "GPU1: expert cache alloc failed, disabling\n");
                        vram_cache_free(&ctx->gpu1.ecache);
                    } else {
                        fprintf(stderr, "GPU1: expert cache: %d slots (%d/layer, %.1f MB), %.1f%% of %d total\n",
                                max_slots, ctx->gpu1.ecache.slots_per_layer,
                                (double)alloc_sz / (1024.0 * 1024.0),
                                100.0 * max_slots / total_experts, total_experts);
                    }
                }
            }
            if (ctx->gpu1.ecache.max_slots == 0)
                fprintf(stderr, "GPU1: expert cache: disabled (insufficient VRAM)\n");
        }

        // Return to GPU0
        (void)hipSetDevice(0);

        int combined_slots = ctx->ecache.max_slots + ctx->gpu1.ecache.max_slots;
        int combined_spl = (ctx->n_layers > 0) ? combined_slots / ctx->n_layers : 0;
        fprintf(stderr, "GPU: dual-GPU expert cache: %d + %d = %d slots (%d/layer combined)\n",
                ctx->ecache.max_slots, ctx->gpu1.ecache.max_slots,
                combined_slots, combined_spl);
    }

    // Inter-GPU pinned staging (needed for both expert-split and ping-pong relay)
    if (ctx->n_devices >= 2) {
        HIP_CHECK_I(hipHostMalloc(&ctx->h_gpu1_norm, ctx->n_embd * sizeof(float), hipHostMallocDefault));
        HIP_CHECK_I(hipHostMalloc(&ctx->h_gpu1_partial, ctx->n_embd * sizeof(float), hipHostMallocDefault));
        HIP_CHECK_I(hipMalloc(&ctx->d_gpu1_partial, ctx->n_embd * sizeof(float)));
    } else {
        ctx->h_gpu1_norm = nullptr;
        ctx->h_gpu1_partial = nullptr;
        ctx->d_gpu1_partial = nullptr;
    }

    // --- Ping-pong dual-GPU: allocate per-device resources ---
    if (ctx->pingpong && ctx->n_devices >= 2) {
        const model_t *m_pp = (const model_t *)model_ptr;
        int n_layers_total = ctx->n_layers;

        size_t q_size = hp->full_q_dim;
        if (hp->ssm_qkv_dim > q_size) q_size = hp->ssm_qkv_dim;
        size_t attn_size = hp->n_heads * hp->head_dim;
        if (hp->ssm_inner_size > attn_size) attn_size = hp->ssm_inner_size;
        size_t max_input = hp->n_embd;
        if (hp->ssm_inner_size > max_input) max_input = hp->ssm_inner_size;
        if ((size_t)(hp->n_heads * hp->head_dim) > max_input)
            max_input = hp->n_heads * hp->head_dim;
        max_input = ((max_input + 31) / 32) * 32;

        for (int d = 0; d < 2; d++) {
            (void)hipSetDevice(d);
            gpu_dev_t *dc = &ctx->dev[d];

            // Count layers for this device
            dc->n_layers = 0;
            dc->n_full_attn = 0;
            dc->n_ssm_layers = 0;
            for (int il = 0; il < n_layers_total; il++) {
                if ((il & 1) == d) {
                    dc->n_layers++;
                    if (m_pp->layers[il].attn_type == LAYER_ATTN_FULL)
                        dc->n_full_attn++;
                    else
                        dc->n_ssm_layers++;
                }
            }

            // Scratch buffers
            HIP_CHECK_I(hipMalloc(&dc->d_x, hp->n_embd * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_norm, hp->n_embd * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_q, q_size * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_k, hp->n_kv_heads * hp->head_dim * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_v, hp->n_kv_heads * hp->head_dim * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_attn_out, attn_size * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_attn_scores, (size_t)hp->n_heads * ctx->ctx_len * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_attn_gate_buf, attn_size * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_router_out, hp->n_experts * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_ffn_buf, hp->n_embd * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_top_k_ids, MAX_EXPERTS * sizeof(int)));
            HIP_CHECK_I(hipMalloc(&dc->d_top_k_vals, MAX_EXPERTS * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_spec_router_out, hp->n_experts * sizeof(float)));
            HIP_CHECK_I(hipMalloc(&dc->d_spec_top_k_ids, MAX_EXPERTS * sizeof(int)));
            HIP_CHECK_I(hipMalloc(&dc->d_spec_top_k_vals, MAX_EXPERTS * sizeof(float)));

            // SSM scratch
            if (dc->n_ssm_layers > 0) {
                HIP_CHECK_I(hipMalloc(&dc->d_ssm_dt, hp->ssm_time_step_rank * sizeof(float)));
                HIP_CHECK_I(hipMalloc(&dc->d_ssm_beta_buf, hp->ssm_time_step_rank * sizeof(float)));
            }

            // Shared expert scratch
            if (hp->has_shared_expert) {
                HIP_CHECK_I(hipMalloc(&dc->d_shared_gate_buf, hp->shared_expert_intermediate * sizeof(float)));
                HIP_CHECK_I(hipMalloc(&dc->d_shared_up_buf, hp->shared_expert_intermediate * sizeof(float)));
                HIP_CHECK_I(hipMalloc(&dc->d_shared_down_buf, hp->n_embd * sizeof(float)));
            }

            // Pre-quantized input buffers
            HIP_CHECK_I(hipMalloc(&dc->d_input_q8, max_input * sizeof(int8_t)));
            HIP_CHECK_I(hipMalloc(&dc->d_input_q8_scales, (max_input / 32) * sizeof(float)));

            // KV cache (this device's full-attention layers only)
            if (dc->n_full_attn > 0) {
                dc->full_attn_layer_ids = (int *)calloc(dc->n_full_attn, sizeof(int));
                int ki = 0;
                for (int il = 0; il < n_layers_total; il++) {
                    if ((il & 1) == d && m_pp->layers[il].attn_type == LAYER_ATTN_FULL) {
                        dc->full_attn_layer_ids[ki++] = il;
                    }
                }
                size_t kv_dim = hp->n_kv_heads * hp->head_dim;
                size_t kv_layer_bytes = (size_t)ctx->ctx_len * kv_dim * sizeof(__half);
                size_t kv_total = (size_t)dc->n_full_attn * kv_layer_bytes;
                HIP_CHECK_I(hipMalloc(&dc->d_kv_k, kv_total));
                HIP_CHECK_I(hipMalloc(&dc->d_kv_v, kv_total));
                HIP_CHECK_I(hipMemset(dc->d_kv_k, 0, kv_total));
                HIP_CHECK_I(hipMemset(dc->d_kv_v, 0, kv_total));
                fprintf(stderr, "GPU: dev[%d] KV cache: %.1f MB (%d full-attn layers)\n",
                        d, (double)(kv_total * 2) / (1024.0 * 1024.0), dc->n_full_attn);
            }

            // SSM state (this device's DeltaNet layers only)
            if (dc->n_ssm_layers > 0) {
                dc->ssm_layer_ids = (int *)calloc(dc->n_ssm_layers, sizeof(int));
                int si = 0;
                for (int il = 0; il < n_layers_total; il++) {
                    if ((il & 1) == d && m_pp->layers[il].attn_type == LAYER_ATTN_LINEAR) {
                        dc->ssm_layer_ids[si++] = il;
                    }
                }
                size_t state_per_layer = (size_t)hp->ssm_time_step_rank * hp->ssm_state_size *
                                          hp->ssm_value_dim * sizeof(float);
                size_t state_total = (size_t)dc->n_ssm_layers * state_per_layer;
                HIP_CHECK_I(hipMalloc(&dc->d_ssm_states, state_total));
                HIP_CHECK_I(hipMemset(dc->d_ssm_states, 0, state_total));

                int conv_steps = hp->ssm_conv_kernel - 1;
                size_t conv_per_layer = (size_t)conv_steps * hp->ssm_qkv_dim * sizeof(float);
                size_t conv_total = (size_t)dc->n_ssm_layers * conv_per_layer;
                HIP_CHECK_I(hipMalloc(&dc->d_ssm_conv_bufs, conv_total));
                HIP_CHECK_I(hipMemset(dc->d_ssm_conv_bufs, 0, conv_total));

                HIP_CHECK_I(hipMalloc(&dc->d_ssm_conv_pos, dc->n_ssm_layers * sizeof(int)));
                HIP_CHECK_I(hipMemset(dc->d_ssm_conv_pos, 0, dc->n_ssm_layers * sizeof(int)));
                HIP_CHECK_I(hipHostMalloc(&dc->h_ssm_conv_pos, dc->n_ssm_layers * sizeof(int)));
                memset(dc->h_ssm_conv_pos, 0, dc->n_ssm_layers * sizeof(int));

                fprintf(stderr, "GPU: dev[%d] SSM state: %.1f MB (%d layers)\n",
                        d, (double)(state_total + conv_total) / (1024.0 * 1024.0), dc->n_ssm_layers);
            }

            // Expert staging
            {
                size_t staging_size = (size_t)ctx->n_experts_used * 2 * ctx->expert_stride;
                HIP_CHECK_I(hipMalloc(&dc->d_expert_staging, staging_size));
                fprintf(stderr, "GPU: dev[%d] expert staging: %.1f MB\n",
                        d, (double)staging_size / (1024.0 * 1024.0));
            }

            // VRAM cache deferred to after weight move (GPU0 needs freed VRAM first)
            memset(&dc->ecache, 0, sizeof(dc->ecache));

            // Pinned host staging
            HIP_CHECK_I(hipHostMalloc(&dc->h_router_out, hp->n_experts * sizeof(float)));
            HIP_CHECK_I(hipHostMalloc(&dc->h_top_k_ids, MAX_EXPERTS * sizeof(int)));
            HIP_CHECK_I(hipHostMalloc(&dc->h_top_k_vals, MAX_EXPERTS * sizeof(float)));
            HIP_CHECK_I(hipHostMalloc(&dc->h_all_scores, hp->n_experts * sizeof(float)));
            HIP_CHECK_I(hipHostMalloc(&dc->h_spec_top_k_ids, MAX_EXPERTS * sizeof(int)));
            HIP_CHECK_I(hipHostMalloc(&dc->h_norm, hp->n_embd * sizeof(float)));

            fprintf(stderr, "GPU: dev[%d] scratch + cache allocated (%d layers: %d full, %d ssm)\n",
                    d, dc->n_layers, dc->n_full_attn, dc->n_ssm_layers);
        }

        // LM head goes on GPU with last layer (dev[1] for 60-layer model, since 59 is odd)
        {
            int last_d = (n_layers_total - 1) & 1;
            gpu_dev_t *last_dc = &ctx->dev[last_d];
            (void)hipSetDevice(last_d);

            // Output norm
            size_t norm_sz = hp->n_embd * sizeof(float);
            HIP_CHECK_I(hipMalloc(&last_dc->d_output_norm, norm_sz));
            HIP_CHECK_I(hipMemcpy(last_dc->d_output_norm, m_pp->output_norm, norm_sz, hipMemcpyHostToDevice));

            // LM head weight
            last_dc->output_type = m_pp->output_type;
            size_t row_blocks = hp->n_embd / QK_K;
            if (m_pp->output_type == GGML_TYPE_Q4_K) {
                last_dc->output_row_bytes = row_blocks * sizeof(block_q4_K);
            } else if (m_pp->output_type == GGML_TYPE_Q6_K) {
                last_dc->output_row_bytes = row_blocks * sizeof(block_q6_K);
            } else {
                last_dc->output_row_bytes = hp->n_embd * sizeof(float);
            }
            size_t output_total = (size_t)hp->vocab_size * last_dc->output_row_bytes;
            HIP_CHECK_I(hipMalloc(&last_dc->d_output, output_total));
            HIP_CHECK_I(hipMemcpy(last_dc->d_output, m_pp->output, output_total, hipMemcpyHostToDevice));

            // Logits buffer + pinned host
            HIP_CHECK_I(hipMalloc(&last_dc->d_logits, hp->vocab_size * sizeof(float)));
            HIP_CHECK_I(hipHostMalloc(&last_dc->h_logits, hp->vocab_size * sizeof(float)));

            fprintf(stderr, "GPU: ping-pong LM head on dev[%d] (%.1f MB)\n",
                    last_d, (double)output_total / (1024.0 * 1024.0));
        }

        // Weight migration NOT needed: upload loop already placed weights
        // on the correct GPU via hipSetDevice(i & 1).
        // Free gpu_create_context expert FFN buffers (unused in pingpong)
        (void)hipSetDevice(0);
        (void)hipFree(ctx->d_gate);     ctx->d_gate = nullptr;
        (void)hipFree(ctx->d_up);       ctx->d_up = nullptr;
        (void)hipFree(ctx->d_down);     ctx->d_down = nullptr;
        (void)hipFree(ctx->d_expert_ptrs); ctx->d_expert_ptrs = nullptr;
        (void)hipFree(ctx->d_scores);   ctx->d_scores = nullptr;
        (void)hipFree(ctx->d_ffn_out);  ctx->d_ffn_out = nullptr;
        (void)hipSetDevice(1);
        (void)hipFree(ctx->gpu1.d_gate);     ctx->gpu1.d_gate = nullptr;
        (void)hipFree(ctx->gpu1.d_up);       ctx->gpu1.d_up = nullptr;
        (void)hipFree(ctx->gpu1.d_down);     ctx->gpu1.d_down = nullptr;
        (void)hipFree(ctx->gpu1.d_expert_ptrs); ctx->gpu1.d_expert_ptrs = nullptr;
        (void)hipFree(ctx->gpu1.d_scores);   ctx->gpu1.d_scores = nullptr;
        (void)hipFree(ctx->gpu1.d_ffn_out);  ctx->gpu1.d_ffn_out = nullptr;
        (void)hipSetDevice(0);

        // Allocate VRAM cache for each device (no fragmentation from migration)
        // CRITICAL FIX: use dc->n_layers (30) not ctx->n_layers (60).
        // Each GPU only runs 30 layers, so the cache should be partitioned
        // over 30 layers. This doubles slots/layer from ~35 to ~69.
        for (int d = 0; d < 2; d++) {
            (void)hipSetDevice(d);
            gpu_dev_t *dc = &ctx->dev[d];
            int dev_n_layers = dc->n_layers;  // 30, not 60
            size_t free_mem = 0, total_mem = 0;
            hipMemGetInfo(&free_mem, &total_mem);
            size_t cache_budget;
            if (ctx->vram_cache_mb > 0) {
                // User-specified limit (per GPU, so divide by 2)
                cache_budget = (size_t)ctx->vram_cache_mb * 1024ULL * 1024ULL / 2;
            } else {
                size_t overhead = 256ULL * 1024 * 1024;
                cache_budget = (free_mem > overhead) ? free_mem - overhead : 0;
            }
            int max_slots = (int)(cache_budget / ctx->expert_stride);
            if (max_slots < 0) max_slots = 0;
            int total_experts_dev = dev_n_layers * ctx->n_experts_total;
            if (max_slots > total_experts_dev) max_slots = total_experts_dev;

            if (max_slots > 0 && dev_n_layers > 0) {
                if (vram_cache_init(&dc->ecache, dev_n_layers, ctx->n_experts_total,
                                    max_slots, ctx->expert_stride) == 0) {
                    max_slots = dc->ecache.max_slots;
                    size_t alloc_sz = (size_t)max_slots * ctx->expert_stride;
                    hipError_t cerr = hipMalloc(&dc->ecache.d_buf, alloc_sz);
                    if (cerr != hipSuccess) {
                        while (max_slots >= dev_n_layers) {
                            max_slots = (max_slots * 3 / 4);
                            max_slots -= max_slots % dev_n_layers;
                            vram_cache_free(&dc->ecache);
                            if (max_slots >= dev_n_layers &&
                                vram_cache_init(&dc->ecache, dev_n_layers, ctx->n_experts_total,
                                                max_slots, ctx->expert_stride) == 0) {
                                max_slots = dc->ecache.max_slots;
                                alloc_sz = (size_t)max_slots * ctx->expert_stride;
                                cerr = hipMalloc(&dc->ecache.d_buf, alloc_sz);
                                if (cerr == hipSuccess) break;
                            }
                        }
                    }
                    if (cerr != hipSuccess) {
                        fprintf(stderr, "GPU: dev[%d] VRAM cache alloc failed\n", d);
                        vram_cache_free(&dc->ecache);
                    } else {
                        fprintf(stderr, "GPU: dev[%d] VRAM cache: %d slots (%d/layer, %.1f MB), "
                                "using %d device layers (not %d total)\n",
                                d, max_slots, dc->ecache.slots_per_layer,
                                (double)alloc_sz / (1024.0 * 1024.0),
                                dev_n_layers, ctx->n_layers);
                    }
                }
            }
        }

#ifdef QMOE_CUDA
        // CUDA: shared memory settings for each device
        for (int d = 0; d < 2; d++) {
            (void)hipSetDevice(d);
            int max_smem = hp->full_q_dim * (int)sizeof(float);
            int attn_smem = ctx->ctx_len * (int)sizeof(float);
            if (attn_smem > max_smem) max_smem = attn_smem;
            hipFuncSetAttribute((const void *)kernel_deinterleave_q_gate,
                                hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
            hipFuncSetAttribute((const void *)kernel_gqa_attention,
                                hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
            hipFuncSetAttribute((const void *)kernel_f32_matvec_coop,
                                hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        }
#endif

        (void)hipSetDevice(0);

        // Start spec routing worker thread
        memset(&ctx->spec_worker, 0, sizeof(ctx->spec_worker));
        if (!getenv("QMOE_NO_SPEC")) {
            pthread_mutex_init(&ctx->spec_worker.mutex, nullptr);
            pthread_cond_init(&ctx->spec_worker.cond, nullptr);
            ctx->spec_worker.active = true;
            ctx->spec_worker.work_done = true;  // no pending work initially
            pthread_create(&ctx->spec_worker.thread, nullptr, pp_spec_worker_fn, ctx);
            fprintf(stderr, "GPU: ping-pong spec routing worker started\n");
        } else {
            fprintf(stderr, "GPU: ping-pong spec routing worker DISABLED\n");
        }

        fprintf(stderr, "GPU: ping-pong dual-GPU setup complete (%d layers per GPU)\n",
                n_layers_total / 2);
    }

    // --- Initialize expert RAM cache (v2 module, pinned host memory) ---
    {
        memset(&ctx->rcache, 0, sizeof(ctx->rcache));
        ctx->rcache_registered = false;

        size_t cache_budget = 0;
        if (ctx->ram_cache_mb == 0) {
            fprintf(stderr, "GPU: RAM cache: disabled by config\n");
        } else if (ctx->ram_cache_mb > 0) {
            cache_budget = (size_t)ctx->ram_cache_mb * 1024ULL * 1024ULL;
            fprintf(stderr, "GPU: RAM cache: configured %d MB\n", ctx->ram_cache_mb);
        } else {
            size_t total_ram = 0, avail_ram = 0;
            FILE *mi = fopen("/proc/meminfo", "r");
            if (mi) {
                char line[256];
                while (fgets(line, sizeof(line), mi)) {
                    unsigned long val;
                    if (sscanf(line, "MemTotal: %lu kB", &val) == 1)
                        total_ram = (size_t)val * 1024;
                    else if (sscanf(line, "MemAvailable: %lu kB", &val) == 1)
                        avail_ram = (size_t)val * 1024;
                }
                fclose(mi);
            }
            if (avail_ram > 0) {
                size_t headroom = 4ULL * 1024 * 1024 * 1024;
                size_t usable = (avail_ram > headroom) ? avail_ram - headroom : 0;
                cache_budget = usable * 3 / 4;
                fprintf(stderr, "GPU: RAM cache auto: total=%.1f GB, available=%.1f GB, budget=%.1f GB\n",
                        (double)total_ram / (1024.0 * 1024.0 * 1024.0),
                        (double)avail_ram / (1024.0 * 1024.0 * 1024.0),
                        (double)cache_budget / (1024.0 * 1024.0 * 1024.0));
            } else {
                fprintf(stderr, "GPU: RAM cache: /proc/meminfo unavailable, disabling\n");
            }
        }

        if (cache_budget > 0 && ctx->expert_stride > 0 && ctx->n_layers > 0) {
            int max_slots = (int)(cache_budget / ctx->expert_stride);
            int total_experts = ctx->n_layers * ctx->n_experts_total;
            if (max_slots > total_experts) max_slots = total_experts;

            // Allocate buffer first, then init metadata
            void *buf = nullptr;
            bool pinned = false;
            int spl = max_slots / ctx->n_layers;
            max_slots = spl * ctx->n_layers;

            if (max_slots > 0) {
                size_t alloc_sz = (size_t)max_slots * ctx->expert_stride;

                // Skip hipHostMalloc for large allocs or multi-GPU — it can hang
                // when GPU BAR space is exhausted. Use malloc + hipHostRegister instead.
                if (!ctx->pingpong && alloc_sz < 12ULL * 1024 * 1024 * 1024) {
                    hipError_t err = hipHostMalloc(&buf, alloc_sz, hipHostMallocDefault);
                    if (err == hipSuccess) pinned = true;
                }
                if (!buf) {
                    buf = malloc(alloc_sz);
                    if (buf) {
                        memset(buf, 0, alloc_sz);
                        // Skip pinning for large allocations (>16GB) — hipHostRegister
                        // of huge regions adds TLB/page-table overhead to ALL HIP calls,
                        // causing ~15-20ms/token regression. Use bounce buffer instead.
                        if (alloc_sz <= 17ULL * 1024 * 1024 * 1024) {
                            hipError_t err = hipHostRegister(buf, alloc_sz, hipHostRegisterDefault);
                            if (err == hipSuccess) {
                                pinned = true;
                                ctx->rcache_registered = true;
                                fprintf(stderr, "GPU: RAM cache: pinned %.1f GB\n",
                                        (double)alloc_sz / (1024.0 * 1024.0 * 1024.0));
                            } else {
                                fprintf(stderr, "GPU: RAM cache: pin failed (%s), using bounce buffer\n",
                                        hipGetErrorString(err));
                            }
                        } else {
                            fprintf(stderr, "GPU: RAM cache: %.1f GB too large to pin, using bounce buffer\n",
                                    (double)alloc_sz / (1024.0 * 1024.0 * 1024.0));
                        }
                    }
                }

                if (buf && max_slots > 0) {
                    if (ram_cache_init(&ctx->rcache, ctx->n_layers, ctx->n_experts_total,
                                       max_slots, ctx->expert_stride) == 0) {
                        ctx->rcache.buf = buf;
                        ctx->rcache.pinned = pinned;
                        fprintf(stderr, "GPU: RAM cache: %d slots (%d/layer, %.1f GB %s), %.1f%% of %d total experts\n",
                                ctx->rcache.max_slots, ctx->rcache.slots_per_layer,
                                (double)alloc_sz / (1024.0 * 1024.0 * 1024.0),
                                pinned ? "pinned" : "unpinned",
                                100.0 * ctx->rcache.max_slots / total_experts, total_experts);
                    } else {
                        // ram_cache_init failed
                        if (ctx->rcache_registered) { hipHostUnregister(buf); free(buf); }
                        else if (pinned) hipHostFree(buf);
                        else free(buf);
                    }
                } else {
                    fprintf(stderr, "GPU: RAM cache: allocation failed, disabled\n");
                }
            }
        }
    }

    // Allocate pinned bounce buffers for unpinned RAM cache → GPU H2D copies
    // Need n_experts_used buffers so all per-layer copies can be async simultaneously
    ctx->h2d_bounce = nullptr;
    ctx->h2d_bounce_slots = 0;
    // Only allocate bounce buffer for unpinned RAM cache (needed for DMA).
    // With pinned RAM, direct scattered H2D is faster (avoids serial CPU memcpy).
    if (ctx->rcache.max_slots > 0 && !ctx->rcache.pinned) {
        size_t bounce_sz = (size_t)ctx->n_experts_used * ctx->expert_stride;
        hipError_t err = hipHostMalloc(&ctx->h2d_bounce, bounce_sz, hipHostMallocDefault);
        if (err == hipSuccess) {
            ctx->h2d_bounce_slots = ctx->n_experts_used;
            fprintf(stderr, "GPU: H2D bounce buffer: %.1f MB pinned (%d slots)\n",
                    (double)bounce_sz / (1024.0 * 1024.0), ctx->n_experts_used);
        }
    }

    // Initialize bounce memcpy worker (runs memcpy in background)
    ctx->bounce_worker.active = false;
    ctx->bounce_worker.work_ready = false;
    ctx->bounce_worker.work_done = true;
    ctx->bounce_worker.shutdown = false;
    if (ctx->h2d_bounce && ctx->rcache.max_slots > 0) {
        ctx->bounce_worker.bounce = ctx->h2d_bounce;
        ctx->bounce_worker.cache = ctx->rcache.buf;
        ctx->bounce_worker.stride = ctx->expert_stride;
        pthread_mutex_init(&ctx->bounce_worker.mutex, NULL);
        pthread_cond_init(&ctx->bounce_worker.cond_work, NULL);
        pthread_cond_init(&ctx->bounce_worker.cond_done, NULL);
        if (pthread_create(&ctx->bounce_worker.thread, NULL, bounce_memcpy_worker, ctx) == 0) {
            ctx->bounce_worker.active = true;
            fprintf(stderr, "GPU: async bounce memcpy worker enabled\n");
        }
    }

    // Initialize async RAM cache population worker
    ctx->rcache_pop.active = false;
    ctx->rcache_pop.work_ready = false;
    ctx->rcache_pop.work_done = true;  // no pending work initially
    ctx->rcache_pop.shutdown = false;
    ctx->rcache_pop.n = 0;
    ctx->rcache_pop.cache_buf = ctx->rcache.buf;
    ctx->rcache_pop.stride = ctx->expert_stride;
    if (ctx->rcache.max_slots > 0) {
        pthread_mutex_init(&ctx->rcache_pop.mutex, NULL);
        pthread_cond_init(&ctx->rcache_pop.cond_work, NULL);
        pthread_cond_init(&ctx->rcache_pop.cond_done, NULL);
        if (pthread_create(&ctx->rcache_pop.thread, NULL, rcache_pop_worker, ctx) == 0) {
            ctx->rcache_pop.active = true;
            fprintf(stderr, "GPU: async RAM cache population enabled\n");
        }
    }

    // CAR quality controls (placeholder — applied after car_init below)
    ctx->car_protect_layers = 0;

    // Initialize profiling events
    for (int i = 0; i < 12; i++) {
        HIP_CHECK_I(hipEventCreate(&ctx->prof.e[i]));
    }
    HIP_CHECK_I(hipEventCreate(&ctx->spec_event));
    HIP_CHECK_I(hipEventCreate(&ctx->spec2_event));
    HIP_CHECK_I(hipEventCreate(&ctx->vram_prefetch_event));
    ctx->vram_prefetch_active = false;
    ctx->prof.token_count = 0;

#ifdef QMOE_CUDA
    // CUDA: opt-in to >48 KB shared memory for kernels that need it.
    // kernel_deinterleave_q_gate uses full_q_dim * 4 = 64 KB.
    // kernel_gqa_attention uses seq_len * 4 (up to ctx_len * 4 at long sequences).
    {
        int max_smem = hp->full_q_dim * (int)sizeof(float);  // 64 KB
        int attn_smem = ctx->ctx_len * (int)sizeof(float);   // up to ctx_len * 4
        if (attn_smem > max_smem) max_smem = attn_smem;
        hipFuncSetAttribute((const void *)kernel_deinterleave_q_gate,
                            hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        hipFuncSetAttribute((const void *)kernel_gqa_attention,
                            hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        hipFuncSetAttribute((const void *)kernel_f32_matvec_coop,
                            hipFuncAttributeMaxDynamicSharedMemorySize, max_smem);
    }
#endif

    // Speculative expert prefetch: skip-attention routing prediction.
    // GPU predicts next layer's experts at ~86% accuracy during Phase 1 FFN,
    // loads uncached predicted experts into RAM cache after e[7].
    ctx->prefetch.active = false;
    ctx->prefetch.has_prev = false;
    if (m->nvme_io && ctx->rcache.max_slots > 0) {
        ctx->prefetch.active = true;
        fprintf(stderr, "GPU: speculative routing enabled (sync prefetch → RAM cache)\n");
    }

    // Initialize NVMe background loader thread
    ctx->nvme_bg.active = false;
    ctx->nvme_bg.work_ready = false;
    ctx->nvme_bg.work_done = true;
    ctx->nvme_bg.shutdown = false;
    if (m->nvme_io) {
        pthread_mutex_init(&ctx->nvme_bg.mutex, NULL);
        pthread_cond_init(&ctx->nvme_bg.cond_work, NULL);
        pthread_cond_init(&ctx->nvme_bg.cond_done, NULL);
        if (pthread_create(&ctx->nvme_bg.thread, NULL, nvme_bg_worker, ctx) == 0) {
            ctx->nvme_bg.active = true;
            fprintf(stderr, "GPU: async NVMe background loader enabled\n");
        }
    }

    // Initialize speculative prefetch (v2 module)
    ctx->pf = nullptr;
    ctx->pf_pending = false;
    if (m->nvme_io && ctx->rcache.max_slots > 0) {
        const char *budget_env = getenv("QMOE_PREFETCH_BUDGET");
        int nvme_budget = budget_env ? atoi(budget_env) : 3;
        int spec_k = ctx->n_experts_used;  // predict same K as active experts
        ctx->pf = prefetch_create(nvme_budget, spec_k,
                                   m->nvme_io, &ctx->rcache, 20);
        if (ctx->pf)
            fprintf(stderr, "GPU: v2 speculative prefetch enabled (budget=%d)\n", nvme_budget);
    }
    ctx->no_cold_fetch = getenv("QMOE_NO_COLD_FETCH") != NULL;
    if (ctx->no_cold_fetch)
        fprintf(stderr, "GPU: no-cold-fetch mode (NVMe only via async prefetch)\n");

    // --- Routing trace ---
    memset(&ctx->trace, 0, sizeof(ctx->trace));
    {
        const char *trace_path = getenv("QMOE_TRACE_FILE");
        if (trace_path && trace_path[0]) {
            ctx->trace.fp = fopen(trace_path, "wb");
            if (ctx->trace.fp) {
                ctx->trace.buf_cap = 256 * 1024;
                ctx->trace.buf = (uint8_t *)malloc(ctx->trace.buf_cap);
                trace_header_t hdr = {};
                hdr.magic = TRACE_MAGIC;
                hdr.version = TRACE_VERSION;
                hdr.n_layers = (uint16_t)ctx->n_layers;
                hdr.n_experts_total = (uint16_t)ctx->n_experts_total;
                hdr.top_k = (uint16_t)ctx->n_experts_used;
                fwrite(&hdr, sizeof(hdr), 1, ctx->trace.fp);
                fprintf(stderr, "GPU: routing trace enabled -> %s\n", trace_path);
            } else {
                fprintf(stderr, "GPU: warning: cannot open trace file %s\n", trace_path);
            }
        }
    }

    // --- Initialize CAR (v2 module) ---
    {
        const char *env = getenv("QMOE_CAR_THRESHOLD");
        float threshold = env ? atof(env) : 0.0f;
        // No-cold-fetch requires CAR to substitute NVMe misses (else they get zeroed)
        if (ctx->no_cold_fetch && threshold == 0.0f) {
            threshold = 0.01f;
            fprintf(stderr, "GPU: no-cold-fetch auto-enabled CAR (threshold=0.01)\n");
        }
        car_init(&ctx->car, threshold);
        if (threshold > 0.0f)
            fprintf(stderr, "GPU: cache-aware routing: threshold=%.2f\n", threshold);
    }

    // CAR quality controls (must be AFTER car_init which zeroes the struct)
    ctx->car.n_layers = ctx->n_layers;
    ctx->car.skip_renorm = getenv("QMOE_CAR_DAMPEN") != NULL;
    ctx->car.layer_weight = getenv("QMOE_CAR_LAYER_WEIGHT") != NULL;
    {
        const char *p = getenv("QMOE_CAR_MAX_SUBS");
        if (p) ctx->car.max_subs_per_layer = atoi(p);
        p = getenv("QMOE_CAR_PROTECT");
        if (p) ctx->car_protect_layers = atoi(p);
        // No-cold-fetch: disable layer protection (can't skip CAR — experts get zeroed)
        if (ctx->no_cold_fetch && ctx->car_protect_layers > 0) {
            fprintf(stderr, "GPU: no-cold-fetch overrides CAR layer protection (was %d)\n",
                    ctx->car_protect_layers);
            ctx->car_protect_layers = 0;
        }
        // Dynamic accumulated budget: QMOE_CAR_BUDGET=<total> QMOE_CAR_FREE_RATIO=<ratio>
        p = getenv("QMOE_CAR_BUDGET");
        if (p && ctx->n_layers > 0) {
            float total = atof(p);
            ctx->car.budget_per_layer = total / ctx->n_layers;
        }
        p = getenv("QMOE_CAR_FREE_RATIO");
        if (p) ctx->car.free_ratio = atof(p);
    }
    if (ctx->car.threshold > 0.0f && ctx->car.threshold < 1.0f) {
        if (ctx->car.budget_per_layer > 0.0f) {
            fprintf(stderr, "GPU: CAR dynamic budget: %.1f/token (%.2f/layer), free_ratio=%.2f\n",
                    ctx->car.budget_per_layer * ctx->n_layers,
                    ctx->car.budget_per_layer, ctx->car.free_ratio);
        } else if (ctx->car.max_subs_per_layer > 0) {
            fprintf(stderr, "GPU: CAR budget: max %d subs/layer (best-ratio-first)%s\n",
                    ctx->car.max_subs_per_layer,
                    ctx->car.layer_weight ? " [layer-weighted]" : "");
        }
        if (ctx->car.skip_renorm)
            fprintf(stderr, "GPU: CAR dampening enabled (skip renormalization)\n");
        if (ctx->car_protect_layers > 0)
            fprintf(stderr, "GPU: CAR layer protection: first/last %d layers exempt\n",
                    ctx->car_protect_layers);
    }

    // CAR warmup: force CAR=1.0 (no substitutions) for first N tokens to seed cache
    {
        const char *warmup_env = getenv("QMOE_CAR_WARMUP");
        if (warmup_env) {
            ctx->car.warmup_tokens = atoi(warmup_env);
            if (ctx->car.warmup_tokens > 0)
                fprintf(stderr, "GPU: CAR warmup: %d tokens at CAR=1.0\n", ctx->car.warmup_tokens);
        }
    }

    // --- Initialize CAR backfill worker ---
    // Loads substituted experts into RAM cache during idle NVMe cycles.
    memset(&ctx->backfill, 0, sizeof(ctx->backfill));
    ctx->backfill.work_done = true;
    if (m->nvme_io && ctx->rcache.max_slots > 0 &&
        ctx->car.threshold > 0.0f && ctx->car.threshold < 1.0f &&
        !getenv("QMOE_NO_BACKFILL")) {
        ctx->backfill.nvme_io = m->nvme_io;
        ctx->backfill.rcache = &ctx->rcache;
        ctx->backfill.expert_stride = ctx->expert_stride;
        const char *bf_env = getenv("QMOE_BACKFILL_N");
        int bf_n = bf_env ? atoi(bf_env) : 4;
        if (bf_n < 1) bf_n = 1;
        if (bf_n > 16) bf_n = 16;
        ctx->backfill.batch_max = bf_n;
        ctx->backfill.no_priority = (getenv("QMOE_NO_PRIORITY_BACKFILL") != nullptr);
        ctx->backfill.sync_mode = (getenv("QMOE_BACKFILL_SYNC") != nullptr);
        // Dedup bitmap: 1 bit per (layer, expert) pair
        ctx->backfill.dedup_n_experts = ctx->n_experts_total;
        size_t dedup_bits = (size_t)ctx->n_layers * ctx->n_experts_total;
        size_t dedup_words = (dedup_bits + 63) / 64;
        ctx->backfill.dedup = (uint64_t *)calloc(dedup_words, sizeof(uint64_t));
        bool alloc_ok = ctx->backfill.dedup != nullptr;
        for (int i = 0; i < bf_n && alloc_ok; i++) {
            if (posix_memalign(&ctx->backfill.buffers[i], 4096, ctx->expert_stride) != 0)
                alloc_ok = false;
        }
        if (alloc_ok) {
            pthread_mutex_init(&ctx->backfill.mutex, nullptr);
            pthread_cond_init(&ctx->backfill.cond_work, nullptr);
            pthread_cond_init(&ctx->backfill.cond_done, nullptr);
            ctx->backfill.active = true;
            pthread_create(&ctx->backfill.thread, nullptr, backfill_worker_fn, ctx);
            fprintf(stderr, "GPU: CAR backfill worker started (batch=%d, queue=%d, dedup=%zuKB%s)\n",
                    bf_n, 512, dedup_words * 8 / 1024,
                    ctx->backfill.sync_mode ? ", SYNC: once-per-token" : "");
        } else {
            fprintf(stderr, "GPU: CAR backfill buffer alloc failed, disabled\n");
            for (int i = 0; i < bf_n; i++) { free(ctx->backfill.buffers[i]); ctx->backfill.buffers[i] = nullptr; }
        }
    }

    // CPU expert compute: allocate pinned host + device buffers
    ctx->h_norm = nullptr;
    ctx->h_cpu_partial = nullptr;
    ctx->d_cpu_partial = nullptr;
    ctx->cpu_expert = nullptr;
    {
        hipError_t e1 = hipHostMalloc(&ctx->h_norm, ctx->n_embd * sizeof(float), hipHostMallocDefault);
        hipError_t e2 = hipHostMalloc(&ctx->h_cpu_partial, ctx->n_embd * sizeof(float), hipHostMallocDefault);
        hipError_t e3 = hipMalloc(&ctx->d_cpu_partial, ctx->n_embd * sizeof(float));
        if (e1 == hipSuccess && e2 == hipSuccess && e3 == hipSuccess) {
            ctx->cpu_expert = cpu_expert_create(ctx->n_embd, ctx->expert_intermediate);
            if (ctx->cpu_expert)
                fprintf(stderr, "GPU: CPU expert compute enabled (n_embd=%d, inter=%d)\n",
                        ctx->n_embd, ctx->expert_intermediate);
            else
                fprintf(stderr, "GPU: WARNING: cpu_expert_create failed\n");
        } else {
            fprintf(stderr, "GPU: WARNING: CPU expert buffer alloc failed\n");
            if (ctx->h_norm) { hipHostFree(ctx->h_norm); ctx->h_norm = nullptr; }
            if (ctx->h_cpu_partial) { hipHostFree(ctx->h_cpu_partial); ctx->h_cpu_partial = nullptr; }
            if (ctx->d_cpu_partial) { hipFree(ctx->d_cpu_partial); ctx->d_cpu_partial = nullptr; }
        }
    }

    return 0;
}

// ============================================================
// PUBLIC API: gpu_seed_vram_cache
// Seed VRAM cache from frequency profile at startup.
// Loads top experts per layer from NVMe → VRAM cache.
// ============================================================

extern "C" int gpu_seed_vram_cache(gpu_ctx_t *ctx, void *model_ptr, const char *freq_profile_path) {
    if (!ctx || !ctx->model_uploaded || !model_ptr || !freq_profile_path)
        return -1;

    // Check if ANY VRAM cache exists (single-GPU, expert-split, ping-pong, or stage)
    bool have_vram = ctx->ecache.max_slots > 0;
    if (ctx->n_devices >= 2) {
        have_vram = have_vram || ctx->gpu1.ecache.max_slots > 0;
        if (ctx->pingpong) {
            have_vram = ctx->dev[0].ecache.max_slots > 0 || ctx->dev[1].ecache.max_slots > 0;
        }
    }
    if (!have_vram) return 0;

    model_t *m = (model_t *)model_ptr;
    nvme_io_t *io = m->nvme_io;
    if (!io) {
        fprintf(stderr, "gpu_seed: no NVMe I/O — cannot seed\n");
        return -1;
    }

    freq_profile_t *fp = freq_profile_load(freq_profile_path);
    if (!fp) return -1;

    int n_layers = ctx->n_layers;
    uint64_t stride = ctx->expert_stride;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int seeded0 = 0, seeded1 = 0;

    if (ctx->pingpong && ctx->n_devices >= 2) {
        // Ping-pong: layer L lives on dev[L & 1]. Seed into that device's ecache.
        // VRAM cache uses device-local layer indices: local = global / 2
        int spl[2] = { ctx->dev[0].ecache.slots_per_layer,
                       ctx->dev[1].ecache.slots_per_layer };
        int seeded[2] = {0, 0};

        for (int L = 0; L < n_layers; L++) {
            int d = L & 1;
            int cache_L = L / 2;  // device-local layer index
            vram_cache_t *ec = &ctx->dev[d].ecache;
            int want = spl[d];
            if (want > 256) want = 256;
            if (want <= 0) continue;
            freq_entry_t top[256];
            int n_top = freq_profile_top_for_layer(fp, L, top, want);

            for (int i = 0; i < n_top && i < spl[d]; i++) {
                int eid = top[i].expert_id;
                int slot = vram_cache_alloc_slot(ec, cache_L, eid);
                void *buf = nullptr;
                if (nvme_io_load_experts_at(io, L, &eid, 1, 0, &buf) != 0) continue;
                hipMemcpy((uint8_t *)ec->d_buf + (size_t)slot * stride,
                          buf, stride, hipMemcpyHostToDevice);
                seeded[d]++;
            }
        }
        seeded0 = seeded[0]; seeded1 = seeded[1];

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "gpu_seed: pingpong dev[0]=%d/%d, dev[1]=%d/%d VRAM slots seeded (%.1f ms)\n",
                seeded0, (n_layers / 2 + (n_layers & 1)) * spl[0],
                seeded1, (n_layers / 2) * spl[1], ms);
    } else {
        // Single-GPU or expert-split: seed into ctx->ecache / ctx->gpu1.ecache
        int spl0 = ctx->ecache.slots_per_layer;
        int spl1 = (ctx->n_devices >= 2) ? ctx->gpu1.ecache.slots_per_layer : 0;

        for (int L = 0; L < n_layers; L++) {
            int want = spl0 + spl1;
            if (want > 256) want = 256;
            freq_entry_t top[256];
            int n_top = freq_profile_top_for_layer(fp, L, top, want);

            for (int i = 0; i < n_top && i < spl0; i++) {
                int eid = top[i].expert_id;
                int slot = L * spl0 + i;
                void *buf = nullptr;
                if (nvme_io_load_experts_at(io, L, &eid, 1, 0, &buf) != 0) continue;
                (void)hipSetDevice(0);
                hipMemcpy((uint8_t *)ctx->ecache.d_buf + (size_t)slot * stride,
                          buf, stride, hipMemcpyHostToDevice);
                ctx->ecache.slot_layer[slot] = L;
                ctx->ecache.slot_expert[slot] = eid;
                ctx->ecache.slot_ts[slot] = ++ctx->ecache.ts;
                ctx->ecache.map[L * ctx->ecache.n_experts + eid] = slot;
                seeded0++;
            }

            if (spl1 > 0) {
                for (int i = spl0; i < n_top && (i - spl0) < spl1; i++) {
                    int eid = top[i].expert_id;
                    int slot = L * spl1 + (i - spl0);
                    void *buf = nullptr;
                    if (nvme_io_load_experts_at(io, L, &eid, 1, 0, &buf) != 0) continue;
                    (void)hipSetDevice(ctx->gpu1.device_id);
                    hipMemcpy((uint8_t *)ctx->gpu1.ecache.d_buf + (size_t)slot * stride,
                              buf, stride, hipMemcpyHostToDevice);
                    ctx->gpu1.ecache.slot_layer[slot] = L;
                    ctx->gpu1.ecache.slot_expert[slot] = eid;
                    ctx->gpu1.ecache.slot_ts[slot] = ++ctx->gpu1.ecache.ts;
                    ctx->gpu1.ecache.map[L * ctx->gpu1.ecache.n_experts + eid] = slot;
                    seeded1++;
                }
            }
        }

        (void)hipSetDevice(0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ctx->n_devices >= 2) {
            fprintf(stderr, "gpu_seed: GPU0=%d/%d, GPU1=%d/%d VRAM slots seeded (%.1f ms)\n",
                    seeded0, n_layers * spl0, seeded1, n_layers * spl1, ms);
        } else {
            fprintf(stderr, "gpu_seed: seeded %d/%d VRAM cache slots from %s (%.1f ms)\n",
                    seeded0, n_layers * spl0, freq_profile_path, ms);
        }
    }

    // Also seed RAM cache if available
    int ram_seeded = 0;
    if (ctx->rcache.max_slots > 0) {
        int rspl = ctx->rcache.slots_per_layer;
        // Determine total VRAM slots per layer to skip
        int vram_total;
        if (ctx->pingpong && ctx->n_devices >= 2) {
            // In pingpong, each layer only lives on one device
            // Use the max of the two devices' spl as a conservative skip count
            vram_total = (ctx->dev[0].ecache.slots_per_layer > ctx->dev[1].ecache.slots_per_layer)
                       ? ctx->dev[0].ecache.slots_per_layer : ctx->dev[1].ecache.slots_per_layer;
        } else {
            int spl0 = ctx->ecache.slots_per_layer;
            int spl1 = (ctx->n_devices >= 2) ? ctx->gpu1.ecache.slots_per_layer : 0;
            vram_total = spl0 + spl1;
        }
        for (int L = 0; L < n_layers; L++) {
            int want = vram_total + rspl;
            if (want > 256) want = 256;
            freq_entry_t top[256];
            int n_top = freq_profile_top_for_layer(fp, L, top, want);

            for (int i = vram_total; i < n_top && (i - vram_total) < rspl; i++) {
                int eid = top[i].expert_id;
                int rslot = L * rspl + (i - vram_total);
                void *buf = nullptr;
                if (nvme_io_load_experts_at(io, L, &eid, 1, 0, &buf) != 0) continue;
                memcpy((uint8_t *)ctx->rcache.buf + (size_t)rslot * stride, buf, stride);
                ctx->rcache.slot_layer[rslot] = L;
                ctx->rcache.slot_expert[rslot] = eid;
                ctx->rcache.slot_ts[rslot] = ++ctx->rcache.ts;
                ctx->rcache.map[L * ctx->rcache.n_experts + eid] = rslot;
                ram_seeded++;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "gpu_seed: seeded %d RAM cache slots (%.1f ms total)\n",
                ram_seeded, ms);
    }

    freq_profile_free(fp);
    return seeded0 + seeded1 + ram_seeded;
}

// ============================================================
// Expert frequency counting API
// ============================================================

extern "C" void gpu_enable_expert_freq(gpu_ctx_t *ctx) {
    if (!ctx || ctx->freq_enabled) return;
    int n = ctx->n_layers * ctx->n_experts_total;
    ctx->expert_freq = (uint32_t *)calloc(n, sizeof(uint32_t));
    ctx->freq_enabled = (ctx->expert_freq != nullptr);
    if (ctx->freq_enabled)
        fprintf(stderr, "GPU: expert frequency counting enabled (%d layers × %d experts)\n",
                ctx->n_layers, ctx->n_experts_total);
}

extern "C" const uint32_t *gpu_get_expert_freq(gpu_ctx_t *ctx) {
    if (!ctx || !ctx->freq_enabled) return nullptr;
    return ctx->expert_freq;
}

extern "C" void gpu_get_expert_dims(gpu_ctx_t *ctx, int *n_layers, int *n_experts) {
    if (!ctx) return;
    if (n_layers) *n_layers = ctx->n_layers;
    if (n_experts) *n_experts = ctx->n_experts_total;
}

// ============================================================
// Helper: queue attention/SSM + residual + router on GPU stream
// All work is async — returns immediately after kernel launches.
// Called before the layer loop for layer 0, and at the end of
// each iteration for layer il+1, so GPU runs attention during
// CPU cache work (making the e[4] sync nearly free).
// ============================================================
static void queue_attn_and_router(gpu_ctx_t *ctx, int il, int kv_idx, int ssm_idx, int pos) {
    hipStream_t s = ctx->stream;
    struct gpu_layer *gl = &ctx->layers[il];
    int n_embd = ctx->n_embd;
    int n_heads = ctx->n_heads;
    int n_kv_heads = ctx->n_kv_heads;
    int head_dim = ctx->head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // --- Attention / SSM ---
    if (ctx->debug_level > 0) hipEventRecord(ctx->prof.e[2], s);

    // RMS norm: d_norm = rms_norm(d_x, attn_norm)
    kernel_rms_norm<<<1, 256, 0, s>>>(
        ctx->d_x, gl->d_attn_norm, ctx->d_norm, n_embd, ctx->rms_norm_eps);

    if (gl->attn_type == LAYER_ATTN_FULL) {
        // ---- Gated Full GQA Attention ----
        int inner_dim = n_heads * head_dim;

        // Quantize d_norm for Q8_0 projections
        quantize_input_q8(s, ctx->d_norm, n_embd,
                          ctx->d_input_q8, ctx->d_input_q8_scales);

        // Q projection: [full_q_dim] = [2 * inner_dim] with interleaved Q+gate
        dispatch_matvec_q8(s, gl->d_wq, gl->wq_type, ctx->d_norm,
                           ctx->d_input_q8, ctx->d_input_q8_scales,
                           ctx->d_q, ctx->full_q_dim, n_embd);
        // K, V projections
        dispatch_matvec_q8(s, gl->d_wk, gl->wk_type, ctx->d_norm,
                           ctx->d_input_q8, ctx->d_input_q8_scales,
                           ctx->d_k, kv_dim, n_embd);
        dispatch_matvec_q8(s, gl->d_wv, gl->wv_type, ctx->d_norm,
                           ctx->d_input_q8, ctx->d_input_q8_scales,
                           ctx->d_v, kv_dim, n_embd);

        // Deinterleave Q+gate: [Q_h0,Gate_h0,...] → Q[inner_dim], gate[inner_dim]
        size_t deinter_smem = ctx->full_q_dim * sizeof(float);
        kernel_deinterleave_q_gate<<<1, 256, deinter_smem, s>>>(
            ctx->d_q, ctx->d_attn_gate_buf, n_heads, head_dim);
        if (il == 0) {
            hipError_t kerr = hipStreamSynchronize(s);
            if (kerr != hipSuccess)
                fprintf(stderr, "GPU: deinterleave kernel error: %s\n", hipGetErrorString(kerr));
            kerr = hipGetLastError();
            if (kerr != hipSuccess)
                fprintf(stderr, "GPU: deinterleave launch error: %s\n", hipGetErrorString(kerr));
        }

        // Per-head RMS norm on Q and K (head_dim=256 needs 256 threads)
        kernel_per_head_rms_norm<<<n_heads, 256, 0, s>>>(
            ctx->d_q, gl->d_q_norm, head_dim, ctx->rms_norm_eps);
        kernel_per_head_rms_norm<<<n_kv_heads, 256, 0, s>>>(
            ctx->d_k, gl->d_k_norm, head_dim, ctx->rms_norm_eps);

        // Partial NEOX RoPE
        kernel_rope<<<1, 256, 0, s>>>(
            ctx->d_q, ctx->d_k, pos, head_dim, ctx->rope_dim,
            n_heads, n_kv_heads, ctx->rope_freq_base);

        // KV cache update (use kv_idx, not il)
        kernel_kv_update<<<1, 256, 0, s>>>(
            ctx->d_kv_k, ctx->d_kv_v, ctx->d_k, ctx->d_v,
            kv_dim, kv_idx, pos, ctx->ctx_len);

        // GQA attention (FP16 KV cache, 3-phase)
        int seq_len = pos + 1;
        size_t smem_bytes = (size_t)seq_len * sizeof(float);
        kernel_gqa_attention<<<n_heads, 256, smem_bytes, s>>>(
            ctx->d_q, ctx->d_kv_k, ctx->d_kv_v, ctx->d_attn_out,
            kv_idx, pos, head_dim, n_heads, n_kv_heads, ctx->ctx_len);

        // Apply output gate: attn_out *= sigmoid(gate)
        kernel_sigmoid_gate<<<(inner_dim + 255) / 256, 256, 0, s>>>(
            ctx->d_attn_out, ctx->d_attn_gate_buf, inner_dim);

        // Output projection: [inner_dim] → [n_embd]
        quantize_input_q8(s, ctx->d_attn_out, inner_dim,
                          ctx->d_input_q8, ctx->d_input_q8_scales);
        dispatch_matvec_q8(s, gl->d_wo, gl->wo_type, ctx->d_attn_out,
                           ctx->d_input_q8, ctx->d_input_q8_scales,
                           ctx->d_ffn_buf, n_embd, inner_dim);

    } else {
        // ---- DeltaNet/SSM ----
        gpu_deltanet_ssm(ctx, gl, il, ssm_idx, s);
    }

    // Fused: residual add + RMS norm (2 kernels → 1)
    // d_x += d_ffn_buf; d_norm = rms_norm(d_x, ffn_norm)
    kernel_residual_rms_norm<<<1, 256, 0, s>>>(
        ctx->d_x, ctx->d_ffn_buf, gl->d_ffn_norm,
        ctx->d_norm, n_embd, ctx->rms_norm_eps);
    if (ctx->debug_level > 0) hipEventRecord(ctx->prof.e[3], s);

    // Router: d_router_out = router @ d_norm
    dispatch_matvec(s, gl->d_router, GGML_TYPE_F32_ID, ctx->d_norm,
                    ctx->d_router_out, ctx->n_experts_total, n_embd);

    // GPU softmax + top-k + D2H
    kernel_softmax<<<1, 256, 0, s>>>(ctx->d_router_out, ctx->n_experts_total);
    // D2H full softmax scores for cache-aware routing (1-2 KB)
    (void)hipMemcpyAsync(ctx->h_all_scores, ctx->d_router_out,
                          ctx->n_experts_total * sizeof(float), hipMemcpyDeviceToHost, s);
    kernel_top_k_parallel<<<1, 32, 0, s>>>(
        ctx->d_router_out, ctx->n_experts_total, ctx->n_experts_used,
        ctx->d_top_k_ids, ctx->d_top_k_vals);
    (void)hipMemcpyAsync(ctx->h_top_k_ids, ctx->d_top_k_ids,
                          ctx->n_experts_used * sizeof(int), hipMemcpyDeviceToHost, s);
    (void)hipMemcpyAsync(ctx->h_top_k_vals, ctx->d_top_k_vals,
                          ctx->n_experts_used * sizeof(float), hipMemcpyDeviceToHost, s);
    // D2H hidden state for CPU expert compute (16 KB — negligible DMA cost)
    if (ctx->h_norm) {
        (void)hipMemcpyAsync(ctx->h_norm, ctx->d_norm,
                              ctx->n_embd * sizeof(float), hipMemcpyDeviceToHost, s);
    }
    hipEventRecord(ctx->prof.e[4], s);

    // SharedFFN: queue on compute stream to fill GPU idle time during CPU sync.
    // SharedFFN depends only on d_norm (set by router norm above) + shared weights.
    // By running here, it overlaps with CPU hipEventSync + classify at loop top.
    // Note: SharedFFN already overlaps with H2D (on transfer_stream) since they use
    // different hardware (SMs vs DMA). A separate shared_stream was tested but provides
    // zero benefit — the overlap algebra is identical.
    if (gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down) {
        int se_inter = ctx->shared_expert_intermediate;
        if (gl->shared_gate_type == GGML_TYPE_Q4_K_ID &&
            gl->shared_up_type == GGML_TYPE_Q4_K_ID) {
            // Fused gate+up+SiLU with multi-warp (4 warps/row) for Q4_K
            // M=1024 with RPB=2 → 512 blocks → 4/SM → 32 warps/SM on 128-SM GPU
            constexpr int NW = 4, RPB = 2;
            dim3 blk(32 * NW, RPB);
            dim3 grd((se_inter + RPB - 1) / RPB);
            int smem = RPB * NW * 2 * (int)sizeof(float);
            kernel_q4k_fused_gus_mw<<<grd, blk, smem, s>>>(
                gl->d_shared_gate, gl->d_shared_up,
                ctx->d_norm, ctx->d_shared_gate_buf,
                se_inter, n_embd);
        } else {
            // Fallback: separate gate + up + silu_mul for non-Q4_K types
            dispatch_matvec(s, gl->d_shared_gate, gl->shared_gate_type,
                            ctx->d_norm, ctx->d_shared_gate_buf, se_inter, n_embd);
            dispatch_matvec(s, gl->d_shared_up, gl->shared_up_type,
                            ctx->d_norm, ctx->d_shared_up_buf, se_inter, n_embd);
            kernel_silu_mul<<<(se_inter + 255) / 256, 256, 0, s>>>(
                ctx->d_shared_gate_buf, ctx->d_shared_up_buf, se_inter);
        }
        dispatch_matvec(s, gl->d_shared_down, gl->shared_down_type,
                        ctx->d_shared_gate_buf, ctx->d_shared_down_buf, n_embd, se_inter);
        if (gl->d_shared_expert_gate) {
            kernel_dot_sigmoid_scale<<<1, 256, 0, s>>>(
                ctx->d_shared_down_buf, ctx->d_norm,
                gl->d_shared_expert_gate, n_embd, n_embd);
        }
    }
    if (ctx->debug_level > 0) hipEventRecord(ctx->prof.e[11], s);  // SharedFFN end
}

// Helper: consume async prefetch NVMe results (non-blocking).
// If NVMe done: H2D to VRAM cache, commit RAM map, record event. Returns true.
// If not done: returns false (caller can try again later).
static bool consume_prefetch_nvme(gpu_ctx_t *ctx) {
    bool pf_done = prefetch_try_complete_nvme(ctx->pf);
    if (!pf_done) return false;

    int pf_vslots[32], pf_rslots[32];
    int n_nvme_h2d = prefetch_get_nvme_transfers(ctx->pf,
        pf_vslots, pf_rslots, 32);
    uint64_t stride_pf = ctx->expert_stride;
    hipStream_t ts_pf = ctx->transfer_stream;
    for (int i = 0; i < n_nvme_h2d; i++) {
        (void)hipMemcpyAsync(
            (uint8_t *)ctx->ecache.d_buf + (size_t)pf_vslots[i] * stride_pf,
            (uint8_t *)ctx->rcache.buf + (size_t)pf_rslots[i] * stride_pf,
            stride_pf, hipMemcpyHostToDevice, ts_pf);
    }
    if (n_nvme_h2d > 0) ctx->prof.spec_prefetched += n_nvme_h2d;
    prefetch_mark_all_h2d_done(ctx->pf);
    prefetch_commit_ram_cache(ctx->pf, &ctx->rcache);

    int total_vpf = prefetch_vram_transfer_count(ctx->pf);
    if (total_vpf > 0) {
        (void)hipEventRecord(ctx->vram_prefetch_event, ctx->transfer_stream);
        ctx->vram_prefetch_active = true;
        ctx->prof.vram_prefetched += total_vpf;
    }
    return true;
}

// ============================================================
// PING-PONG: DeltaNet SSM using per-device buffers
// Same logic as gpu_deltanet_ssm but with explicit buffer pointers
// ============================================================

static void pp_deltanet_ssm(gpu_ctx_t *ctx, gpu_dev_t *dc,
                              struct gpu_layer *gl, int il,
                              int ssm_idx, hipStream_t s) {
    int n_embd = ctx->n_embd;
    int qkv_dim = ctx->ssm_qkv_dim;
    int inner = ctx->ssm_inner_size;
    int sd = ctx->ssm_state_size;
    int ng = ctx->ssm_group_count;
    int nh = ctx->ssm_time_step_rank;
    int vd = ctx->ssm_value_dim;
    int conv_k = ctx->ssm_conv_kernel;
    int kd = ng * sd;  // 2048

    // Quantize d_norm once for all Q8_0 matvecs sharing this input
    quantize_input_q8(s, dc->d_norm, n_embd,
                      dc->d_input_q8, dc->d_input_q8_scales);

    // Step 1: QKV projection
    dispatch_matvec_q8(s, gl->d_attn_qkv, gl->attn_qkv_type, dc->d_norm,
                       dc->d_input_q8, dc->d_input_q8_scales,
                       dc->d_q, qkv_dim, n_embd);

    // Step 2: Conv1d + SiLU
    int conv_steps = conv_k - 1;
    float *conv_buf = dc->d_ssm_conv_bufs +
                       (size_t)ssm_idx * conv_steps * qkv_dim;
    int conv_pos = dc->h_ssm_conv_pos[ssm_idx];

    kernel_ssm_conv1d_silu<<<(qkv_dim + 255) / 256, 256, 0, s>>>(
        dc->d_q, conv_buf, gl->d_ssm_conv1d, conv_pos, qkv_dim, conv_k);
    dc->h_ssm_conv_pos[ssm_idx] = conv_pos + 1;

    // Step 3: Split QKV and L2-normalize Q, K
    float *d_Q = dc->d_q;
    float *d_K = dc->d_q + kd;
    float *d_V = dc->d_q + kd + kd;
    kernel_l2_norm_groups<<<ng, 128, 0, s>>>(d_Q, d_K, sd);

    // Step 4: Decay and beta
    dispatch_matvec_q8(s, gl->d_ssm_alpha, gl->ssm_alpha_type, dc->d_norm,
                       dc->d_input_q8, dc->d_input_q8_scales,
                       dc->d_ssm_dt, nh, n_embd);
    dispatch_matvec_q8(s, gl->d_ssm_beta, gl->ssm_beta_type, dc->d_norm,
                       dc->d_input_q8, dc->d_input_q8_scales,
                       dc->d_ssm_beta_buf, nh, n_embd);

    bool has_bias = (gl->d_ssm_dt_bias != nullptr);
    kernel_decay_beta<<<1, nh, 0, s>>>(
        dc->d_ssm_dt, dc->d_ssm_beta_buf,
        gl->d_ssm_a, gl->d_ssm_dt_bias, nh, has_bias);

    // Step 5: State update + output
    size_t state_per_layer = (size_t)nh * sd * vd;
    float *d_state = dc->d_ssm_states + (size_t)ssm_idx * state_per_layer;
    float q_scale = 1.0f / sqrtf((float)sd);

    kernel_deltanet_state_update<<<nh, vd, 0, s>>>(
        d_state, d_Q, d_K, d_V,
        dc->d_ssm_dt, dc->d_ssm_beta_buf,
        dc->d_attn_out, sd, vd, ng, q_scale);

    // Step 6: Per-head RMS norm
    kernel_ssm_rms_norm<<<nh, vd, 0, s>>>(
        dc->d_attn_out, gl->d_ssm_norm, vd, ctx->rms_norm_eps);

    // Step 7: Output gate (SiLU)
    dispatch_matvec_q8(s, gl->d_attn_gate, gl->attn_gate_type, dc->d_norm,
                       dc->d_input_q8, dc->d_input_q8_scales,
                       dc->d_attn_gate_buf, inner, n_embd);
    kernel_silu_gate<<<(inner + 255) / 256, 256, 0, s>>>(
        dc->d_attn_out, dc->d_attn_gate_buf, inner);

    // Step 8: Output projection [inner] -> [n_embd]
    quantize_input_q8(s, dc->d_attn_out, inner,
                      dc->d_input_q8, dc->d_input_q8_scales);
    dispatch_matvec_q8(s, gl->d_ssm_out, gl->ssm_out_type, dc->d_attn_out,
                       dc->d_input_q8, dc->d_input_q8_scales,
                       dc->d_ffn_buf, n_embd, inner);
}

// ============================================================
// PING-PONG: Attention + router using per-device buffers
// Same logic as queue_attn_and_router but with explicit dev pointers
// ============================================================

static void pp_queue_attn_and_router(gpu_ctx_t *ctx, gpu_dev_t *dc,
                                       int il, int kv_idx, int ssm_idx, int pos) {
    hipStream_t s = dc->stream;
    struct gpu_layer *gl = &ctx->layers[il];
    int n_embd = ctx->n_embd;
    int n_heads = ctx->n_heads;
    int n_kv_heads = ctx->n_kv_heads;
    int head_dim = ctx->head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // --- Attention / SSM ---
    hipEventRecord(dc->prof_e[2], s);

    // RMS norm: d_norm = rms_norm(d_x, attn_norm)
    kernel_rms_norm<<<1, 256, 0, s>>>(
        dc->d_x, gl->d_attn_norm, dc->d_norm, n_embd, ctx->rms_norm_eps);

    if (gl->attn_type == LAYER_ATTN_FULL) {
        // ---- Gated Full GQA Attention ----
        int inner_dim = n_heads * head_dim;

        quantize_input_q8(s, dc->d_norm, n_embd,
                          dc->d_input_q8, dc->d_input_q8_scales);

        // Q projection
        dispatch_matvec_q8(s, gl->d_wq, gl->wq_type, dc->d_norm,
                           dc->d_input_q8, dc->d_input_q8_scales,
                           dc->d_q, ctx->full_q_dim, n_embd);
        // K, V projections
        dispatch_matvec_q8(s, gl->d_wk, gl->wk_type, dc->d_norm,
                           dc->d_input_q8, dc->d_input_q8_scales,
                           dc->d_k, kv_dim, n_embd);
        dispatch_matvec_q8(s, gl->d_wv, gl->wv_type, dc->d_norm,
                           dc->d_input_q8, dc->d_input_q8_scales,
                           dc->d_v, kv_dim, n_embd);

        // Deinterleave Q+gate
        size_t deinter_smem = ctx->full_q_dim * sizeof(float);
        kernel_deinterleave_q_gate<<<1, 256, deinter_smem, s>>>(
            dc->d_q, dc->d_attn_gate_buf, n_heads, head_dim);

        // Per-head RMS norm on Q and K
        kernel_per_head_rms_norm<<<n_heads, 256, 0, s>>>(
            dc->d_q, gl->d_q_norm, head_dim, ctx->rms_norm_eps);
        kernel_per_head_rms_norm<<<n_kv_heads, 256, 0, s>>>(
            dc->d_k, gl->d_k_norm, head_dim, ctx->rms_norm_eps);

        // Partial NEOX RoPE
        kernel_rope<<<1, 256, 0, s>>>(
            dc->d_q, dc->d_k, pos, head_dim, ctx->rope_dim,
            n_heads, n_kv_heads, ctx->rope_freq_base);

        // KV cache update
        kernel_kv_update<<<1, 256, 0, s>>>(
            dc->d_kv_k, dc->d_kv_v, dc->d_k, dc->d_v,
            kv_dim, kv_idx, pos, ctx->ctx_len);

        // GQA attention
        int seq_len = pos + 1;
        size_t smem_bytes = (size_t)seq_len * sizeof(float);
        kernel_gqa_attention<<<n_heads, 256, smem_bytes, s>>>(
            dc->d_q, dc->d_kv_k, dc->d_kv_v, dc->d_attn_out,
            kv_idx, pos, head_dim, n_heads, n_kv_heads, ctx->ctx_len);

        // Apply output gate
        kernel_sigmoid_gate<<<(inner_dim + 255) / 256, 256, 0, s>>>(
            dc->d_attn_out, dc->d_attn_gate_buf, inner_dim);

        // Output projection
        quantize_input_q8(s, dc->d_attn_out, inner_dim,
                          dc->d_input_q8, dc->d_input_q8_scales);
        dispatch_matvec_q8(s, gl->d_wo, gl->wo_type, dc->d_attn_out,
                           dc->d_input_q8, dc->d_input_q8_scales,
                           dc->d_ffn_buf, n_embd, inner_dim);

    } else {
        // ---- DeltaNet/SSM ----
        pp_deltanet_ssm(ctx, dc, gl, il, ssm_idx, s);
    }

    // Fused: residual add + RMS norm
    kernel_residual_rms_norm<<<1, 256, 0, s>>>(
        dc->d_x, dc->d_ffn_buf, gl->d_ffn_norm,
        dc->d_norm, n_embd, ctx->rms_norm_eps);
    hipEventRecord(dc->prof_e[3], s);

    // Router
    dispatch_matvec(s, gl->d_router, GGML_TYPE_F32_ID, dc->d_norm,
                    dc->d_router_out, ctx->n_experts_total, n_embd);

    // GPU softmax + top-k + D2H
    kernel_softmax<<<1, 256, 0, s>>>(dc->d_router_out, ctx->n_experts_total);
    (void)hipMemcpyAsync(dc->h_all_scores, dc->d_router_out,
                          ctx->n_experts_total * sizeof(float), hipMemcpyDeviceToHost, s);
    kernel_top_k_parallel<<<1, 32, 0, s>>>(
        dc->d_router_out, ctx->n_experts_total, ctx->n_experts_used,
        dc->d_top_k_ids, dc->d_top_k_vals);
    (void)hipMemcpyAsync(dc->h_top_k_ids, dc->d_top_k_ids,
                          ctx->n_experts_used * sizeof(int), hipMemcpyDeviceToHost, s);
    (void)hipMemcpyAsync(dc->h_top_k_vals, dc->d_top_k_vals,
                          ctx->n_experts_used * sizeof(float), hipMemcpyDeviceToHost, s);
    // D2H hidden state for expert classification
    (void)hipMemcpyAsync(dc->h_norm, dc->d_norm,
                          ctx->n_embd * sizeof(float), hipMemcpyDeviceToHost, s);
    hipEventRecord(dc->prof_e[4], s);

    // SharedFFN
    if (gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down) {
        int se_inter = ctx->shared_expert_intermediate;
        if (gl->shared_gate_type == GGML_TYPE_Q4_K_ID &&
            gl->shared_up_type == GGML_TYPE_Q4_K_ID) {
            constexpr int NW = 4, RPB = 2;
            dim3 blk(32 * NW, RPB);
            dim3 grd((se_inter + RPB - 1) / RPB);
            int smem = RPB * NW * 2 * (int)sizeof(float);
            kernel_q4k_fused_gus_mw<<<grd, blk, smem, s>>>(
                gl->d_shared_gate, gl->d_shared_up,
                dc->d_norm, dc->d_shared_gate_buf,
                se_inter, n_embd);
        } else {
            dispatch_matvec(s, gl->d_shared_gate, gl->shared_gate_type,
                            dc->d_norm, dc->d_shared_gate_buf, se_inter, n_embd);
            dispatch_matvec(s, gl->d_shared_up, gl->shared_up_type,
                            dc->d_norm, dc->d_shared_up_buf, se_inter, n_embd);
            kernel_silu_mul<<<(se_inter + 255) / 256, 256, 0, s>>>(
                dc->d_shared_gate_buf, dc->d_shared_up_buf, se_inter);
        }
        dispatch_matvec(s, gl->d_shared_down, gl->shared_down_type,
                        dc->d_shared_gate_buf, dc->d_shared_down_buf, n_embd, se_inter);
        if (gl->d_shared_expert_gate) {
            kernel_dot_sigmoid_scale<<<1, 256, 0, s>>>(
                dc->d_shared_down_buf, dc->d_norm,
                gl->d_shared_expert_gate, n_embd, n_embd);
        }
    }
    hipEventRecord(dc->prof_e[11], s);
}

// ============================================================
// PING-PONG: Spec routing worker thread
// Runs CPU-side router matvec + RAM→VRAM prefetch in background,
// so the main loop never blocks on hipEventSynchronize.
// ============================================================

static void *pp_spec_worker_fn(void *arg) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)arg;
    auto *w = &ctx->spec_worker;

    while (true) {
        pthread_mutex_lock(&w->mutex);
        while (!w->work_ready && !w->shutdown)
            pthread_cond_wait(&w->cond, &w->mutex);
        if (w->shutdown) { pthread_mutex_unlock(&w->mutex); break; }
        w->work_ready = false;
        w->work_done = false;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        pthread_mutex_unlock(&w->mutex);

        // Wait for GPU to finish D2H
        (void)hipEventSynchronize(w->gpu_event);

        // CPU: RMS norm
        float spec_buf[4096];
        rms_norm(spec_buf, w->h_input, w->ffn_norm, w->n_embd, w->rms_eps);

        // CPU: router matvec (f32, n_experts × n_embd → n_experts scores)
        float router_scores[512];
        for (int e = 0; e < w->n_experts; e++) {
            float dot = 0.0f;
            const float *row = w->router + (size_t)e * w->n_embd;
            for (int j = 0; j < w->n_embd; j++)
                dot += row[j] * spec_buf[j];
            router_scores[e] = dot;
        }

        // CPU: softmax + top-k
        softmax(router_scores, w->n_experts);
        w->n_pred = w->n_used;
        for (int k = 0; k < w->n_pred; k++) {
            int best = 0;
            for (int e = 1; e < w->n_experts; e++)
                if (router_scores[e] > router_scores[best]) best = e;
            w->pred_ids[k] = best;
            router_scores[best] = -1.0f;
        }

        // Save predictions for accuracy tracking in main loop
        memcpy(ctx->spec_ids[w->next_il], w->pred_ids, w->n_pred * sizeof(int));
        ctx->spec_n[w->next_il] = w->n_pred;
        ctx->spec_valid[w->next_il] = true;

        // Prefetch predicted misses → next GPU's VRAM cache
        // RAM cache hits get H2D'd directly. For experts not in any cache,
        // save their IDs in spec_nvme_ids so the main loop can read them from
        // NVMe during the next layer's attention (when NVMe is idle).
        int pf_eids[16];
        void *pf_srcs[16];
        int n_to_pf = 0;
        // Device-local layer index for VRAM cache (cache uses n_layers=30)
        int next_cache_il = w->next_il / 2;

        // Reset spec NVMe request list
        ctx->spec_nvme_n[w->next_il] = 0;

        for (int k = 0; k < w->n_pred; k++) {
            int eid = w->pred_ids[k];
            int slot = w->next_ec->map[next_cache_il * w->next_ec->n_experts + eid];
            if (slot >= 0) {
                vram_cache_touch(w->next_ec, slot);
                continue;
            }
            // Check RAM cache — uses global layer ID
            int rslot = -1;
            if (w->rcache->max_slots > 0)
                rslot = w->rcache->map[w->next_il * w->rcache->n_experts + eid];
            if (rslot >= 0) {
                pf_eids[n_to_pf] = eid;
                pf_srcs[n_to_pf] = (uint8_t *)w->rcache->buf + (size_t)rslot * w->expert_stride;
                ram_cache_touch(w->rcache, rslot);
                n_to_pf++;
            } else if (w->nvme_io && ctx->spec_nvme_n[w->next_il] < 8) {
                // Not in VRAM or RAM — tell main loop to read from NVMe
                ctx->spec_nvme_ids[w->next_il][ctx->spec_nvme_n[w->next_il]++] = eid;
            }
        }

        // Save deferred prefetch list for main thread (VRAM alloc is not thread-safe)
        for (int j = 0; j < n_to_pf; j++) {
            w->pf_eids[j] = pf_eids[j];
            w->pf_srcs[j] = pf_srcs[j];
        }
        w->n_pf = n_to_pf;
        w->n_prefetched = n_to_pf;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        w->work_done = true;
    }
    return nullptr;
}

// ============================================================
// PING-PONG: Forward pass with alternating GPUs
// Even layers on dev[0], odd layers on dev[1]
// ============================================================

static float *gpu_forward_pingpong(gpu_ctx_t *ctx, void *model_ptr, int token_id, int pos) {
    model_t *m = (model_t *)model_ptr;
    const model_hparams_t *hp = &m->hparams;

    int n_embd = ctx->n_embd;
    int n_layers = ctx->n_layers;

    struct timespec gpu_fwd_wall_start;
    clock_gettime(CLOCK_MONOTONIC, &gpu_fwd_wall_start);

    // Reset per-token profiling accumulators
    ctx->prof.embed_ms = 0;
    ctx->prof.attn_ms = 0;
    ctx->prof.attn_full_ms = 0;
    ctx->prof.attn_ssm_ms = 0;
    ctx->prof.router_ms = 0;
    ctx->prof.nvme_ms = 0;
    ctx->prof.moe_pipe_ms = 0;
    ctx->prof.output_ms = 0;
    ctx->prof.rcwait_ms = 0;
    ctx->prof.hipsync_ms = 0;
    ctx->prof.classify_ms = 0;
    ctx->prof.gpu_ffn_ms = 0;
    ctx->prof.h2d_wait_ms = 0;
    ctx->prof.phase1_ffn_ms = 0;
    ctx->prof.shared_ffn_ms = 0;
    ctx->prof.post_ffn_ms = 0;
    ctx->prof.nvme_wait_ms = 0;
    ctx->prof.ecache_d2d_ms = 0;
    ctx->prof.spec_pf_ms = 0;
    ctx->prof.overlap_total = 0;
    ctx->prof.overlap_hits = 0;
    ctx->prof.spec_total = 0;
    ctx->prof.spec_hits = 0;
    ctx->prof.spec2_total = 0;
    ctx->prof.spec2_hits = 0;
    ctx->prof.spec_prefetched = 0;
    ctx->prof.vram_prefetched = 0;
    ctx->prof.car_substitutions = 0;
    ctx->prof.car_nvme_avoided = 0;
    ctx->prof.car_nvme_remaining = 0;
    ctx->prof.car_free_subs = 0;
    ctx->prof.car_score_ratio_sum = 0.0;
    ctx->prof.cold_fetch_dropped = 0;
    ctx->prof.cpu_expert_ms = 0;
    ctx->prof.cpu_expert_count = 0;
    ctx->prof.gpu1_expert_count = 0;
    ctx->prof.gpu1_wait_ms = 0;
    memset(ctx->spec_valid, 0, sizeof(ctx->spec_valid));
    memset(ctx->spec2_valid, 0, sizeof(ctx->spec2_valid));
    memset(ctx->spec_nvme_n, 0, sizeof(ctx->spec_nvme_n));
    car_begin_token(&ctx->car);

    int max_layers_gpu = n_layers;
    {
        const char *ll = getenv("QMOE_MAX_LAYERS");
        if (ll) {
            max_layers_gpu = atoi(ll);
            if (max_layers_gpu > n_layers) max_layers_gpu = n_layers;
        }
    }

    // Pre-compute per-layer KV/SSM indices within each device
    // For layer il on device d = il & 1:
    //   pp_kv_idx[il] = index among full-attn layers on that device
    //   pp_ssm_idx[il] = index among SSM layers on that device
    int pp_kv_idx[MAX_LAYERS], pp_ssm_idx[MAX_LAYERS];
    {
        int ki[2] = {0, 0}, si[2] = {0, 0};
        for (int il = 0; il < max_layers_gpu; il++) {
            int d = il & 1;
            if (ctx->layers[il].attn_type == LAYER_ATTN_FULL) {
                pp_kv_idx[il] = ki[d]++;
                pp_ssm_idx[il] = 0;
            } else {
                pp_kv_idx[il] = 0;
                pp_ssm_idx[il] = si[d]++;
            }
        }
    }

    // ---- Embedding lookup on GPU0 (layer 0 is even) ----
    gpu_dev_t *dc0 = &ctx->dev[0];
    hipStream_t s0 = dc0->stream;
    (void)hipSetDevice(0);
    hipEventRecord(dc0->prof_e[0], s0);
    if (ctx->d_token_embd && ctx->token_embd_type == GGML_TYPE_Q4_K_ID) {
        int nb = n_embd / QK_K;
        kernel_embed_lookup_q4k<<<(nb + 255) / 256, 256, 0, s0>>>(
            ctx->d_token_embd, dc0->d_x, token_id, n_embd);
    } else if (ctx->d_token_embd && ctx->token_embd_type == GGML_TYPE_Q8_0_ID) {
        int nb = n_embd / QK8_0;
        kernel_embed_lookup_q8_0<<<(nb + 255) / 256, 256, 0, s0>>>(
            ctx->d_token_embd, dc0->d_x, token_id, n_embd);
    } else {
        (void)hipMemcpyAsync(dc0->d_x, m->buf_x, n_embd * sizeof(float),
                              hipMemcpyHostToDevice, s0);
    }
    hipEventRecord(dc0->prof_e[1], s0);

    // Pre-queue layer 0's attention + router
    if (max_layers_gpu > 0) {
        (void)hipSetDevice(0);
        pp_queue_attn_and_router(ctx, dc0, 0, pp_kv_idx[0], pp_ssm_idx[0], pos);
    }

    // ---- Layer loop ----
    for (int il = 0; il < max_layers_gpu; il++) {
        int d = il & 1;
        gpu_dev_t *dc = &ctx->dev[d];
        struct gpu_layer *gl = &ctx->layers[il];
        hipStream_t s = dc->stream;
        hipStream_t ts = dc->transfer_stream;
        // Local layer index for this device's cache (0..29)
        // Cache was initialized with n_layers=30 (device-local), not 60
        int cache_il = il / 2;

        (void)hipSetDevice(d);

        // Ensure previous layer's async H2D from RAM cache is complete before
        // backfill can evict RAM slots. With pinned rcache (no bounce buffer),
        // hipMemcpyAsync reads directly from rcache.buf pages — if backfill
        // evicts a slot the DMA engine is still reading, we get data corruption.
        if (il > 0 && ctx->rcache.pinned && !ctx->h2d_bounce) {
            (void)hipStreamSynchronize(ctx->dev[1 - d].transfer_stream);
        }

        // Commit any completed backfill reads to RAM cache (non-blocking check).
        // This runs during the attention/SSM sync stall, so it's free.
        // In sync mode, backfill_dispatch_sync handles commit once per token.
        if (!ctx->backfill.sync_mode)
            backfill_try_complete(ctx);

        // Note: Spec NVMe prefetch was tested but adds NVMe latency to critical
        // path that exceeds the VRAM hit rate improvement. The spec worker does
        // RAM→VRAM prefetch (fast, non-blocking H2D) which provides ~1-3% boost.

        // Sync: wait for attention + router results to be on host
        {
            struct timespec sync_t0, sync_t1;
            clock_gettime(CLOCK_MONOTONIC, &sync_t0);
            (void)hipEventSynchronize(dc->prof_e[4]);
            clock_gettime(CLOCK_MONOTONIC, &sync_t1);
            ctx->prof.hipsync_ms += (sync_t1.tv_sec - sync_t0.tv_sec) * 1000.0 +
                                     (sync_t1.tv_nsec - sync_t0.tv_nsec) / 1e6;
        }

        // (spec NVMe prefetch collection was here — disabled, see note above)

        // Query GPU-side timings
        {
            float ms;
            if (il == 0) {
                hipEventElapsedTime(&ms, dc0->prof_e[0], dc0->prof_e[1]);
                ctx->prof.embed_ms = ms;
            }
            hipEventElapsedTime(&ms, dc->prof_e[2], dc->prof_e[3]);
            ctx->prof.attn_ms += ms;
            if (gl->attn_type == LAYER_ATTN_FULL)
                ctx->prof.attn_full_ms += ms;
            else
                ctx->prof.attn_ssm_ms += ms;
            hipEventElapsedTime(&ms, dc->prof_e[3], dc->prof_e[4]);
            ctx->prof.router_ms += ms;
            // Deferred MoE timing from previous iteration on same device
            if (il >= 2) {
                hipEventElapsedTime(&ms, dc->prof_e[5], dc->prof_e[7]);
                ctx->prof.moe_pipe_ms += ms;
                hipEventElapsedTime(&ms, dc->prof_e[4], dc->prof_e[11]);
                ctx->prof.shared_ffn_ms += ms;
                hipEventElapsedTime(&ms, dc->prof_e[5], dc->prof_e[8]);
                ctx->prof.phase1_ffn_ms += ms;
                hipEventElapsedTime(&ms, dc->prof_e[8], dc->prof_e[9]);
                ctx->prof.h2d_wait_ms += ms;
                hipEventElapsedTime(&ms, dc->prof_e[9], dc->prof_e[10]);
                ctx->prof.gpu_ffn_ms += ms;
                hipEventElapsedTime(&ms, dc->prof_e[10], dc->prof_e[7]);
                ctx->prof.post_ffn_ms += ms;
            }
        }

        // Cross-GPU prefetch moved to after Phase 1 FFN (needs partial MoE output)

        // --- MoE expert pipeline ---
        hipEventRecord(dc->prof_e[5], s);

        if (!m->nvme_io) {
            fprintf(stderr, "gpu_forward_pingpong: no NVMe I/O configured\n");
            return nullptr;
        }

        int n_used = ctx->n_experts_used;
        int M_gate = ctx->expert_intermediate;
        int down_type = m->layers[il].down_type;
        uint64_t down_offset = ctx->expert_gate_size + ctx->expert_up_size;
        uint64_t stride = ctx->expert_stride;

        int buf_set = il & 1;  // NOTE: each device sees alternating il values (0,2,4... or 1,3,5...)
                                // buf_set = il & 1 = d, so always same for a given device. Use il/2 & 1 instead.
        int dev_buf_set = (il / 2) & 1;
        size_t stg_base = (size_t)dev_buf_set * n_used * stride;
        int nvme_buf_off = dev_buf_set * n_used;

        struct timespec sp_t0, sp_t1;
        clock_gettime(CLOCK_MONOTONIC, &sp_t0);

        // Track speculative routing prediction accuracy
        if (ctx->spec_valid[il]) {
            for (int i = 0; i < n_used; i++) {
                int eid = dc->h_top_k_ids[i];
                ctx->prof.spec_total++;
                for (int j = 0; j < ctx->spec_n[il]; j++) {
                    if (ctx->spec_ids[il][j] == eid) {
                        ctx->prof.spec_hits++;
                        break;
                    }
                }
            }
        }

        // Expert classification: this device's VRAM cache + shared RAM cache
        int cached_stg[16], cached_slot[16], n_cached = 0;
        int ramhit_stg[16], ramhit_slot[16], n_ramhit = 0;
        int uncached_stg[16], uncached_ids[16], n_uncached = 0;

        for (int i = 0; i < n_used; i++) {
            int eid = dc->h_top_k_ids[i];
            // Check THIS device's VRAM cache (using device-local layer index)
            int slot = -1;
            if (dc->ecache.max_slots > 0) {
                slot = dc->ecache.map[cache_il * dc->ecache.n_experts + eid];
            }
            if (slot >= 0) {
                cached_stg[n_cached] = i;
                cached_slot[n_cached] = slot;
                dc->ecache.slot_ts[slot] = ++dc->ecache.ts;
                n_cached++;
                dc->ecache.hits++;
                // Keep RAM cache entry alive while expert is in VRAM (L2 behavior).
                // When expert is eventually evicted from VRAM, the RAM entry is
                // still fresh and can serve the next access — preventing NVMe reads.
                if (ctx->rcache.max_slots > 0) {
                    int rslot = ctx->rcache.map[il * ctx->rcache.n_experts + eid];
                    if (rslot >= 0)
                        ctx->rcache.slot_ts[rslot] = ++ctx->rcache.ts;
                }
            } else {
                // Check shared RAM cache
                int rslot = -1;
                if (ctx->rcache.max_slots > 0) {
                    rslot = ctx->rcache.map[il * ctx->rcache.n_experts + eid];
                }
                if (rslot >= 0) {
                    ramhit_stg[n_ramhit] = i;
                    ramhit_slot[n_ramhit] = rslot;
                    ctx->rcache.slot_ts[rslot] = ++ctx->rcache.ts;
                    n_ramhit++;
                    ctx->rcache.hits++;
                } else {
                    uncached_stg[n_uncached] = i;
                    uncached_ids[n_uncached] = eid;
                    n_uncached++;
                    dc->ecache.misses++;
                    ctx->rcache.misses++;
                }
            }
        }

        // Snapshot original expert IDs before CAR modifies them in-place.
        // Used for (1) correct frequency counting and (2) backfill enqueue.
        int orig_expert_ids[16];
        float orig_expert_scores[16];
        memcpy(orig_expert_ids, dc->h_top_k_ids, n_used * sizeof(int));
        memcpy(orig_expert_scores, dc->h_top_k_vals, n_used * sizeof(float));

        // Cache-aware routing (substitution)
        // During warmup, skip CAR so all experts load from NVMe (seeds the cache)
        bool in_warmup = (ctx->car.warmup_tokens > 0 &&
                          ctx->prof.token_count < ctx->car.warmup_tokens);
        if (ctx->car.threshold > 0.0f && !in_warmup && n_uncached > 0 &&
            il >= ctx->car_protect_layers && il < max_layers_gpu - ctx->car_protect_layers) {
            int vram_sub_idx[16], vram_sub_slot[16], n_vram_subs = 0;
            int vram1_sub_idx[16], vram1_sub_slot[16], n_vram1_subs = 0;
            int ram_sub_idx[16], ram_sub_slot[16], n_ram_subs = 0;

            int subs = car_evaluate(
                &ctx->car, cache_il, il,
                dc->h_top_k_ids, dc->h_top_k_vals, n_used,
                dc->h_all_scores, ctx->n_experts_total,
                &dc->ecache,
                nullptr,  // no second VRAM cache in pingpong
                &ctx->rcache,
                uncached_stg, &n_uncached,
                vram_sub_idx, vram_sub_slot, &n_vram_subs,
                vram1_sub_idx, vram1_sub_slot, &n_vram1_subs,
                ram_sub_idx, ram_sub_slot, &n_ram_subs
            );

            if (subs > 0) {
                ctx->prof.car_substitutions += subs;
                ctx->prof.car_nvme_avoided += n_vram_subs + n_ram_subs;
                ctx->prof.car_score_ratio_sum = ctx->car.score_ratio_sum;
                ctx->prof.car_free_subs = (int)ctx->car.free_subs;

                for (int j = 0; j < n_vram_subs; j++) {
                    cached_stg[n_cached] = vram_sub_idx[j];
                    cached_slot[n_cached] = vram_sub_slot[j];
                    vram_cache_touch(&dc->ecache, vram_sub_slot[j]);
                    n_cached++;
                }
                for (int j = 0; j < n_ram_subs; j++) {
                    ramhit_stg[n_ramhit] = ram_sub_idx[j];
                    ramhit_slot[n_ramhit] = ram_sub_slot[j];
                    ram_cache_touch(&ctx->rcache, ram_sub_slot[j]);
                    n_ramhit++;
                }

                for (int u = 0; u < n_uncached; u++)
                    uncached_ids[u] = dc->h_top_k_ids[uncached_stg[u]];

                // Enqueue substituted originals for background NVMe backfill.
                // Priority = original router score (higher = more wanted = load first).
                if (ctx->backfill.active) {
                    for (int i = 0; i < n_used; i++) {
                        if (orig_expert_ids[i] != dc->h_top_k_ids[i]) {
                            float priority = 0.0f;
                            if (!ctx->backfill.no_priority) {
                                float s_orig = dc->h_all_scores[orig_expert_ids[i]];
                                // Negate so highest scores sort to front of ascending queue.
                                priority = -s_orig;
                            }
                            backfill_enqueue(ctx, il, orig_expert_ids[i], priority);
                        }
                    }
                }
            }
            ctx->prof.car_nvme_remaining += n_uncached;
        }

        // No-cold-fetch: drop remaining uncached
        int pre_ncf_n_uncached = n_uncached;
        if (ctx->no_cold_fetch && n_uncached > 0) {
            for (int u = 0; u < n_uncached; u++)
                dc->h_top_k_vals[uncached_stg[u]] = 0.0f;
            float sum = 0.0f;
            for (int i = 0; i < n_used; i++) sum += dc->h_top_k_vals[i];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int i = 0; i < n_used; i++) dc->h_top_k_vals[i] *= inv;
            }
            ctx->prof.cold_fetch_dropped += n_uncached;
            n_uncached = 0;
        }

        // Routing trace: emit per-layer record
        if (ctx->trace.fp) {
            trace_record_t rec;
            memset(&rec, 0, sizeof(rec));
            rec.token_id = (uint32_t)ctx->prof.token_count;
            rec.layer_id = (uint16_t)il;
            rec.n_used = (uint8_t)n_used;
            int n_subs = 0;
            for (int i = 0; i < n_used; i++) {
                rec.orig_ids[i] = (int16_t)orig_expert_ids[i];
                rec.orig_scores[i] = orig_expert_scores[i];
                rec.final_ids[i] = (int16_t)dc->h_top_k_ids[i];
                rec.final_scores[i] = dc->h_top_k_vals[i];
                if (orig_expert_ids[i] != dc->h_top_k_ids[i]) n_subs++;
            }
            rec.n_substitutions = (uint8_t)n_subs;
            for (int j = 0; j < n_cached; j++)
                rec.source[cached_stg[j]] = TRACE_SRC_VRAM;
            for (int j = 0; j < n_ramhit; j++)
                rec.source[ramhit_stg[j]] = TRACE_SRC_RAM;
            for (int j = 0; j < pre_ncf_n_uncached; j++)
                rec.source[uncached_stg[j]] = TRACE_SRC_NVME;
            for (int i = 0; i < n_used; i++) {
                if (orig_expert_ids[i] != dc->h_top_k_ids[i])
                    rec.source[i] = TRACE_SRC_CAR;
                else if (dc->h_top_k_vals[i] == 0.0f)
                    rec.source[i] = TRACE_SRC_DROPPED;
            }
            if (ctx->trace.buf_used + sizeof(rec) > ctx->trace.buf_cap) {
                fwrite(ctx->trace.buf, 1, ctx->trace.buf_used, ctx->trace.fp);
                ctx->trace.buf_used = 0;
            }
            memcpy(ctx->trace.buf + ctx->trace.buf_used, &rec, sizeof(rec));
            ctx->trace.buf_used += sizeof(rec);
        }

        // Build ordered expert pointers for FFN
        int n_gpu_total = n_cached + n_ramhit + n_uncached;
        void *h_ptrs_ordered[16];
        float h_scores_ordered[16];
        int p2_eids[16];
        int n_p1 = 0, n_p2 = 0;

        for (int j = 0; j < n_cached; j++) {
            h_ptrs_ordered[n_p1] = (uint8_t *)dc->ecache.d_buf + (size_t)cached_slot[j] * stride;
            h_scores_ordered[n_p1] = dc->h_top_k_vals[cached_stg[j]];
            n_p1++;
        }
        for (int j = 0; j < n_ramhit; j++) {
            int p = n_p2++;
            h_ptrs_ordered[n_p1 + p] = (uint8_t *)dc->d_expert_staging + stg_base + (size_t)p * stride;
            h_scores_ordered[n_p1 + p] = dc->h_top_k_vals[ramhit_stg[j]];
            p2_eids[p] = dc->h_top_k_ids[ramhit_stg[j]];
        }
        int p2_nv_start = n_p2;
        for (int j = 0; j < n_uncached; j++) {
            int p = n_p2++;
            h_ptrs_ordered[n_p1 + p] = (uint8_t *)dc->d_expert_staging + stg_base + (size_t)p * stride;
            h_scores_ordered[n_p1 + p] = dc->h_top_k_vals[uncached_stg[j]];
            p2_eids[p] = uncached_ids[j];
        }

        // Upload pointers + scores
        memcpy(dc->h_expert_ptrs, h_ptrs_ordered, n_gpu_total * sizeof(void *));
        (void)hipMemcpyAsync(dc->d_expert_ptrs, dc->h_expert_ptrs,
                             n_gpu_total * sizeof(void *), hipMemcpyHostToDevice, s);
        memcpy(dc->h_top_k_vals, h_scores_ordered, n_gpu_total * sizeof(float));
        (void)hipMemcpyAsync(dc->d_scores, dc->h_top_k_vals,
                              n_gpu_total * sizeof(float), hipMemcpyHostToDevice, s);

        // PHASE 1 FFN: VRAM-cached experts
        if (n_p1 > 0) {
            constexpr int RPB = 8;
            dim3 wblock(32, RPB);
            dim3 grid_gu((M_gate + RPB - 1) / RPB, n_p1);
            if (ctx->gate_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    dc->d_expert_ptrs, dc->d_norm, dc->d_gate,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            } else {
                kernel_q6k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    dc->d_expert_ptrs, dc->d_norm, dc->d_gate,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            }
            dim3 grid_d((n_embd + RPB - 1) / RPB, n_p1);
            if (down_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    dc->d_expert_ptrs, dc->d_gate, dc->d_down,
                    n_embd, M_gate, down_offset);
            } else if (down_type == GGML_TYPE_Q5_K_ID) {
                kernel_q5k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    dc->d_expert_ptrs, dc->d_gate, dc->d_down,
                    n_embd, M_gate, down_offset);
            } else {
                kernel_q6k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    dc->d_expert_ptrs, dc->d_gate, dc->d_down,
                    n_embd, M_gate, down_offset);
            }
        }
        hipEventRecord(dc->prof_e[8], s);

        // === CPU-side spec routing + cross-GPU prefetch ===
        // After Phase 1 FFN, compute partial MoE output on GPU, D2H to CPU,
        // then predict next layer's experts using host-side router weights.
        // Prefetch predicted misses (NVMe → other GPU's VRAM cache) while
        // current layer's NVMe reads and Phase 2 FFN proceed.
        if (il + 1 < max_layers_gpu && m->nvme_io) {
            int next_il = il + 1;
            int next_d = 1 - d;
            gpu_dev_t *next_dc = &ctx->dev[next_d];
            vram_cache_t *next_ec = &next_dc->ecache;

            if (next_ec->max_slots > 0 && n_p1 > 0 && ctx->spec_worker.active) {
                int blk = (n_embd + 255) / 256;

                // GPU: partial weighted sum of Phase 1 experts + residual
                kernel_weighted_sum<<<blk, 256, 0, s>>>(
                    dc->d_down, dc->d_scores, dc->d_attn_out, n_embd, n_p1);
                if (gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down) {
                    kernel_vec_add<<<blk, 256, 0, s>>>(
                        dc->d_attn_out, dc->d_attn_out, dc->d_shared_down_buf, n_embd);
                }
                kernel_vec_add<<<blk, 256, 0, s>>>(
                    dc->d_attn_out, dc->d_attn_out, dc->d_x, n_embd);
                // D2H partial output (16KB)
                (void)hipMemcpyAsync(dc->h_norm, dc->d_attn_out,
                                      n_embd * sizeof(float), hipMemcpyDeviceToHost, s);
                (void)hipEventRecord(dc->spec_event, s);

                // Fire-and-forget: signal worker thread (no blocking)
                auto *w = &ctx->spec_worker;
                // Collect previous worker's results if any
                // (worker modifies prefetch_pending atomically, no explicit join needed)
                w->gpu_event = dc->spec_event;
                w->h_input = dc->h_norm;
                w->ffn_norm = m->layers[next_il].ffn_norm;
                w->router = m->layers[next_il].router;
                w->n_embd = n_embd;
                w->n_experts = ctx->n_experts_total;
                w->n_used = ctx->n_experts_used;
                w->rms_eps = ctx->rms_norm_eps;
                w->next_il = next_il;
                w->next_d = next_d;
                w->next_dc = next_dc;
                w->next_ec = next_ec;
                w->rcache = &ctx->rcache;
                w->expert_stride = ctx->expert_stride;
                w->nvme_io = m->nvme_io;

                pthread_mutex_lock(&w->mutex);
                w->work_ready = true;
                pthread_cond_signal(&w->cond);
                pthread_mutex_unlock(&w->mutex);
            }
        }

        // Reset classify timer after spec routing (so classify_ms only measures classification)
        clock_gettime(CLOCK_MONOTONIC, &sp_t1);
        ctx->prof.spec_pf_ms += (sp_t1.tv_sec - sp_t0.tv_sec) * 1000.0 +
                                  (sp_t1.tv_nsec - sp_t0.tv_nsec) / 1e6;
        clock_gettime(CLOCK_MONOTONIC, &sp_t0);

        // NVMe load for uncached experts
        struct timespec nvme_t0, nvme_t1;
        clock_gettime(CLOCK_MONOTONIC, &nvme_t0);
        bool nvme_bg_active = false;

        // The NVMe background loader uses ctx->nvme_bg which targets GPU0 by default.
        // For ping-pong, we need to point it at the correct device's staging buffer.
        if (n_uncached > 0 && ctx->nvme_bg.active) {
            ctx->nvme_bg.h2d_stream = ts;
            ctx->nvme_bg.d_staging = (uint8_t *)dc->d_expert_staging;
            ctx->nvme_bg.stg_base = stg_base;
            ctx->nvme_bg.nv_start = p2_nv_start;
            ctx->nvme_bg.stride = stride;
            ctx->nvme_bg.h2d_enabled = true;
            nvme_bg_start(ctx, m->nvme_io, il, uncached_ids, n_uncached, nvme_buf_off);
            nvme_bg_active = true;
        }

        // Double-buffer wait: ensure staging from 2 layers ago on this device is free
        clock_gettime(CLOCK_MONOTONIC, &sp_t1);
        ctx->prof.classify_ms += (sp_t1.tv_sec - sp_t0.tv_sec) * 1000.0 +
                                  (sp_t1.tv_nsec - sp_t0.tv_nsec) / 1e6;
        clock_gettime(CLOCK_MONOTONIC, &sp_t0);
        if (il >= 4) {
            // This device last used its staging 2 iterations ago (il-2)
            (void)hipEventSynchronize(dc->batch_event[dev_buf_set]);
        }
        // Sync deferred spec prefetch from previous iteration (DMA had full layer to overlap)
        if (dc->prefetch_pending) {
            (void)hipEventSynchronize(dc->prefetch_event);
            dc->prefetch_pending = false;
        }
        // Execute deferred spec prefetch: worker saved RAM-hit expert list,
        // main thread does VRAM alloc + DMA (thread-safe, single-threaded).
        // Non-blocking: only if worker already finished, skip otherwise.
        if (ctx->spec_worker.active && il > 0) {
            auto *w = &ctx->spec_worker;
            if (w->work_done) {
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                if (w->n_pf > 0) {
                    int pf_cache_il = (il) / 2;
                    for (int j = 0; j < w->n_pf; j++) {
                        int vslot = vram_cache_alloc_slot(&dc->ecache, pf_cache_il, w->pf_eids[j]);
                        (void)hipMemcpyAsync(
                            (uint8_t *)dc->ecache.d_buf + (size_t)vslot * ctx->expert_stride,
                            w->pf_srcs[j], ctx->expert_stride,
                            hipMemcpyHostToDevice, ts);
                    }
                    (void)hipEventRecord(dc->prefetch_event, ts);
                    dc->prefetch_pending = true;
                    ctx->prof.vram_prefetched += w->n_pf;
                    w->n_pf = 0;
                }
            }
        }
        rcache_pop_wait(ctx);

        // RAM hit H2D
        if (n_ramhit > 0) {
            void *ramhit_srcs[16];
            for (int j = 0; j < n_ramhit; j++) {
                ramhit_srcs[j] = (uint8_t *)ctx->rcache.buf + (size_t)ramhit_slot[j] * stride;
            }
            if (ctx->h2d_bounce && n_ramhit > 0) {
                // Use bounce buffer for coalesced transfer
                for (int j = 0; j < n_ramhit; j++) {
                    memcpy((uint8_t *)ctx->h2d_bounce + (size_t)j * stride,
                           ramhit_srcs[j], stride);
                }
                (void)hipMemcpyAsync(
                    (uint8_t *)dc->d_expert_staging + stg_base,
                    ctx->h2d_bounce,
                    (size_t)n_ramhit * stride, hipMemcpyHostToDevice, ts);
            } else {
                for (int j = 0; j < n_ramhit; j++) {
                    (void)hipMemcpyAsync(
                        (uint8_t *)dc->d_expert_staging + stg_base + (size_t)j * stride,
                        ramhit_srcs[j], stride, hipMemcpyHostToDevice, ts);
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &sp_t1);
        ctx->prof.rcwait_ms += (sp_t1.tv_sec - sp_t0.tv_sec) * 1000.0 +
                                (sp_t1.tv_nsec - sp_t0.tv_nsec) / 1e6;

        // Wait for NVMe
        void *expert_buffers[16];
        if (nvme_bg_active) {
            struct timespec nw_t0, nw_t1;
            clock_gettime(CLOCK_MONOTONIC, &nw_t0);
            nvme_bg_wait(ctx);
            clock_gettime(CLOCK_MONOTONIC, &nw_t1);
            ctx->prof.nvme_wait_ms += (nw_t1.tv_sec - nw_t0.tv_sec) * 1000.0 +
                                       (nw_t1.tv_nsec - nw_t0.tv_nsec) / 1e6;
            if (ctx->nvme_bg.result != 0) {
                fprintf(stderr, "gpu_forward_pingpong: NVMe failed layer %d\n", il);
                return nullptr;
            }
            memcpy(expert_buffers, ctx->nvme_bg.out_buffers, n_uncached * sizeof(void *));
        } else if (n_uncached > 0) {
            if (nvme_io_load_experts_at(m->nvme_io, il, uncached_ids,
                                         n_uncached, nvme_buf_off, expert_buffers) != 0) {
                fprintf(stderr, "gpu_forward_pingpong: NVMe failed layer %d\n", il);
                return nullptr;
            }
            for (int j = 0; j < n_uncached; j++) {
                (void)hipMemcpyAsync(
                    (uint8_t *)dc->d_expert_staging + stg_base + (size_t)(p2_nv_start + j) * stride,
                    expert_buffers[j], stride, hipMemcpyHostToDevice, ts);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &nvme_t1);
        ctx->prof.nvme_ms += (nvme_t1.tv_sec - nvme_t0.tv_sec) * 1000.0 +
                              (nvme_t1.tv_nsec - nvme_t0.tv_nsec) / 1e6;

        // Record event on transfer stream
        (void)hipEventRecord(dc->batch_event[dev_buf_set], ts);

        // PHASE 2 FFN: staged experts (after H2D completes)
        if (n_p2 > 0) {
            (void)hipStreamWaitEvent(s, dc->batch_event[dev_buf_set], 0);
            hipEventRecord(dc->prof_e[9], s);

            constexpr int RPB = 8;
            dim3 wblock(32, RPB);
            void **d_ptrs_p2 = dc->d_expert_ptrs + n_p1;
            float *d_gate_p2 = dc->d_gate + (size_t)n_p1 * M_gate;
            float *d_down_p2 = dc->d_down + (size_t)n_p1 * n_embd;

            dim3 grid_gu((M_gate + RPB - 1) / RPB, n_p2);
            if (ctx->gate_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    d_ptrs_p2, dc->d_norm, d_gate_p2,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            } else {
                kernel_q6k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    d_ptrs_p2, dc->d_norm, d_gate_p2,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            }
            dim3 grid_d((n_embd + RPB - 1) / RPB, n_p2);
            if (down_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            } else if (down_type == GGML_TYPE_Q5_K_ID) {
                kernel_q5k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            } else {
                kernel_q6k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            }
        } else {
            hipEventRecord(dc->prof_e[9], s);
        }

        // Fused: weighted_sum + shared_expert_add + residual
        {
            bool has_shared = gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down;
            kernel_moe_output_residual<<<(n_embd + 255) / 256, 256, 0, s>>>(
                dc->d_down, dc->d_scores,
                dc->d_shared_down_buf, dc->d_x,
                dc->d_x, n_embd, n_gpu_total, has_shared);
        }
        hipEventRecord(dc->prof_e[10], s);
        hipEventRecord(dc->prof_e[7], s);

        // VRAM cache update: Phase 2 experts
        int spl = dc->ecache.slots_per_layer;
        if (n_p2 > 0 && spl > 0) {
            for (int j = 0; j < n_p2; j++) {
                int slot = vram_cache_alloc_slot(&dc->ecache, cache_il, p2_eids[j]);
                (void)hipMemcpyAsync(
                    (uint8_t *)dc->ecache.d_buf + (size_t)slot * stride,
                    (uint8_t *)dc->d_expert_staging + stg_base + (size_t)j * stride,
                    stride, hipMemcpyDeviceToDevice, ts);
            }
        }

        // RAM cache update — skip when VRAM cache is active.
        // With freq-seeded RAM + L2 touch (VRAM hits keep RAM entries alive),
        // the seeded entries provide good coverage. Populating RAM during
        // generation causes expensive CPU memcpy into large pinned buffers,
        // adding ~15ms overhead per token. Only populate in fallback mode.
        int rspl = ctx->rcache.slots_per_layer;
        if (n_uncached > 0 && rspl > 0 && dc->ecache.slots_per_layer <= 0) {
            void *pop_srcs[16];
            int pop_slots[16];
            int pop_n = 0;
            for (int j = 0; j < n_uncached; j++) {
                int eid = uncached_ids[j];
                if (ctx->rcache.map[il * ctx->rcache.n_experts + eid] >= 0) continue;
                int rslot = ram_cache_alloc_slot(&ctx->rcache, il, eid);
                pop_srcs[pop_n] = expert_buffers[j];
                pop_slots[pop_n] = rslot;
                pop_n++;
            }
            if (pop_n > 0) {
                rcache_pop_start(ctx, pop_srcs, pop_slots, pop_n);
            }
        }

        // Expert frequency counting — use ORIGINAL router selections, not CAR
        // substitutions. This prevents a feedback loop where substitutes inflate
        // their own frequency counts and crowd out the true working set.
        if (ctx->freq_enabled) {
            for (int i = 0; i < n_used; i++)
                ctx->expert_freq[il * ctx->n_experts_total + orig_expert_ids[i]]++;
        }

        // Save current expert selections for next token's prefetch
        for (int i = 0; i < n_used; i++) {
            ctx->prefetch.prev_experts[il][i] = dc->h_top_k_ids[i];
        }

        // Dispatch backfill reads — NVMe will be idle during the next
        // layer's attention/SSM phase, so these reads don't contend.
        // In sync mode, backfill runs once per token after the layer loop.
        if (!ctx->backfill.sync_mode)
            backfill_dispatch(ctx);

        // === Relay d_x to next layer's GPU ===
        // After this layer's output residual, d_x is on device d.
        // Next layer is on device 1-d. D2H from d, H2D to 1-d.
        if (il + 1 < max_layers_gpu) {
            int next_d = 1 - d;
            gpu_dev_t *next_dc = &ctx->dev[next_d];

            // D2H: device d -> pinned host (use h_gpu1_norm as relay buffer)
            // Use event-based sync instead of CPU-blocking hipStreamSynchronize
            (void)hipMemcpyAsync(ctx->h_gpu1_norm, dc->d_x,
                                  n_embd * sizeof(float), hipMemcpyDeviceToHost, s);
            (void)hipEventRecord(dc->batch_event[dev_buf_set], s);

            // H2D: pinned host -> device 1-d
            // Next device's stream waits for D2H event before reading relay buffer
            (void)hipSetDevice(next_d);
            (void)hipStreamWaitEvent(next_dc->stream, dc->batch_event[dev_buf_set], 0);
            (void)hipMemcpyAsync(next_dc->d_x, ctx->h_gpu1_norm,
                                  n_embd * sizeof(float), hipMemcpyHostToDevice, next_dc->stream);

            // Pre-queue next layer's attention + router on the OTHER GPU
            pp_queue_attn_and_router(ctx, next_dc, il + 1,
                                      pp_kv_idx[il + 1], pp_ssm_idx[il + 1], pos);
        }
    }

    ctx->prefetch.has_prev = true;

    // Deterministic backfill: exactly batch_max experts, once per token
    backfill_dispatch_sync(ctx);

    // ---- Final output: LM head ----
    int last_d = (max_layers_gpu - 1) & 1;
    gpu_dev_t *last_dc = &ctx->dev[last_d];
    (void)hipSetDevice(last_d);
    hipStream_t last_s = last_dc->stream;

    hipEventRecord(last_dc->prof_e[2], last_s);

    // RMS norm
    kernel_rms_norm<<<1, 256, 0, last_s>>>(
        last_dc->d_x, last_dc->d_output_norm, last_dc->d_norm,
        n_embd, ctx->rms_norm_eps);

    // LM head: logits = output @ d_norm
    dispatch_matvec(last_s, last_dc->d_output, last_dc->output_type,
                    last_dc->d_norm, last_dc->d_logits,
                    ctx->vocab_size, n_embd);

    // D2H logits
    (void)hipMemcpyAsync(last_dc->h_logits, last_dc->d_logits,
                          ctx->vocab_size * sizeof(float), hipMemcpyDeviceToHost, last_s);
    hipEventRecord(last_dc->prof_e[3], last_s);
    (void)hipStreamSynchronize(last_s);

    // Query last layer's deferred MoE pipeline timing
    {
        float ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[5], last_dc->prof_e[7]);
        ctx->prof.moe_pipe_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[4], last_dc->prof_e[11]);
        ctx->prof.shared_ffn_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[5], last_dc->prof_e[8]);
        ctx->prof.phase1_ffn_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[8], last_dc->prof_e[9]);
        ctx->prof.h2d_wait_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[9], last_dc->prof_e[10]);
        ctx->prof.gpu_ffn_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[10], last_dc->prof_e[7]);
        ctx->prof.post_ffn_ms += ms;
        hipEventElapsedTime(&ms, last_dc->prof_e[2], last_dc->prof_e[3]);
        ctx->prof.output_ms = ms;
    }

    // Wall clock
    struct timespec gpu_fwd_wall_end;
    clock_gettime(CLOCK_MONOTONIC, &gpu_fwd_wall_end);
    double wall_ms = (gpu_fwd_wall_end.tv_sec - gpu_fwd_wall_start.tv_sec) * 1000.0 +
                     (gpu_fwd_wall_end.tv_nsec - gpu_fwd_wall_start.tv_nsec) / 1e6;

    // Print profiling summary
    ctx->prof.token_count++;
    double total = ctx->prof.embed_ms + ctx->prof.attn_ms + ctx->prof.router_ms +
                   ctx->prof.moe_pipe_ms + ctx->prof.output_ms;
    const char *dbg_env = getenv("QMOE_DEBUG");
    int dbg_level = dbg_env ? atoi(dbg_env) : 1;
    if (dbg_level > 0 && (ctx->prof.token_count <= 3 || ctx->prof.token_count % 10 == 0)) {
        fprintf(stderr,
            "\n[PingPong] tok=%d  pos=%d  layers=%d  gpu=%.1f ms  wall=%.1f ms (%.2f tok/s)\n"
            "  Embed:      %7.2f ms\n"
            "  Attention:  %7.2f ms  (%.2f ms/layer)\n"
            "  Router+D2H: %7.2f ms  (%.2f ms/layer)\n"
            "    Sync stall:%6.2f ms\n"
            "  MoE pipe:   %7.2f ms  (%.2f ms/layer)\n"
            "    P1 FFN:   %7.2f ms  (%.2f ms/layer)\n"
            "    H2D wait: %7.2f ms  (%.2f ms/layer)\n"
            "    P2 FFN:   %7.2f ms  (%.2f ms/layer)\n"
            "    NVMe wait:%7.2f ms  (%.2f ms/layer)\n"
            "    NVMe(CPU):%7.2f ms  (%.2f ms/layer)\n"
            "    SpecPF:   %7.2f ms  (%.2f ms/layer)\n"
            "    Classify: %7.2f ms  (%.2f ms/layer)\n"
            "    RC wait:  %7.2f ms  (%.2f ms/layer)\n"
            "    VRAM$[0]:  %llu hits, %llu misses\n"
            "    VRAM$[1]:  %llu hits, %llu misses\n"
            "    RAM$:      %llu hits, %llu misses\n"
            "    CAR:      %d subs, %d NVMe avoided, %d remaining, %d dropped\n"
            "    SpecRoute: %d/%d predicted (%.1f%%)\n"
            "    Prefetch:  %d experts (RAM→VRAM cross-GPU)\n"
            "    Backfill:  %llu queued, %llu loaded, %llu deduped, %llu skipped, %llu dropped\n"
            "  Output:     %7.2f ms\n",
            ctx->prof.token_count, pos, n_layers, total, wall_ms, 1000.0 / wall_ms,
            ctx->prof.embed_ms,
            ctx->prof.attn_ms, ctx->prof.attn_ms / n_layers,
            ctx->prof.router_ms, ctx->prof.router_ms / n_layers,
            ctx->prof.hipsync_ms,
            ctx->prof.moe_pipe_ms, ctx->prof.moe_pipe_ms / n_layers,
            ctx->prof.phase1_ffn_ms, ctx->prof.phase1_ffn_ms / n_layers,
            ctx->prof.h2d_wait_ms, ctx->prof.h2d_wait_ms / n_layers,
            ctx->prof.gpu_ffn_ms, ctx->prof.gpu_ffn_ms / n_layers,
            ctx->prof.nvme_wait_ms, ctx->prof.nvme_wait_ms / n_layers,
            ctx->prof.nvme_ms, ctx->prof.nvme_ms / n_layers,
            ctx->prof.spec_pf_ms, ctx->prof.spec_pf_ms / n_layers,
            ctx->prof.classify_ms, ctx->prof.classify_ms / n_layers,
            ctx->prof.rcwait_ms, ctx->prof.rcwait_ms / n_layers,
            (unsigned long long)ctx->dev[0].ecache.hits,
            (unsigned long long)ctx->dev[0].ecache.misses,
            (unsigned long long)ctx->dev[1].ecache.hits,
            (unsigned long long)ctx->dev[1].ecache.misses,
            (unsigned long long)ctx->rcache.hits,
            (unsigned long long)ctx->rcache.misses,
            ctx->prof.car_substitutions, ctx->prof.car_nvme_avoided,
            ctx->prof.car_nvme_remaining, ctx->prof.cold_fetch_dropped,
            ctx->prof.spec_hits, ctx->prof.spec_total,
            ctx->prof.spec_total > 0 ? 100.0 * ctx->prof.spec_hits / ctx->prof.spec_total : 0.0,
            ctx->prof.vram_prefetched,
            (unsigned long long)ctx->backfill.total_queued,
            (unsigned long long)ctx->backfill.total_backfilled,
            (unsigned long long)ctx->backfill.total_deduped,
            (unsigned long long)ctx->backfill.total_skipped,
            (unsigned long long)ctx->backfill.total_dropped,
            ctx->prof.output_ms);
    }

    (void)hipSetDevice(0);  // restore default
    return last_dc->h_logits;
}

// ============================================================
// PUBLIC API: gpu_forward
// Full forward pass: CPU only does NVMe expert I/O
// ============================================================

extern "C" float *gpu_forward(gpu_ctx_t *ctx, void *model_ptr, int token_id, int pos) {
    if (!ctx || !ctx->model_uploaded || !model_ptr) return nullptr;

    // Dispatch to ping-pong pipeline if enabled
    if (ctx->pingpong && ctx->n_devices >= 2) {
        return gpu_forward_pingpong(ctx, model_ptr, token_id, pos);
    }

    model_t *m = (model_t *)model_ptr;
    const model_hparams_t *hp = &m->hparams;
    hipStream_t s = ctx->stream;

    int n_embd = ctx->n_embd;
    int n_heads = ctx->n_heads;
    int n_kv_heads = ctx->n_kv_heads;
    int head_dim = ctx->head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // Wall clock for entire gpu_forward (includes CPU overhead between layers)
    struct timespec gpu_fwd_wall_start;
    clock_gettime(CLOCK_MONOTONIC, &gpu_fwd_wall_start);

    // Reset per-token profiling accumulators
    ctx->prof.embed_ms = 0;
    ctx->prof.attn_ms = 0;
    ctx->prof.attn_full_ms = 0;
    ctx->prof.attn_ssm_ms = 0;
    ctx->prof.router_ms = 0;
    ctx->prof.nvme_ms = 0;
    ctx->prof.moe_pipe_ms = 0;
    ctx->prof.output_ms = 0;
    ctx->prof.rcwait_ms = 0;
    ctx->prof.hipsync_ms = 0;
    ctx->prof.classify_ms = 0;
    ctx->prof.gpu_ffn_ms = 0;
    ctx->prof.h2d_wait_ms = 0;
    ctx->prof.phase1_ffn_ms = 0;
    ctx->prof.shared_ffn_ms = 0;
    ctx->prof.post_ffn_ms = 0;
    ctx->prof.nvme_wait_ms = 0;
    ctx->prof.ecache_d2d_ms = 0;
    ctx->prof.spec_pf_ms = 0;
    ctx->prof.overlap_total = 0;
    ctx->prof.overlap_hits = 0;
    ctx->prof.spec_total = 0;
    ctx->prof.spec_hits = 0;
    ctx->prof.spec2_total = 0;
    ctx->prof.spec2_hits = 0;
    ctx->prof.spec_prefetched = 0;
    ctx->prof.vram_prefetched = 0;
    ctx->prof.car_substitutions = 0;
    ctx->prof.car_nvme_avoided = 0;
    ctx->prof.car_nvme_remaining = 0;
    ctx->prof.car_free_subs = 0;
    ctx->prof.car_score_ratio_sum = 0.0;
    ctx->prof.cold_fetch_dropped = 0;
    ctx->prof.cpu_expert_ms = 0;
    ctx->prof.cpu_expert_count = 0;
    ctx->prof.gpu1_expert_count = 0;
    ctx->prof.gpu1_wait_ms = 0;
    memset(ctx->spec_valid, 0, sizeof(ctx->spec_valid));
    memset(ctx->spec2_valid, 0, sizeof(ctx->spec2_valid));
    memset(ctx->spec_nvme_n, 0, sizeof(ctx->spec_nvme_n));
    car_begin_token(&ctx->car);

    // ---- Embedding lookup ----
    hipEventRecord(ctx->prof.e[0], s);
    if (ctx->d_token_embd && ctx->token_embd_type == GGML_TYPE_Q4_K_ID) {
        int nb = n_embd / QK_K;
        kernel_embed_lookup_q4k<<<(nb + 255) / 256, 256, 0, s>>>(
            ctx->d_token_embd, ctx->d_x, token_id, n_embd);
    } else if (ctx->d_token_embd && ctx->token_embd_type == GGML_TYPE_Q8_0_ID) {
        int nb = n_embd / QK8_0;
        kernel_embed_lookup_q8_0<<<(nb + 255) / 256, 256, 0, s>>>(
            ctx->d_token_embd, ctx->d_x, token_id, n_embd);
    } else {
        // Fallback: CPU forward() has filled buf_x, upload it
        (void)hipMemcpyAsync(ctx->d_x, m->buf_x, n_embd * sizeof(float),
                              hipMemcpyHostToDevice, s);
    }
    hipEventRecord(ctx->prof.e[1], s);

    // ---- Layer loop ----
    int max_layers_gpu = ctx->n_layers;
    {
        const char *ll = getenv("QMOE_MAX_LAYERS");
        if (ll) {
            max_layers_gpu = atoi(ll);
            if (max_layers_gpu > ctx->n_layers) max_layers_gpu = ctx->n_layers;
        }
    }

    // Pre-compute per-layer KV/SSM indices (avoids tracking during pre-queued attention)
    int layer_kv_idx[MAX_LAYERS], layer_ssm_idx[MAX_LAYERS];
    {
        int ki = 0, si = 0;
        for (int il = 0; il < max_layers_gpu; il++) {
            if (ctx->layers[il].attn_type == LAYER_ATTN_FULL) {
                layer_kv_idx[il] = ki++;
                layer_ssm_idx[il] = 0;
            } else {
                layer_kv_idx[il] = 0;
                layer_ssm_idx[il] = si++;
            }
        }
    }

    // Pre-queue layer 0's attention + router on GPU stream.
    // GPU starts executing while we enter the loop and do first sync.
    if (max_layers_gpu > 0) {
        queue_attn_and_router(ctx, 0, layer_kv_idx[0], layer_ssm_idx[0], pos);
    }

    for (int il = 0; il < max_layers_gpu; il++) {
        struct gpu_layer *gl = &ctx->layers[il];

        // hipEventSync: waits for pre-queued attention+router results.
        // With pre-queuing, GPU started ATTN at end of previous iteration.
        // The spec_pf bg thread also runs during this sync, getting overlap.
        {
            struct timespec sync_t0, sync_t1;
            clock_gettime(CLOCK_MONOTONIC, &sync_t0);
            (void)hipEventSynchronize(ctx->prof.e[4]);
            clock_gettime(CLOCK_MONOTONIC, &sync_t1);
            ctx->prof.hipsync_ms += (sync_t1.tv_sec - sync_t0.tv_sec) * 1000.0 +
                                     (sync_t1.tv_nsec - sync_t0.tv_nsec) / 1e6;
        }

        // Query GPU-side timings (e[4] sync ensures all prior events completed)
        {
            float ms;
            if (il == 0) {
                hipEventElapsedTime(&ms, ctx->prof.e[0], ctx->prof.e[1]);
                ctx->prof.embed_ms = ms;
            }
            hipEventElapsedTime(&ms, ctx->prof.e[2], ctx->prof.e[3]);
            ctx->prof.attn_ms += ms;
            if (gl->attn_type == LAYER_ATTN_FULL) {
                ctx->prof.attn_full_ms += ms;
            } else {
                ctx->prof.attn_ssm_ms += ms;
            }
            hipEventElapsedTime(&ms, ctx->prof.e[3], ctx->prof.e[4]);
            ctx->prof.router_ms += ms;
            // Query PREVIOUS layer's deferred MoE pipeline time + sub-phases
            // SharedFFN is now pre-queued in helper: e[4]→e[11]
            // MoE pipe (Phase 1 + H2D + Phase 2 + post): e[5]→e[7]
            if (il > 0) {
                hipEventElapsedTime(&ms, ctx->prof.e[5], ctx->prof.e[7]);
                ctx->prof.moe_pipe_ms += ms;
                hipEventElapsedTime(&ms, ctx->prof.e[4], ctx->prof.e[11]);
                ctx->prof.shared_ffn_ms += ms;
                hipEventElapsedTime(&ms, ctx->prof.e[5], ctx->prof.e[8]);
                ctx->prof.phase1_ffn_ms += ms;
                hipEventElapsedTime(&ms, ctx->prof.e[8], ctx->prof.e[9]);
                ctx->prof.h2d_wait_ms += ms;
                hipEventElapsedTime(&ms, ctx->prof.e[9], ctx->prof.e[10]);
                ctx->prof.gpu_ffn_ms += ms;
                hipEventElapsedTime(&ms, ctx->prof.e[10], ctx->prof.e[7]);
                ctx->prof.post_ffn_ms += ms;
            }
        }

        // --- PASS 1: Consume async prefetch from previous layer ---
        // Non-blocking check. If NVMe done: H2D to VRAM cache + commit RAM map.
        // If not done: leave pf_pending=true for Pass 2 (after Phase 1 FFN).
        if (ctx->pf_pending) {
            if (consume_prefetch_nvme(ctx))
                ctx->pf_pending = false;
        }

        // --- Cache-aware MoE with speculative prefetch ---
        // SharedFFN already pre-queued by queue_attn_and_router() (overlaps with CPU sync)
        // e[5] marks start of Phase 1 FFN (after SharedFFN + CPU classify)
        hipEventRecord(ctx->prof.e[5], s);  // Phase 1 FFN start

        if (!m->nvme_io) {
            fprintf(stderr, "gpu_forward: no NVMe I/O configured\n");
            return nullptr;
        }

        int n_used = ctx->n_experts_used;
        int M_gate = ctx->expert_intermediate;
        int down_type = m->layers[il].down_type;
        uint64_t down_offset = ctx->expert_gate_size + ctx->expert_up_size;
        uint64_t stride = ctx->expert_stride;
        hipStream_t ts = ctx->transfer_stream;

        // Collect prefetch results from nvme_bg_worker's piggybacked reads.
        // Data is available after nvme_bg_wait at previous layer completed.
        // Double-buffered staging: alternate GPU staging + NVMe buffers per layer
        int buf_set = il & 1;
        size_t stg_base = (size_t)buf_set * n_used * stride;  // GPU staging offset
        int nvme_buf_off = buf_set * n_used;                    // NVMe buffer pool offset

        // Sub-phase timing
        struct timespec sp_t0, sp_t1;
        clock_gettime(CLOCK_MONOTONIC, &sp_t0);

        // Track speculative routing prediction accuracy
        if (ctx->spec_valid[il]) {
            for (int i = 0; i < n_used; i++) {
                int eid = ctx->h_top_k_ids[i];
                ctx->prof.spec_total++;
                for (int j = 0; j < ctx->spec_n[il]; j++) {
                    if (ctx->spec_ids[il][j] == eid) {
                        ctx->prof.spec_hits++;
                        break;
                    }
                }
            }
        }
        // Track L+2 speculative routing prediction accuracy
        if (ctx->spec2_valid[il]) {
            for (int i = 0; i < n_used; i++) {
                int eid = ctx->h_top_k_ids[i];
                ctx->prof.spec2_total++;
                for (int j = 0; j < ctx->spec2_n[il]; j++) {
                    if (ctx->spec2_ids[il][j] == eid) {
                        ctx->prof.spec2_hits++;
                        break;
                    }
                }
            }
        }
        // GPU1 expert engine toggle (QMOE_NO_GPU1=1 to disable for benchmarking)
        static int use_gpu1 = -1;
        if (use_gpu1 < 0) use_gpu1 = (ctx->n_devices >= 2 && !getenv("QMOE_NO_GPU1"));

        // 4-way classify: GPU0 VRAM / GPU1 VRAM / RAM cache / NVMe miss
        int cached_stg[16], cached_slot[16], n_cached = 0;
        int gpu1_stg[16], gpu1_slot[16], n_gpu1 = 0;
        int ramhit_stg[16], ramhit_slot[16], n_ramhit = 0;
        int uncached_stg[16], uncached_ids[16], n_uncached = 0;

        for (int i = 0; i < n_used; i++) {
            int eid = ctx->h_top_k_ids[i];
            // Check GPU0 VRAM cache
            int slot = -1;
            if (ctx->ecache.max_slots > 0) {
                slot = ctx->ecache.map[il * ctx->ecache.n_experts + eid];
            }
            if (slot >= 0) {
                cached_stg[n_cached] = i;
                cached_slot[n_cached] = slot;
                ctx->ecache.slot_ts[slot] = ++ctx->ecache.ts;
                n_cached++;
                ctx->ecache.hits++;
                // Keep RAM cache entry alive (L2 behavior)
                if (ctx->rcache.max_slots > 0) {
                    int rslot = ctx->rcache.map[il * ctx->rcache.n_experts + eid];
                    if (rslot >= 0)
                        ctx->rcache.slot_ts[rslot] = ++ctx->rcache.ts;
                }
            } else {
                // Check GPU1 VRAM cache
                int slot1 = -1;
                if (use_gpu1 && ctx->gpu1.ecache.max_slots > 0) {
                    slot1 = ctx->gpu1.ecache.map[il * ctx->gpu1.ecache.n_experts + eid];
                }
                if (slot1 >= 0) {
                    gpu1_stg[n_gpu1] = i;
                    gpu1_slot[n_gpu1] = slot1;
                    ctx->gpu1.ecache.slot_ts[slot1] = ++ctx->gpu1.ecache.ts;
                    n_gpu1++;
                    ctx->gpu1.ecache.hits++;
                } else {
                    // Check RAM cache
                    int rslot = -1;
                    if (ctx->rcache.max_slots > 0) {
                        rslot = ctx->rcache.map[il * ctx->rcache.n_experts + eid];
                    }
                    if (rslot >= 0) {
                        ramhit_stg[n_ramhit] = i;
                        ramhit_slot[n_ramhit] = rslot;
                        ctx->rcache.slot_ts[rslot] = ++ctx->rcache.ts;
                        n_ramhit++;
                        ctx->rcache.hits++;
                    } else {
                        uncached_stg[n_uncached] = i;
                        uncached_ids[n_uncached] = eid;
                        n_uncached++;
                        ctx->ecache.misses++;
                        ctx->rcache.misses++;
                    }
                }
            }
        }

        // Measure: how many NVMe misses would token-persistence predict?
        if (ctx->prefetch.has_prev && n_uncached > 0) {
            for (int u = 0; u < n_uncached; u++) {
                int eid = uncached_ids[u];
                for (int j = 0; j < n_used; j++) {
                    if (ctx->prefetch.prev_experts[il][j] == eid) {
                        ctx->prof.overlap_hits++;
                        break;
                    }
                }
            }
            ctx->prof.overlap_total += n_uncached;
        }

        // --- Cache-aware routing (v2 module): substitute NVMe misses with cached experts ---
        // Adaptive threshold: protect first/last N layers from CAR
        // During warmup, skip CAR so all experts load from NVMe (seeds the cache)
        bool in_warmup = (ctx->car.warmup_tokens > 0 &&
                          ctx->prof.token_count < ctx->car.warmup_tokens);
        if (ctx->car.threshold > 0.0f && !in_warmup && n_uncached > 0 &&
            il >= ctx->car_protect_layers && il < max_layers_gpu - ctx->car_protect_layers) {
            int vram_sub_idx[16], vram_sub_slot[16], n_vram_subs = 0;
            int vram1_sub_idx[16], vram1_sub_slot[16], n_vram1_subs = 0;
            int ram_sub_idx[16], ram_sub_slot[16], n_ram_subs = 0;

            int subs = car_evaluate(
                &ctx->car, il, -1,
                ctx->h_top_k_ids, ctx->h_top_k_vals, n_used,
                ctx->h_all_scores, ctx->n_experts_total,
                &ctx->ecache,
                use_gpu1 ? &ctx->gpu1.ecache : nullptr,
                &ctx->rcache,
                uncached_stg, &n_uncached,
                vram_sub_idx, vram_sub_slot, &n_vram_subs,
                vram1_sub_idx, vram1_sub_slot, &n_vram1_subs,
                ram_sub_idx, ram_sub_slot, &n_ram_subs
            );

            if (subs > 0) {
                ctx->prof.car_substitutions += subs;
                ctx->prof.car_nvme_avoided += n_vram_subs + n_vram1_subs + n_ram_subs;
                ctx->prof.car_score_ratio_sum = ctx->car.score_ratio_sum;
                ctx->prof.car_free_subs = (int)ctx->car.free_subs;

                // GPU0 VRAM substitutes → Phase 1
                for (int j = 0; j < n_vram_subs; j++) {
                    cached_stg[n_cached] = vram_sub_idx[j];
                    cached_slot[n_cached] = vram_sub_slot[j];
                    vram_cache_touch(&ctx->ecache, vram_sub_slot[j]);
                    n_cached++;
                }
                // GPU1 VRAM substitutes → GPU1 FFN
                for (int j = 0; j < n_vram1_subs; j++) {
                    gpu1_stg[n_gpu1] = vram1_sub_idx[j];
                    gpu1_slot[n_gpu1] = vram1_sub_slot[j];
                    vram_cache_touch(&ctx->gpu1.ecache, vram1_sub_slot[j]);
                    n_gpu1++;
                }
                // RAM substitutes → Phase 2 staging
                for (int j = 0; j < n_ram_subs; j++) {
                    ramhit_stg[n_ramhit] = ram_sub_idx[j];
                    ramhit_slot[n_ramhit] = ram_sub_slot[j];
                    ram_cache_touch(&ctx->rcache, ram_sub_slot[j]);
                    n_ramhit++;
                }

                for (int u = 0; u < n_uncached; u++)
                    uncached_ids[u] = ctx->h_top_k_ids[uncached_stg[u]];
            }
            ctx->prof.car_nvme_remaining += n_uncached;
        }

        // No-cold-fetch: drop remaining uncached experts (no NVMe blocking).
        // Prefetch pipeline handles NVMe loads asynchronously.
        if (ctx->no_cold_fetch && n_uncached > 0) {
            for (int u = 0; u < n_uncached; u++)
                ctx->h_top_k_vals[uncached_stg[u]] = 0.0f;
            // Renormalize remaining scores
            float sum = 0.0f;
            for (int i = 0; i < n_used; i++) sum += ctx->h_top_k_vals[i];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int i = 0; i < n_used; i++) ctx->h_top_k_vals[i] *= inv;
            }
            ctx->prof.cold_fetch_dropped += n_uncached;
            n_uncached = 0;
        }

        // === Phase 2 H2D split: assign some RAM-hit experts to GPU1 ===
        // Both GPUs' SDMAs transfer in parallel, doubling effective bandwidth.
        // NVMe misses stay on GPU0 (nvme_bg callback targets GPU0's staging).
        int gpu1_p2_n = 0;                    // GPU1 Phase 2 (staged) expert count
        int gpu1_p2_stg[16], gpu1_p2_slot[16]; // staging index + RAM cache slot
        static int h2d_split = -1;
        if (h2d_split < 0) h2d_split = !getenv("QMOE_NO_H2D_SPLIT");
        if (use_gpu1 && h2d_split && n_ramhit > 0) {
            int split = 0;  // send ALL RAM hits to GPU1
            for (int j = split; j < n_ramhit; j++) {
                gpu1_p2_stg[gpu1_p2_n] = ramhit_stg[j];
                gpu1_p2_slot[gpu1_p2_n] = ramhit_slot[j];
                gpu1_p2_n++;
            }
            n_ramhit = split;  // GPU0 keeps first half
        }

        // === GPU1 expert engine: VRAM hits + staged RAM-hit experts ===
        // GPU1 processes both zero-copy VRAM-cached experts AND RAM-hit experts
        // that are H2D'd to GPU1's staging buffer via GPU1's SDMA (in parallel
        // with GPU0's H2D on GPU0's SDMA).
        int n_gpu1_total = n_gpu1 + gpu1_p2_n;
        if (use_gpu1 && n_gpu1_total > 0) {
            (void)hipSetDevice(ctx->gpu1.device_id);
            hipStream_t s1 = ctx->gpu1.stream;
            hipStream_t ts1 = ctx->gpu1.transfer_stream;
            size_t gpu1_stg_base = (size_t)buf_set * n_used * stride;

            // Wait for any pending GPU1 VRAM cache population from previous layer
            (void)hipStreamWaitEvent(s1, ctx->gpu1.cache_event, 0);

            // Prepare GPU1 expert pointers and scores on host:
            //   [0..n_gpu1-1] = VRAM cache hits (zero-copy)
            //   [n_gpu1..n_gpu1_total-1] = staged RAM-hit experts
            float h_gpu1_scores[MAX_EXPERTS];
            for (int j = 0; j < n_gpu1; j++) {
                ctx->gpu1.h_expert_ptrs[j] = (uint8_t *)ctx->gpu1.ecache.d_buf +
                                              (size_t)gpu1_slot[j] * stride;
                h_gpu1_scores[j] = ctx->h_top_k_vals[gpu1_stg[j]];
            }
            for (int j = 0; j < gpu1_p2_n; j++) {
                ctx->gpu1.h_expert_ptrs[n_gpu1 + j] = (uint8_t *)ctx->gpu1.d_expert_staging +
                                                        gpu1_stg_base + (size_t)j * stride;
                h_gpu1_scores[n_gpu1 + j] = ctx->h_top_k_vals[gpu1_p2_stg[j]];
            }

            // Upload expert ptrs + scores to GPU1
            (void)hipMemcpyAsync(ctx->gpu1.d_expert_ptrs, ctx->gpu1.h_expert_ptrs,
                                  n_gpu1_total * sizeof(void *), hipMemcpyHostToDevice, s1);
            (void)hipMemcpyAsync(ctx->gpu1.d_scores, h_gpu1_scores,
                                  n_gpu1_total * sizeof(float), hipMemcpyHostToDevice, s1);

            // H2D hidden state to GPU1 (16 KB, from pinned h_norm)
            (void)hipMemcpyAsync(ctx->gpu1.d_norm, ctx->h_norm,
                                  n_embd * sizeof(float), hipMemcpyHostToDevice, s1);

            // H2D RAM-hit expert weights to GPU1 staging (on GPU1's SDMA, parallel)
            if (gpu1_p2_n > 0) {
                for (int j = 0; j < gpu1_p2_n; j++) {
                    (void)hipMemcpyAsync(
                        (uint8_t *)ctx->gpu1.d_expert_staging + gpu1_stg_base + (size_t)j * stride,
                        (uint8_t *)ctx->rcache.buf + (size_t)gpu1_p2_slot[j] * stride,
                        stride, hipMemcpyHostToDevice, ts1);
                }
                (void)hipEventRecord(ctx->gpu1.batch_event[buf_set], ts1);
                // Compute stream waits for H2D before FFN (VRAM-hit ptrs don't need this
                // but combined batch simplicity is worth the tiny extra latency)
                (void)hipStreamWaitEvent(s1, ctx->gpu1.batch_event[buf_set], 0);
            }

            // FFN on GPU1: fused gate+up+SiLU → down projection (all experts)
            {
                constexpr int RPB = 8;
                dim3 wblock(32, RPB);
                dim3 grid_gu((M_gate + RPB - 1) / RPB, n_gpu1_total);
                if (ctx->gate_type == GGML_TYPE_Q4_K_ID) {
                    kernel_q4k_fused_gate_up_silu<<<grid_gu, wblock, 0, s1>>>(
                        ctx->gpu1.d_expert_ptrs, ctx->gpu1.d_norm, ctx->gpu1.d_gate,
                        M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
                } else {
                    kernel_q6k_fused_gate_up_silu<<<grid_gu, wblock, 0, s1>>>(
                        ctx->gpu1.d_expert_ptrs, ctx->gpu1.d_norm, ctx->gpu1.d_gate,
                        M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
                }
                dim3 grid_d((n_embd + RPB - 1) / RPB, n_gpu1_total);
                if (down_type == GGML_TYPE_Q4_K_ID) {
                    kernel_q4k_matvec_per_expert<<<grid_d, wblock, 0, s1>>>(
                        ctx->gpu1.d_expert_ptrs, ctx->gpu1.d_gate, ctx->gpu1.d_down,
                        n_embd, M_gate, down_offset);
                } else if (down_type == GGML_TYPE_Q5_K_ID) {
                    kernel_q5k_matvec_per_expert<<<grid_d, wblock, 0, s1>>>(
                        ctx->gpu1.d_expert_ptrs, ctx->gpu1.d_gate, ctx->gpu1.d_down,
                        n_embd, M_gate, down_offset);
                } else {
                    kernel_q6k_matvec_per_expert<<<grid_d, wblock, 0, s1>>>(
                        ctx->gpu1.d_expert_ptrs, ctx->gpu1.d_gate, ctx->gpu1.d_down,
                        n_embd, M_gate, down_offset);
                }
            }

            // Weighted sum → d_partial, then D2H to pinned staging
            kernel_weighted_sum<<<(n_embd + 255) / 256, 256, 0, s1>>>(
                ctx->gpu1.d_down, ctx->gpu1.d_scores, ctx->gpu1.d_partial, n_embd, n_gpu1_total);
            (void)hipMemcpyAsync(ctx->h_gpu1_partial, ctx->gpu1.d_partial,
                                  n_embd * sizeof(float), hipMemcpyDeviceToHost, s1);
            // Use a different event for FFN completion (batch_event may be used for H2D above)
            (void)hipEventRecord(ctx->gpu1.cache_event, s1);

            (void)hipSetDevice(0);
            ctx->prof.gpu1_expert_count += n_gpu1_total;
        }

        // === Two-phase FFN pipelining (GPU0) ===
        // Phase 1: GPU0 VRAM-cached experts — FFN overlaps with H2D on transfer stream
        // Phase 2: Staged experts (GPU0's share of RAM + NVMe) — FFN after H2D
        // GPU1 experts excluded — processed on GPU1 above
        int n_gpu0 = n_cached + n_ramhit + n_uncached;
        void *h_ptrs_ordered[16];
        float h_scores_ordered[16];
        int   p2_eids[16];
        int n_p1 = 0, n_p2 = 0;

        for (int j = 0; j < n_cached; j++) {
            h_ptrs_ordered[n_p1] = (uint8_t *)ctx->ecache.d_buf + (size_t)cached_slot[j] * stride;
            h_scores_ordered[n_p1] = ctx->h_top_k_vals[cached_stg[j]];
            n_p1++;
        }
        for (int j = 0; j < n_ramhit; j++) {
            int p = n_p2++;
            h_ptrs_ordered[n_p1 + p] = (uint8_t *)ctx->d_expert_staging + stg_base + (size_t)p * stride;
            h_scores_ordered[n_p1 + p] = ctx->h_top_k_vals[ramhit_stg[j]];
            p2_eids[p] = ctx->h_top_k_ids[ramhit_stg[j]];
        }
        int p2_nv_start = n_p2;
        for (int j = 0; j < n_uncached; j++) {
            int p = n_p2++;
            h_ptrs_ordered[n_p1 + p] = (uint8_t *)ctx->d_expert_staging + stg_base + (size_t)p * stride;
            h_scores_ordered[n_p1 + p] = ctx->h_top_k_vals[uncached_stg[j]];
            p2_eids[p] = uncached_ids[j];
        }

        // CPU expert compute: avoid H2D by computing Phase 2 experts on CPU.
        // When few Phase 2 experts (RAM + NVMe ≤ threshold), CPU AVX2 FFN is faster
        // than H2D of 7.25 MB weights per expert. Sends only 16 KB result.
        static int cpu_threshold = -1;
        if (cpu_threshold < 0) {
            const char *env = getenv("QMOE_CPU_THRESHOLD");
            cpu_threshold = env ? atoi(env) : 0;
        }
        bool use_cpu_experts = (n_p2 > 0 && n_p2 <= cpu_threshold &&
                                ctx->cpu_expert != nullptr);

        // Upload GPU0 pointers + scores on compute stream
        memcpy(ctx->h_expert_ptrs, h_ptrs_ordered, n_gpu0 * sizeof(void *));
        (void)hipMemcpyAsync(ctx->d_expert_ptrs, ctx->h_expert_ptrs,
                             n_gpu0 * sizeof(void *), hipMemcpyHostToDevice, s);
        memcpy(ctx->h_top_k_vals, h_scores_ordered, n_gpu0 * sizeof(float));
        (void)hipMemcpyAsync(ctx->d_scores, ctx->h_top_k_vals,
                              n_gpu0 * sizeof(float), hipMemcpyHostToDevice, s);

        // Wait for VRAM prefetch H2D from previous layer's prediction (if active)
        if (ctx->vram_prefetch_active) {
            (void)hipStreamWaitEvent(s, ctx->vram_prefetch_event, 0);
            ctx->vram_prefetch_active = false;
        }

        // === PHASE 1 FFN: VRAM-cached experts (overlaps with H2D) ===
        // e[11] now recorded in queue_attn_and_router (after SharedFFN)
        if (n_p1 > 0) {
            constexpr int RPB = 8;
            dim3 wblock(32, RPB);
            // Fused gate+up+SiLU: reads input once, outputs SiLU(gate)*up to d_gate
            dim3 grid_gu((M_gate + RPB - 1) / RPB, n_p1);
            if (ctx->gate_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    ctx->d_expert_ptrs, ctx->d_norm, ctx->d_gate,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            } else {
                kernel_q6k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    ctx->d_expert_ptrs, ctx->d_norm, ctx->d_gate,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            }
            dim3 grid_d((n_embd + RPB - 1) / RPB, n_p1);
            if (down_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
                    n_embd, M_gate, down_offset);
            } else if (down_type == GGML_TYPE_Q5_K_ID) {
                kernel_q5k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
                    n_embd, M_gate, down_offset);
            } else {
                kernel_q6k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
                    n_embd, M_gate, down_offset);
            }
        }

        hipEventRecord(ctx->prof.e[8], s);   // Phase 1 end (= before H2D wait)

        // --- PASS 2: Late prefetch consumption (NVMe may have completed during Phase 1) ---
        // Gives ~0.6ms more overlap time for stragglers to arrive.
        if (ctx->pf_pending) {
            consume_prefetch_nvme(ctx);  // try once more
            ctx->pf_pending = false;     // give up if still not done
        }

        // === SPECULATIVE ROUTING: predict next layer's experts during NVMe wait ===
        // Uses Phase 1 partial FFN + shared expert + residual as approximate
        // post-FFN hidden state. Runs on compute stream s (fills GPU idle time).
        // Reuses d_attn_out (scratch1) and d_attn_gate_buf (scratch2) which are
        // Disabled with QMOE_NO_SPEC_ROUTING=1
        // free after attention and not needed until next layer.
        static int spec_disabled = -1;
        if (spec_disabled < 0) spec_disabled = getenv("QMOE_NO_SPEC_ROUTING") != NULL;
        if (!spec_disabled && il + 1 < max_layers_gpu) {
            struct gpu_layer *next_gl = &ctx->layers[il + 1];
            int blk = (n_embd + 255) / 256;

            // 1. Partial weighted sum of Phase 1 (VRAM-cached) expert outputs
            if (n_p1 > 0) {
                kernel_weighted_sum<<<blk, 256, 0, s>>>(
                    ctx->d_down, ctx->d_scores, ctx->d_attn_out, n_embd, n_p1);
            } else {
                (void)hipMemsetAsync(ctx->d_attn_out, 0, n_embd * sizeof(float), s);
            }

            // 2. Add shared expert output
            if (gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down) {
                kernel_vec_add<<<blk, 256, 0, s>>>(
                    ctx->d_attn_out, ctx->d_attn_out, ctx->d_shared_down_buf, n_embd);
            }

            // 3. Add residual → approximate post-FFN hidden state
            kernel_vec_add<<<blk, 256, 0, s>>>(
                ctx->d_attn_out, ctx->d_attn_out, ctx->d_x, n_embd);

            // 4. RMS norm with next layer's ffn_norm
            kernel_rms_norm<<<1, 256, 0, s>>>(
                ctx->d_attn_out, next_gl->d_ffn_norm, ctx->d_attn_gate_buf,
                n_embd, ctx->rms_norm_eps);

            // 5. Router matvec with next layer's router weights
            dispatch_matvec(s, next_gl->d_router, GGML_TYPE_F32_ID,
                            ctx->d_attn_gate_buf, ctx->d_spec_router_out,
                            ctx->n_experts_total, n_embd);

            // 6. Softmax + top-k
            kernel_softmax<<<1, 256, 0, s>>>(
                ctx->d_spec_router_out, ctx->n_experts_total);
            kernel_top_k_parallel<<<1, 32, 0, s>>>(
                ctx->d_spec_router_out, ctx->n_experts_total, ctx->n_experts_used,
                ctx->d_spec_top_k_ids, ctx->d_spec_top_k_vals);

            // 7. D2H predicted expert IDs
            (void)hipMemcpyAsync(ctx->h_spec_top_k_ids, ctx->d_spec_top_k_ids,
                                  ctx->n_experts_used * sizeof(int),
                                  hipMemcpyDeviceToHost, s);

            // 8. Record event — synced after NVMe wait to get prediction to CPU
            (void)hipEventRecord(ctx->spec_event, s);

            // L+2 speculative routing tested: 12.6% accuracy (shared expert only
            // doesn't approximate full FFN transform well enough). Not worth the
            // GPU overhead of chaining through L+1's shared expert.

        }

        // === SPECULATIVE PREFETCH (v2 module) ===
        // MOVED EARLY: sync spec routing results and submit prefetch BEFORE main
        // NVMe, maximizing overlap time. Prefetched experts arrive during:
        //   Phase 2 FFN + residual + L+1 attn+router + sync stall + classify + Phase 1
        //   ≈ 1.7ms with 3 NVMe drives → 8-12 experts achievable
        if (!spec_disabled && il + 1 < max_layers_gpu && ctx->pf &&
            ctx->prefetch.active) {
            (void)hipEventSynchronize(ctx->spec_event);

            // Save predictions for accuracy tracking at next layer
            int n_spec = ctx->n_experts_used;
            memcpy(ctx->spec_ids[il + 1], ctx->h_spec_top_k_ids, n_spec * sizeof(int));
            ctx->spec_n[il + 1] = n_spec;
            ctx->spec_valid[il + 1] = true;

            int next_layer = il + 1;

            // Classify predicted experts + submit NVMe reads to worker
            int nvme_submitted = prefetch_classify_and_submit(
                ctx->pf, next_layer,
                ctx->h_spec_top_k_ids, nullptr, n_spec,
                &ctx->ecache,
                use_gpu1 ? &ctx->gpu1.ecache : nullptr,
                &ctx->rcache);

            // Part A: RAM→VRAM H2D for predicted experts already in RAM cache
            int pfa_vslots[32], pfa_rslots[32];
            int n_ram_h2d = prefetch_get_ram_transfers(ctx->pf,
                pfa_vslots, pfa_rslots, 32);
            for (int i = 0; i < n_ram_h2d; i++) {
                (void)hipMemcpyAsync(
                    (uint8_t *)ctx->ecache.d_buf + (size_t)pfa_vslots[i] * stride,
                    (uint8_t *)ctx->rcache.buf + (size_t)pfa_rslots[i] * stride,
                    stride, hipMemcpyHostToDevice, ts);
            }

            if (nvme_submitted > 0) {
                ctx->pf_pending = true;
            } else {
                int total_vpf = prefetch_vram_transfer_count(ctx->pf);
                if (total_vpf > 0) {
                    prefetch_mark_all_h2d_done(ctx->pf);
                    (void)hipEventRecord(ctx->vram_prefetch_event, ts);
                    ctx->vram_prefetch_active = true;
                    ctx->prof.vram_prefetched += total_vpf;
                }
            }
        }

        // Step 1: Start NVMe load in background with per-expert H2D callback
        // Each expert's H2D is queued on the transfer stream as its read completes,
        // eliminating the idle gap between RAM H2D and NVMe H2D.
        struct timespec nvme_t0, nvme_t1;
        clock_gettime(CLOCK_MONOTONIC, &nvme_t0);
        bool nvme_bg_active = false;
        if (n_uncached > 0 && ctx->nvme_bg.active) {
            ctx->nvme_bg.h2d_stream = ts;
            ctx->nvme_bg.d_staging = (uint8_t *)ctx->d_expert_staging;
            ctx->nvme_bg.stg_base = stg_base;
            ctx->nvme_bg.nv_start = p2_nv_start;
            ctx->nvme_bg.stride = stride;
            ctx->nvme_bg.h2d_enabled = !use_cpu_experts;
            nvme_bg_start(ctx, m->nvme_io, il, uncached_ids, n_uncached, nvme_buf_off);
            nvme_bg_active = true;
        }

        // Step 2: Double-buffer wait (ensures staging buffer from 2 layers ago is free)
        clock_gettime(CLOCK_MONOTONIC, &sp_t1);
        ctx->prof.classify_ms += (sp_t1.tv_sec - sp_t0.tv_sec) * 1000.0 +
                                  (sp_t1.tv_nsec - sp_t0.tv_nsec) / 1e6;
        clock_gettime(CLOCK_MONOTONIC, &sp_t0);
        if (il >= 2) {
            (void)hipEventSynchronize(ctx->batch_event[buf_set]);
        }
        rcache_pop_wait(ctx);

        // Step 3: RAM hit H2D (GPU path only — CPU path computes on host data)
        if (!use_cpu_experts) {
            // Coalesced: gather into contiguous bounce buffer, then single large H2D transfer.
            // One large DMA gets much closer to PCIe peak bandwidth than N small ones.
            if (n_ramhit > 0 && ctx->h2d_bounce) {
                if (ctx->bounce_worker.active) {
                    bounce_worker_start(ctx, ramhit_slot, n_ramhit);
                    bounce_worker_wait(ctx);
                } else {
                    for (int j = 0; j < n_ramhit; j++) {
                        memcpy((uint8_t *)ctx->h2d_bounce + (size_t)j * stride,
                               (uint8_t *)ctx->rcache.buf + (size_t)ramhit_slot[j] * stride,
                               stride);
                    }
                }
                // Single coalesced H2D transfer (bounce and staging are both contiguous)
                (void)hipMemcpyAsync(
                    (uint8_t *)ctx->d_expert_staging + stg_base,
                    ctx->h2d_bounce,
                    (size_t)n_ramhit * stride, hipMemcpyHostToDevice, ts);
            } else if (n_ramhit > 0) {
                for (int j = 0; j < n_ramhit; j++) {
                    (void)hipMemcpyAsync(
                        (uint8_t *)ctx->d_expert_staging + stg_base + (size_t)j * stride,
                        (uint8_t *)ctx->rcache.buf + (size_t)ramhit_slot[j] * stride,
                        stride, hipMemcpyHostToDevice, ts);
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &sp_t1);
        ctx->prof.rcwait_ms += (sp_t1.tv_sec - sp_t0.tv_sec) * 1000.0 +
                                (sp_t1.tv_nsec - sp_t0.tv_nsec) / 1e6;

        // Step 4: Wait for NVMe background load + H2D for results
        void *expert_buffers[16];
        struct timespec nw_t0, nw_t1;
        if (nvme_bg_active) {
            clock_gettime(CLOCK_MONOTONIC, &nw_t0);
            nvme_bg_wait(ctx);
            clock_gettime(CLOCK_MONOTONIC, &nw_t1);
            ctx->prof.nvme_wait_ms += (nw_t1.tv_sec - nw_t0.tv_sec) * 1000.0 +
                                       (nw_t1.tv_nsec - nw_t0.tv_nsec) / 1e6;
            if (ctx->nvme_bg.result != 0) {
                fprintf(stderr, "gpu_forward: NVMe failed layer %d\n", il);
                return nullptr;
            }
            memcpy(expert_buffers, ctx->nvme_bg.out_buffers, n_uncached * sizeof(void *));
            // H2D already queued by per-expert callback — no loop needed here
        } else if (n_uncached > 0) {
            if (nvme_io_load_experts_at(m->nvme_io, il, uncached_ids,
                                         n_uncached, nvme_buf_off, expert_buffers) != 0) {
                fprintf(stderr, "gpu_forward: NVMe failed layer %d\n", il);
                return nullptr;
            }
            if (!use_cpu_experts) {
                for (int j = 0; j < n_uncached; j++) {
                    (void)hipMemcpyAsync(
                        (uint8_t *)ctx->d_expert_staging + stg_base + (size_t)(p2_nv_start + j) * stride,
                        expert_buffers[j], stride, hipMemcpyHostToDevice, ts);
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &nvme_t1);
        ctx->prof.nvme_ms += (nvme_t1.tv_sec - nvme_t0.tv_sec) * 1000.0 +
                              (nvme_t1.tv_nsec - nvme_t0.tv_nsec) / 1e6;

        // CPU expert compute: submit to background worker thread (non-blocking).
        // Worker runs concurrently with GPU Phase 1 FFN and output_residual.
        if (use_cpu_experts) {
            cpu_expert_ctx_t *cex = (cpu_expert_ctx_t *)ctx->cpu_expert;
            cpu_expert_work_t cpu_work[16];
            int wi = 0;

            // RAM hits (data in rcache.buf, available immediately)
            for (int j = 0; j < n_ramhit; j++) {
                cpu_expert_work_t *w = &cpu_work[wi++];
                w->expert_data = (uint8_t *)ctx->rcache.buf + (size_t)ramhit_slot[j] * stride;
                w->score = h_scores_ordered[n_p1 + j];
                w->gate_type = ctx->gate_type;
                w->up_type = ctx->up_type;
                w->down_type = down_type;
                w->expert_gate_size = ctx->expert_gate_size;
                w->expert_up_size = ctx->expert_up_size;
            }

            // NVMe misses (data available after nvme_bg_wait above)
            for (int j = 0; j < n_uncached; j++) {
                cpu_expert_work_t *w = &cpu_work[wi++];
                w->expert_data = expert_buffers[j];
                w->score = h_scores_ordered[n_p1 + p2_nv_start + j];
                w->gate_type = ctx->gate_type;
                w->up_type = ctx->up_type;
                w->down_type = down_type;
                w->expert_gate_size = ctx->expert_gate_size;
                w->expert_up_size = ctx->expert_up_size;
            }

            cpu_expert_submit_async(cex, ctx->h_norm, cpu_work, wi);
            ctx->prof.cpu_expert_count += wi;
        }

        // Record event on ts — Phase 2 will wait for this.
        // For CPU expert path, batch_event is recorded later after cpu_expert_wait + H2D.
        if (!use_cpu_experts) {
            (void)hipEventRecord(ctx->batch_event[buf_set], ts);
        }

        // === PHASE 2 FFN: staged experts (after H2D completes) ===
        // Skipped when CPU expert compute handles Phase 2.
        if (!use_cpu_experts && n_p2 > 0) {
            (void)hipStreamWaitEvent(s, ctx->batch_event[buf_set], 0);
            hipEventRecord(ctx->prof.e[9], s);   // after H2D wait

            constexpr int RPB = 8;
            dim3 wblock(32, RPB);
            void **d_ptrs_p2 = ctx->d_expert_ptrs + n_p1;
            float *d_gate_p2 = ctx->d_gate + (size_t)n_p1 * M_gate;
            float *d_up_p2   = ctx->d_up   + (size_t)n_p1 * M_gate;
            float *d_down_p2 = ctx->d_down + (size_t)n_p1 * n_embd;

            // Fused gate+up+SiLU: reads input once, outputs SiLU(gate)*up to d_gate_p2
            dim3 grid_gu((M_gate + RPB - 1) / RPB, n_p2);
            if (ctx->gate_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    d_ptrs_p2, ctx->d_norm, d_gate_p2,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            } else {
                kernel_q6k_fused_gate_up_silu<<<grid_gu, wblock, 0, s>>>(
                    d_ptrs_p2, ctx->d_norm, d_gate_p2,
                    M_gate, n_embd, (uint64_t)0, ctx->expert_gate_size);
            }

            dim3 grid_d((n_embd + RPB - 1) / RPB, n_p2);
            if (down_type == GGML_TYPE_Q4_K_ID) {
                kernel_q4k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            } else if (down_type == GGML_TYPE_Q5_K_ID) {
                kernel_q5k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            } else {
                kernel_q6k_matvec_per_expert<<<grid_d, wblock, 0, s>>>(
                    d_ptrs_p2, d_gate_p2, d_down_p2,
                    n_embd, M_gate, down_offset);
            }
        } else {
            hipEventRecord(ctx->prof.e[9], s);   // no Phase 2 needed
        }

        // Fused: weighted_sum + shared_expert_add + residual (3 kernels → 1)
        {
            bool has_shared = gl->d_shared_gate && gl->d_shared_up && gl->d_shared_down;
            int n_gpu = use_cpu_experts ? n_p1 : n_gpu0;
            kernel_moe_output_residual<<<(n_embd + 255) / 256, 256, 0, s>>>(
                ctx->d_down, ctx->d_scores,
                ctx->d_shared_down_buf, ctx->d_x,
                ctx->d_x, n_embd, n_gpu, has_shared);

            // CPU expert: wait for async worker, H2D result, add to output.
            if (use_cpu_experts && n_p2 > 0) {
                struct timespec cpu_t0, cpu_t1;
                clock_gettime(CLOCK_MONOTONIC, &cpu_t0);

                cpu_expert_ctx_t *cex = (cpu_expert_ctx_t *)ctx->cpu_expert;
                cpu_expert_wait(cex);

                memcpy(ctx->h_cpu_partial, cpu_expert_result(cex),
                       n_embd * sizeof(float));
                (void)hipMemcpyAsync(ctx->d_cpu_partial, ctx->h_cpu_partial,
                                      n_embd * sizeof(float), hipMemcpyHostToDevice, ts);
                (void)hipEventRecord(ctx->batch_event[buf_set], ts);
                (void)hipStreamWaitEvent(s, ctx->batch_event[buf_set], 0);
                kernel_vec_add<<<(n_embd + 255) / 256, 256, 0, s>>>(
                    ctx->d_x, ctx->d_x, ctx->d_cpu_partial, n_embd);

                clock_gettime(CLOCK_MONOTONIC, &cpu_t1);
                ctx->prof.cpu_expert_ms += (cpu_t1.tv_sec - cpu_t0.tv_sec) * 1000.0 +
                                            (cpu_t1.tv_nsec - cpu_t0.tv_nsec) / 1e6;
            }

            // GPU1 expert engine: collect partial sum and add to d_x.
            // Use GPU-side stream wait instead of CPU-blocking hipEventSynchronize.
            // GPU0's transfer stream waits for GPU1's event, relays h_gpu1_partial
            // to d_gpu1_partial, then compute stream waits and does vec_add.
            // The CPU never blocks — it can immediately proceed to cache updates
            // and pre-queue the next layer's attention.
            if (use_gpu1 && n_gpu1_total > 0) {
                // Transfer stream: wait for GPU1 D2H, then H2D partial to GPU0
                (void)hipStreamWaitEvent(ts, ctx->gpu1.cache_event, 0);
                (void)hipMemcpyAsync(ctx->d_gpu1_partial, ctx->h_gpu1_partial,
                                      n_embd * sizeof(float), hipMemcpyHostToDevice, ts);
                (void)hipEventRecord(ctx->batch_event[buf_set], ts);

                // Compute stream: wait for relay, add to d_x
                (void)hipStreamWaitEvent(s, ctx->batch_event[buf_set], 0);
                kernel_vec_add<<<(n_embd + 255) / 256, 256, 0, s>>>(
                    ctx->d_x, ctx->d_x, ctx->d_gpu1_partial, n_embd);
            }
        }
        hipEventRecord(ctx->prof.e[10], s);  // FFN end
        hipEventRecord(ctx->prof.e[7], s);   // MoE pipeline end (= e[10] after fusion)

        // VRAM cache update: Phase 2 experts only (Phase 1 already in VRAM)
        int spl = ctx->ecache.slots_per_layer;
        {
            if (!use_cpu_experts && n_p2 > 0 && spl > 0) {
                for (int j = 0; j < n_p2; j++) {
                    int slot = vram_cache_alloc_slot(&ctx->ecache, il, p2_eids[j]);
                    (void)hipMemcpyAsync(
                        (uint8_t *)ctx->ecache.d_buf + (size_t)slot * stride,
                        (uint8_t *)ctx->d_expert_staging + stg_base + (size_t)j * stride,
                        stride, hipMemcpyDeviceToDevice, ts);
                }
            }
        }

        // RAM cache update: metadata sync, data copy async
        // Collect slots + sources, update metadata now, dispatch memcpy to worker
        int rspl = ctx->rcache.slots_per_layer;
        {
            void *pop_srcs[16];
            int   pop_slots[16];
            int   pop_n = 0;

            // NVMe misses → RAM cache
            if (n_uncached > 0 && rspl > 0) {
                for (int j = 0; j < n_uncached; j++) {
                    int eid = uncached_ids[j];
                    if (ctx->rcache.map[il * ctx->rcache.n_experts + eid] >= 0) continue;

                    int rslot = ram_cache_alloc_slot(&ctx->rcache, il, eid);

                    pop_srcs[pop_n] = expert_buffers[j];
                    pop_slots[pop_n] = rslot;
                    pop_n++;
                }
            }

            // Dispatch async memcpy (overlaps with GPU FFN + next layer's attention)
            if (pop_n > 0) {
                rcache_pop_start(ctx, pop_srcs, pop_slots, pop_n);
            }
        }

        // Expert frequency counting
        if (ctx->freq_enabled) {
            for (int i = 0; i < n_used; i++)
                ctx->expert_freq[il * ctx->n_experts_total + ctx->h_top_k_ids[i]]++;
        }

        // Dump expert IDs if QMOE_DUMP_EXPERTS=1
        if (getenv("QMOE_DUMP_EXPERTS")) {
            fprintf(stderr, "EXPERTS tok=%d layer=%d:", ctx->prof.token_count, il);
            for (int i = 0; i < n_used; i++)
                fprintf(stderr, " %d", ctx->h_top_k_ids[i]);
            fprintf(stderr, "\n");
        }

        // Save current expert selections for next token's prefetch prediction
        for (int i = 0; i < n_used; i++) {
            ctx->prefetch.prev_experts[il][i] = ctx->h_top_k_ids[i];
        }

        // NOTE: Speculative routing + prefetch is handled above:
        // - GPU prediction queued after Phase 1 FFN (fills idle time during NVMe wait)
        // - CPU submits prefetch AFTER nvme_bg_wait (no io_uring race)
        // - Prefetch results consumed at next layer's classify step

        // Pre-queue NEXT layer's attention + router on GPU stream.
        // GPU starts executing immediately after current layer's FFN residual,
        // overlapping with spec_pf wait + CPU cache work at the top of the next iteration.
        if (il + 1 < max_layers_gpu) {
            queue_attn_and_router(ctx, il + 1, layer_kv_idx[il + 1],
                                  layer_ssm_idx[il + 1], pos);
        }
    }

    // Mark that we have previous token's expert selections for next call
    ctx->prefetch.has_prev = true;

    // ---- Final output ----
    hipEventRecord(ctx->prof.e[2], s);

    // RMS norm
    kernel_rms_norm<<<1, 256, 0, s>>>(
        ctx->d_x, ctx->d_output_norm, ctx->d_norm, n_embd, ctx->rms_norm_eps);

    // LM head: logits = output @ d_norm
    dispatch_matvec(s, ctx->d_output, ctx->output_type, ctx->d_norm, ctx->d_logits,
                    ctx->vocab_size, n_embd);

    // D2H: logits
    (void)hipMemcpyAsync(ctx->h_logits, ctx->d_logits,
                          ctx->vocab_size * sizeof(float), hipMemcpyDeviceToHost, s);
    hipEventRecord(ctx->prof.e[3], s);
    (void)hipStreamSynchronize(s);

    // Query last layer's deferred MoE pipeline, and final output timing
    // SharedFFN is pre-queued: e[4]→e[11], MoE pipe: e[5]→e[7]
    {
        float ms;
        hipEventElapsedTime(&ms, ctx->prof.e[5], ctx->prof.e[7]);
        ctx->prof.moe_pipe_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[4], ctx->prof.e[11]);
        ctx->prof.shared_ffn_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[5], ctx->prof.e[8]);
        ctx->prof.phase1_ffn_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[8], ctx->prof.e[9]);
        ctx->prof.h2d_wait_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[9], ctx->prof.e[10]);
        ctx->prof.gpu_ffn_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[10], ctx->prof.e[7]);
        ctx->prof.post_ffn_ms += ms;
        hipEventElapsedTime(&ms, ctx->prof.e[2], ctx->prof.e[3]);
        ctx->prof.output_ms = ms;
    }

    // Measure wall clock for entire gpu_forward
    struct timespec gpu_fwd_wall_end;
    clock_gettime(CLOCK_MONOTONIC, &gpu_fwd_wall_end);
    double wall_ms = (gpu_fwd_wall_end.tv_sec - gpu_fwd_wall_start.tv_sec) * 1000.0 +
                     (gpu_fwd_wall_end.tv_nsec - gpu_fwd_wall_start.tv_nsec) / 1e6;

    // Print profiling summary
    // NVMe(CPU) overlaps with MoE pipe GPU idle wait, so exclude from total
    ctx->prof.token_count++;
    double total = ctx->prof.embed_ms + ctx->prof.attn_ms + ctx->prof.router_ms +
                   ctx->prof.moe_pipe_ms + ctx->prof.output_ms;
    const char *dbg_env = getenv("QMOE_DEBUG");
    int dbg_level = dbg_env ? atoi(dbg_env) : 1;
    if (dbg_level > 0 && (ctx->prof.token_count <= 3 || ctx->prof.token_count % 10 == 0)) {
        uint64_t ch = ctx->ecache.hits, cm = ctx->ecache.misses;
        uint64_t g1h = ctx->gpu1.ecache.hits;
        uint64_t total_vhits = ch + g1h;
        uint64_t total_vmisses = cm;
        double hit_pct = (total_vhits + total_vmisses > 0) ? 100.0 * total_vhits / (total_vhits + total_vmisses) : 0;
        uint64_t rh = ctx->rcache.hits, rm = ctx->rcache.misses;
        double ram_pct = (rh + rm > 0) ? 100.0 * rh / (rh + rm) : 0;
        // VRAM/RAM cache stats + speculative routing
        fprintf(stderr,
            "\n[Profile] tok=%d  pos=%d  layers=%d  gpu=%.1f ms  wall=%.1f ms (%.2f tok/s)\n"
            "  Embed:      %7.2f ms  %5.1f%%\n"
            "  Attention:  %7.2f ms  %5.1f%%  (%.2f ms/layer)\n"
            "    FullAttn: %7.2f ms         (%.2f ms/layer, %d layers)\n"
            "    SSM:      %7.2f ms         (%.2f ms/layer, %d layers)\n"
            "  Router+D2H: %7.2f ms  %5.1f%%  (%.2f ms/layer)\n"
            "    Sync stall:%6.2f ms         (stream sync blocking)\n"
            "  MoE pipe:   %7.2f ms  %5.1f%%  (%.2f ms/layer)\n"
            "    SharedFFN:%7.2f ms         (%.2f ms/layer, e4→e11 overlapped)\n"
            "    P1 FFN:   %7.2f ms         (%.2f ms/layer, VRAM-cached)\n"
            "    H2D wait: %7.2f ms         (%.2f ms/layer, StreamWaitEvent)\n"
            "    P2 FFN:   %7.2f ms         (%.2f ms/layer, staged)\n"
            "    PostFFN:  %7.2f ms         (%.2f ms/layer, e10→e7)\n"
            "    NVMe wait:%7.2f ms         (%.2f ms/layer, CPU blocking)\n"
            "    NVMe(CPU):%7.2f ms         (%.2f ms/layer, overlapped)\n"
            "    Classify: %7.2f ms         (%.2f ms/layer)\n"
            "    RC wait:  %7.2f ms         (%.2f ms/layer, incl dbl-buf sync)\n"
            "    SpecPF:   %7.2f ms         (%.2f ms/layer, async NVMe wait)\n"
            "    CPU FFN:  %7.2f ms         (%d experts, %.2f ms/expert, wakeup=%.0fus compute=%.0fus n=%d)\n"
            "    VRAM$:     %llu hits, %llu misses (%.1f%% hit)%s\n"
            "    RAM$:      %llu hits, %llu misses (%.1f%% hit)\n"
            "    GPU1:      %d experts (%.2f ms wait)\n"
            "    Prefetch:  %d RAM$, %d VRAM$ (spec routing)\n"
            "    SpecRoute: %d/%d predicted (%.1f%% accuracy)\n"
            "    Overlap:  %d/%d NVMe misses predictable (%.1f%% by prev-token)\n"
            "    CAR:      %d subs (%d free), %d NVMe avoided, %d remaining, %d dropped (avg ratio %.2f)\n"
            "  Output:     %7.2f ms  %5.1f%%\n",
            ctx->prof.token_count, pos, ctx->n_layers, total, wall_ms, 1000.0 / wall_ms,
            ctx->prof.embed_ms, 100.0 * ctx->prof.embed_ms / total,
            ctx->prof.attn_ms, 100.0 * ctx->prof.attn_ms / total,
            ctx->prof.attn_ms / ctx->n_layers,
            ctx->prof.attn_full_ms,
            ctx->n_full_attn > 0 ? ctx->prof.attn_full_ms / ctx->n_full_attn : 0.0,
            ctx->n_full_attn,
            ctx->prof.attn_ssm_ms,
            (ctx->n_layers - ctx->n_full_attn) > 0 ? ctx->prof.attn_ssm_ms / (ctx->n_layers - ctx->n_full_attn) : 0.0,
            ctx->n_layers - ctx->n_full_attn,
            ctx->prof.router_ms, 100.0 * ctx->prof.router_ms / total,
            ctx->prof.router_ms / ctx->n_layers,
            ctx->prof.hipsync_ms,
            ctx->prof.moe_pipe_ms, 100.0 * ctx->prof.moe_pipe_ms / total,
            ctx->prof.moe_pipe_ms / ctx->n_layers,
            ctx->prof.shared_ffn_ms, ctx->prof.shared_ffn_ms / ctx->n_layers,
            ctx->prof.phase1_ffn_ms, ctx->prof.phase1_ffn_ms / ctx->n_layers,
            ctx->prof.h2d_wait_ms, ctx->prof.h2d_wait_ms / ctx->n_layers,
            ctx->prof.gpu_ffn_ms, ctx->prof.gpu_ffn_ms / ctx->n_layers,
            ctx->prof.post_ffn_ms, ctx->prof.post_ffn_ms / ctx->n_layers,
            ctx->prof.nvme_wait_ms, ctx->prof.nvme_wait_ms / ctx->n_layers,
            ctx->prof.nvme_ms, ctx->prof.nvme_ms / ctx->n_layers,
            ctx->prof.classify_ms, ctx->prof.classify_ms / ctx->n_layers,
            ctx->prof.rcwait_ms, ctx->prof.rcwait_ms / ctx->n_layers,
            ctx->prof.spec_pf_ms, ctx->prof.spec_pf_ms / ctx->n_layers,
            ctx->prof.cpu_expert_ms, ctx->prof.cpu_expert_count,
            ctx->prof.cpu_expert_count > 0 ? ctx->prof.cpu_expert_ms / ctx->prof.cpu_expert_count : 0.0,
            [&]() -> double { double w, c; int n; if (ctx->cpu_expert) { cpu_expert_get_timing((cpu_expert_ctx_t *)ctx->cpu_expert, &w, &c, &n); return w; } return 0.0; }(),
            [&]() -> double { double w, c; int n; if (ctx->cpu_expert) { cpu_expert_get_timing((cpu_expert_ctx_t *)ctx->cpu_expert, &w, &c, &n); return c; } return 0.0; }(),
            [&]() -> int { double w, c; int n; if (ctx->cpu_expert) { cpu_expert_get_timing((cpu_expert_ctx_t *)ctx->cpu_expert, &w, &c, &n); return n; } return 0; }(),
            (unsigned long long)(ch + g1h), (unsigned long long)cm, hit_pct,
            (ctx->n_devices >= 2) ? (g1h > 0 ? " [GPU0+GPU1]" : " [GPU0 only]") : "",
            (unsigned long long)rh, (unsigned long long)rm, ram_pct,
            ctx->prof.gpu1_expert_count, ctx->prof.gpu1_wait_ms,
            ctx->prof.spec_prefetched, ctx->prof.vram_prefetched,
            ctx->prof.spec_hits, ctx->prof.spec_total,
            ctx->prof.spec_total > 0 ? 100.0 * ctx->prof.spec_hits / ctx->prof.spec_total : 0.0,
            ctx->prof.overlap_hits, ctx->prof.overlap_total,
            ctx->prof.overlap_total > 0 ? 100.0 * ctx->prof.overlap_hits / ctx->prof.overlap_total : 0.0,
            ctx->prof.car_substitutions, ctx->prof.car_free_subs,
            ctx->prof.car_nvme_avoided, ctx->prof.car_nvme_remaining,
            ctx->prof.cold_fetch_dropped,
            ctx->prof.car_substitutions > 0 ? ctx->prof.car_score_ratio_sum / ctx->prof.car_substitutions : 0.0,
            ctx->prof.output_ms, 100.0 * ctx->prof.output_ms / total);
    }

    return ctx->h_logits;
}

// ============================================================
// PUBLIC API: gpu_reset_state (for chunked perplexity)
// ============================================================

extern "C" void gpu_reset_state(gpu_ctx_t *ctx) {
    if (!ctx || !ctx->model_uploaded) return;

    // Drain backfill worker: wait for any in-flight NVMe reads to finish,
    // then flush the queue (stale entries from the previous chunk).
    if (ctx->backfill.active) {
        // Wait for worker to finish current batch
        pthread_mutex_lock(&ctx->backfill.mutex);
        while (!ctx->backfill.work_done && !ctx->backfill.shutdown)
            pthread_cond_wait(&ctx->backfill.cond_done, &ctx->backfill.mutex);
        pthread_mutex_unlock(&ctx->backfill.mutex);
        // Commit any completed reads before we reset
        backfill_try_complete(ctx);
        // Flush queue and dedup bitmap
        ctx->backfill.queue_count = 0;
        size_t dedup_bits = (size_t)ctx->n_layers * ctx->n_experts_total;
        size_t dedup_words = (dedup_bits + 63) / 64;
        memset(ctx->backfill.dedup, 0, dedup_words * sizeof(uint64_t));
    }

    size_t state_per_layer = (size_t)ctx->ssm_time_step_rank *
        ctx->ssm_state_size * ctx->ssm_value_dim * sizeof(float);
    int conv_steps = ctx->ssm_conv_kernel - 1;
    size_t conv_per_layer = (size_t)conv_steps * ctx->ssm_qkv_dim * sizeof(float);
    size_t kv_dim = (size_t)ctx->n_kv_heads * ctx->head_dim;

    if (ctx->pingpong) {
        for (int d = 0; d < 2; d++) {
            gpu_dev_t *dc = &ctx->dev[d];
            hipSetDevice(dc->device_id);

            if (dc->d_kv_k && dc->n_full_attn > 0) {
                size_t kv_total = (size_t)dc->n_full_attn * ctx->ctx_len * kv_dim * sizeof(__half);
                hipMemset(dc->d_kv_k, 0, kv_total);
                hipMemset(dc->d_kv_v, 0, kv_total);
            }
            if (dc->d_ssm_states && dc->n_ssm_layers > 0) {
                hipMemset(dc->d_ssm_states, 0, dc->n_ssm_layers * state_per_layer);
                hipMemset(dc->d_ssm_conv_bufs, 0, dc->n_ssm_layers * conv_per_layer);
                hipMemset(dc->d_ssm_conv_pos, 0, dc->n_ssm_layers * sizeof(int));
                memset(dc->h_ssm_conv_pos, 0, dc->n_ssm_layers * sizeof(int));
            }
        }
        hipSetDevice(ctx->dev[0].device_id);
    } else {
        if (ctx->d_kv_k && ctx->n_full_attn > 0) {
            size_t kv_total = (size_t)ctx->n_full_attn * ctx->ctx_len * kv_dim * sizeof(__half);
            hipMemset(ctx->d_kv_k, 0, kv_total);
            hipMemset(ctx->d_kv_v, 0, kv_total);
        }
        if (ctx->d_ssm_states && ctx->n_ssm_layers > 0) {
            hipMemset(ctx->d_ssm_states, 0, ctx->n_ssm_layers * state_per_layer);
            hipMemset(ctx->d_ssm_conv_bufs, 0, ctx->n_ssm_layers * conv_per_layer);
            hipMemset(ctx->d_ssm_conv_pos, 0, ctx->n_ssm_layers * sizeof(int));
            memset(ctx->h_ssm_conv_pos, 0, ctx->n_ssm_layers * sizeof(int));
        }
    }

    // Clear spec predictions
    memset(ctx->spec_valid, 0, sizeof(ctx->spec_valid));
    memset(ctx->spec2_valid, 0, sizeof(ctx->spec2_valid));

    hipDeviceSynchronize();
}

// ============================================================
// PUBLIC API: gpu_free
// ============================================================

extern "C" void gpu_free(gpu_ctx_t *ctx) {
    if (!ctx) return;

    // Destroy profiling events
    if (ctx->model_uploaded) {
        for (int i = 0; i < 12; i++) {
            (void)hipEventDestroy(ctx->prof.e[i]);
        }
        (void)hipEventDestroy(ctx->spec_event);
        (void)hipEventDestroy(ctx->vram_prefetch_event);
    }

    // Shutdown spec routing worker
    if (ctx->spec_worker.active) {
        pthread_mutex_lock(&ctx->spec_worker.mutex);
        ctx->spec_worker.shutdown = true;
        pthread_cond_signal(&ctx->spec_worker.cond);
        pthread_mutex_unlock(&ctx->spec_worker.mutex);
        pthread_join(ctx->spec_worker.thread, nullptr);
        pthread_mutex_destroy(&ctx->spec_worker.mutex);
        pthread_cond_destroy(&ctx->spec_worker.cond);
        ctx->spec_worker.active = false;
    }

    // Shutdown backfill worker
    if (ctx->backfill.active) {
        pthread_mutex_lock(&ctx->backfill.mutex);
        ctx->backfill.shutdown = true;
        pthread_cond_signal(&ctx->backfill.cond_work);
        pthread_mutex_unlock(&ctx->backfill.mutex);
        pthread_join(ctx->backfill.thread, nullptr);
        pthread_mutex_destroy(&ctx->backfill.mutex);
        pthread_cond_destroy(&ctx->backfill.cond_work);
        pthread_cond_destroy(&ctx->backfill.cond_done);
        for (int i = 0; i < ctx->backfill.batch_max; i++) free(ctx->backfill.buffers[i]);
        free(ctx->backfill.dedup);
        ctx->backfill.active = false;
    }

    // Shutdown NVMe background loader
    if (ctx->nvme_bg.active) {
        pthread_mutex_lock(&ctx->nvme_bg.mutex);
        ctx->nvme_bg.shutdown = true;
        pthread_cond_signal(&ctx->nvme_bg.cond_work);
        pthread_mutex_unlock(&ctx->nvme_bg.mutex);
        pthread_join(ctx->nvme_bg.thread, NULL);
        pthread_mutex_destroy(&ctx->nvme_bg.mutex);
        pthread_cond_destroy(&ctx->nvme_bg.cond_work);
        pthread_cond_destroy(&ctx->nvme_bg.cond_done);
    }

    // Shutdown speculative prefetch (v2 module)
    if (ctx->pf) {
        prefetch_destroy(ctx->pf);
        ctx->pf = nullptr;
    }

    // Destroy pipeline resources
    (void)hipEventDestroy(ctx->batch_event[0]);
    (void)hipEventDestroy(ctx->batch_event[1]);
    (void)hipEventDestroy(ctx->shared_event);
    (void)hipStreamDestroy(ctx->transfer_stream);
    (void)hipStreamDestroy(ctx->shared_stream);

    // Unregister any remaining buffers
    if (ctx->registered_bufs) {
        for (int i = 0; i < ctx->n_registered; i++) {
            if (ctx->registered_bufs[i])
                (void)hipHostUnregister(ctx->registered_bufs[i]);
        }
        free(ctx->registered_bufs);
    }

    // Shutdown RAM cache population worker
    if (ctx->rcache_pop.active) {
        pthread_mutex_lock(&ctx->rcache_pop.mutex);
        ctx->rcache_pop.shutdown = true;
        pthread_cond_signal(&ctx->rcache_pop.cond_work);
        pthread_mutex_unlock(&ctx->rcache_pop.mutex);
        pthread_join(ctx->rcache_pop.thread, NULL);
        pthread_mutex_destroy(&ctx->rcache_pop.mutex);
        pthread_cond_destroy(&ctx->rcache_pop.cond_work);
        pthread_cond_destroy(&ctx->rcache_pop.cond_done);
    }

    // H2D bounce buffer
    // Shutdown bounce memcpy worker
    if (ctx->bounce_worker.active) {
        pthread_mutex_lock(&ctx->bounce_worker.mutex);
        ctx->bounce_worker.shutdown = true;
        pthread_cond_signal(&ctx->bounce_worker.cond_work);
        pthread_mutex_unlock(&ctx->bounce_worker.mutex);
        pthread_join(ctx->bounce_worker.thread, NULL);
        pthread_mutex_destroy(&ctx->bounce_worker.mutex);
        pthread_cond_destroy(&ctx->bounce_worker.cond_work);
        pthread_cond_destroy(&ctx->bounce_worker.cond_done);
    }

    if (ctx->h2d_bounce) (void)hipHostFree(ctx->h2d_bounce);

    // Expert frequency counter
    free(ctx->expert_freq);

    // Flush and close routing trace
    if (ctx->trace.fp) {
        if (ctx->trace.buf_used > 0)
            fwrite(ctx->trace.buf, 1, ctx->trace.buf_used, ctx->trace.fp);
        fclose(ctx->trace.fp);
        free(ctx->trace.buf);
        ctx->trace.fp = nullptr;
    }

    // Expert VRAM cache (v2 module)
    (void)hipFree(ctx->ecache.d_buf);
    ctx->ecache.d_buf = nullptr;
    vram_cache_free(&ctx->ecache);

    // Expert RAM cache (v2 module)
    if (ctx->rcache.buf) {
        if (ctx->rcache_registered) {
            (void)hipHostUnregister(ctx->rcache.buf);
            free(ctx->rcache.buf);
        } else if (ctx->rcache.pinned) {
            (void)hipHostFree(ctx->rcache.buf);
        } else {
            free(ctx->rcache.buf);
        }
        ctx->rcache.buf = nullptr;
    }
    ram_cache_free(&ctx->rcache);

    // Expert FFN buffers
    (void)hipFree(ctx->d_expert_staging);
    (void)hipFree(ctx->d_expert_ptrs);
    (void)hipFree(ctx->d_gate);
    (void)hipFree(ctx->d_up);
    (void)hipFree(ctx->d_down);
    (void)hipFree(ctx->d_scores);
    (void)hipFree(ctx->d_ffn_out);

    // Per-layer weights
    if (ctx->layers) {
        for (int i = 0; i < ctx->n_layers; i++) {
            struct gpu_layer *gl = &ctx->layers[i];
            (void)hipFree(gl->d_attn_norm);
            (void)hipFree(gl->d_ffn_norm);
            (void)hipFree(gl->d_router);
            if (gl->attn_type == LAYER_ATTN_FULL) {
                (void)hipFree(gl->d_wq);
                (void)hipFree(gl->d_wk);
                (void)hipFree(gl->d_wv);
                (void)hipFree(gl->d_wo);
                (void)hipFree(gl->d_q_norm);
                (void)hipFree(gl->d_k_norm);
            } else {
                (void)hipFree(gl->d_attn_qkv);
                (void)hipFree(gl->d_attn_gate);
                (void)hipFree(gl->d_ssm_a);
                (void)hipFree(gl->d_ssm_alpha);
                (void)hipFree(gl->d_ssm_beta);
                (void)hipFree(gl->d_ssm_conv1d);
                (void)hipFree(gl->d_ssm_dt_bias);
                (void)hipFree(gl->d_ssm_norm);
                (void)hipFree(gl->d_ssm_out);
            }
            (void)hipFree(gl->d_shared_expert_gate);
            (void)hipFree(gl->d_shared_gate);
            (void)hipFree(gl->d_shared_up);
            (void)hipFree(gl->d_shared_down);
        }
        free(ctx->layers);
    }

    // Global weights
    (void)hipFree(ctx->d_token_embd);
    (void)hipFree(ctx->d_output);
    (void)hipFree(ctx->d_output_norm);

    // KV cache
    (void)hipFree(ctx->d_kv_k);
    (void)hipFree(ctx->d_kv_v);
    free(ctx->full_attn_layer_ids);

    // SSM state
    (void)hipFree(ctx->d_ssm_states);
    (void)hipFree(ctx->d_ssm_conv_bufs);
    (void)hipFree(ctx->d_ssm_conv_pos);
    (void)hipHostFree(ctx->h_ssm_conv_pos);
    free(ctx->ssm_layer_ids);

    // Scratch
    (void)hipFree(ctx->d_x);
    (void)hipFree(ctx->d_norm);
    (void)hipFree(ctx->d_q);
    (void)hipFree(ctx->d_k);
    (void)hipFree(ctx->d_v);
    (void)hipFree(ctx->d_attn_out);
    (void)hipFree(ctx->d_attn_scores);
    (void)hipFree(ctx->d_attn_gate_buf);
    (void)hipFree(ctx->d_router_out);
    (void)hipFree(ctx->d_ffn_buf);
    (void)hipFree(ctx->d_logits);
    (void)hipFree(ctx->d_top_k_ids);
    (void)hipFree(ctx->d_top_k_vals);
    (void)hipFree(ctx->d_spec_router_out);
    (void)hipFree(ctx->d_spec_top_k_ids);
    (void)hipFree(ctx->d_spec_top_k_vals);
    (void)hipFree(ctx->d_ssm_dt);
    (void)hipFree(ctx->d_ssm_beta_buf);
    (void)hipFree(ctx->d_shared_gate_buf);
    (void)hipFree(ctx->d_shared_up_buf);
    (void)hipFree(ctx->d_shared_down_buf);

    // Pinned host
    (void)hipHostFree(ctx->h_router_out);
    (void)hipHostFree(ctx->h_top_k_ids);
    (void)hipHostFree(ctx->h_top_k_vals);
    (void)hipHostFree(ctx->h_spec_top_k_ids);
    (void)hipHostFree(ctx->h_all_scores);
    (void)hipHostFree(ctx->h_logits);

    // CPU expert compute
    if (ctx->h_norm) (void)hipHostFree(ctx->h_norm);
    if (ctx->h_cpu_partial) (void)hipHostFree(ctx->h_cpu_partial);
    (void)hipFree(ctx->d_cpu_partial);
    if (ctx->cpu_expert) cpu_expert_free((cpu_expert_ctx_t *)ctx->cpu_expert);

    // GPU1 expert engine cleanup
    if (ctx->n_devices >= 2) {
        (void)hipSetDevice(ctx->gpu1.device_id);

        // GPU1 VRAM cache
        (void)hipFree(ctx->gpu1.ecache.d_buf);
        ctx->gpu1.ecache.d_buf = nullptr;
        vram_cache_free(&ctx->gpu1.ecache);

        // GPU1 expert FFN buffers
        (void)hipFree(ctx->gpu1.d_expert_staging);
        (void)hipFree(ctx->gpu1.d_expert_ptrs);
        (void)hipFree(ctx->gpu1.d_gate);
        (void)hipFree(ctx->gpu1.d_up);
        (void)hipFree(ctx->gpu1.d_down);
        (void)hipFree(ctx->gpu1.d_scores);
        (void)hipFree(ctx->gpu1.d_ffn_out);

        // GPU1 scratch
        (void)hipFree(ctx->gpu1.d_norm);
        (void)hipFree(ctx->gpu1.d_partial);

        // GPU1 streams and events
        (void)hipEventDestroy(ctx->gpu1.cache_event);
        (void)hipEventDestroy(ctx->gpu1.batch_event[0]);
        (void)hipEventDestroy(ctx->gpu1.batch_event[1]);
        (void)hipStreamDestroy(ctx->gpu1.transfer_stream);
        (void)hipStreamDestroy(ctx->gpu1.stream);

        (void)hipSetDevice(0);
    }

    // Inter-GPU pinned staging
    if (ctx->h_gpu1_norm) (void)hipHostFree(ctx->h_gpu1_norm);
    if (ctx->h_gpu1_partial) (void)hipHostFree(ctx->h_gpu1_partial);
    (void)hipFree(ctx->d_gpu1_partial);

    (void)hipStreamDestroy(ctx->stream);
    free(ctx);
}

// ============================================================
// PUBLIC API: gpu_register_buffers / gpu_unregister_buffers
// ============================================================

extern "C" void gpu_register_buffers(gpu_ctx_t *ctx, void **buffers, int n, uint64_t size) {
    if (!ctx || !buffers || n <= 0) return;

    ctx->registered_bufs = (void **)calloc(n, sizeof(void *));
    ctx->n_registered = n;

    for (int i = 0; i < n; i++) {
        hipError_t err = hipHostRegister(buffers[i], size, hipHostRegisterDefault);
        if (err == hipSuccess) {
            ctx->registered_bufs[i] = buffers[i];
        } else {
            fprintf(stderr, "GPU: warning: failed to register buffer %d: %s\n",
                    i, hipGetErrorString(err));
            ctx->registered_bufs[i] = nullptr;
        }
    }
    fprintf(stderr, "GPU: registered %d expert buffers (%lu KB each)\n",
            n, (unsigned long)(size / 1024));
}

extern "C" void gpu_unregister_buffers(gpu_ctx_t *ctx, void **buffers, int n) {
    if (!ctx || !buffers) return;
    for (int i = 0; i < n && i < ctx->n_registered; i++) {
        if (ctx->registered_bufs && ctx->registered_bufs[i]) {
            (void)hipHostUnregister(ctx->registered_bufs[i]);
            ctx->registered_bufs[i] = nullptr;
        }
    }
}

// ============================================================
// PUBLIC API: gpu_expert_ffn (legacy, kept for CPU fallback path)
// ============================================================

extern "C" int gpu_expert_ffn(gpu_ctx_t *ctx,
                               const float *h, void **expert_buffers,
                               const float *expert_scores,
                               int n_experts, int n_embd, int expert_intermediate,
                               uint64_t expert_gate_size, uint64_t expert_up_size,
                               uint64_t expert_stride,
                               int gate_type, int up_type, int down_type,
                               float *ffn_out) {
    if (!ctx) return -1;

    if (n_experts > MAX_EXPERTS || n_embd > MAX_N_EMBD ||
        expert_intermediate > MAX_INTERMEDIATE ||
        expert_stride > MAX_EXPERT_STRIDE) {
        return -1;
    }

    if (gate_type != GGML_TYPE_Q4_K_ID && gate_type != GGML_TYPE_Q5_K_ID && gate_type != GGML_TYPE_Q6_K_ID) return -1;
    if (up_type   != GGML_TYPE_Q4_K_ID && up_type   != GGML_TYPE_Q5_K_ID && up_type   != GGML_TYPE_Q6_K_ID) return -1;
    if (down_type != GGML_TYPE_Q4_K_ID && down_type != GGML_TYPE_Q5_K_ID && down_type != GGML_TYPE_Q6_K_ID) return -1;

    hipStream_t s = ctx->stream;

    // H2D: input + expert data
    (void)hipMemcpyAsync(ctx->d_ffn_out, h, n_embd * sizeof(float),
                         hipMemcpyHostToDevice, s);  // reuse d_ffn_out as temp for h
    // Actually use a separate temp - use d_norm which is scratch
    float *d_h_temp = ctx->d_norm;  // borrow scratch buffer
    (void)hipMemcpyAsync(d_h_temp, h, n_embd * sizeof(float),
                         hipMemcpyHostToDevice, s);

    for (int e = 0; e < n_experts; e++) {
        (void)hipMemcpyAsync((uint8_t *)ctx->d_expert_staging + e * expert_stride,
                             expert_buffers[e], expert_stride,
                             hipMemcpyHostToDevice, s);
        ctx->h_expert_ptrs[e] = (uint8_t *)ctx->d_expert_staging + e * expert_stride;
    }
    (void)hipMemcpyAsync(ctx->d_expert_ptrs, ctx->h_expert_ptrs,
                         n_experts * sizeof(void *), hipMemcpyHostToDevice, s);

    // Gate — warp-per-row: Block(32, RPB), Grid(ceil(M/RPB), n_experts)
    constexpr int RPB = 8;
    dim3 wblock(32, RPB);
    dim3 grid_gate((expert_intermediate + RPB - 1) / RPB, n_experts);
    if (gate_type == GGML_TYPE_Q4_K_ID) {
        kernel_q4k_matvec_shared<<<grid_gate, wblock, 0, s>>>(
            ctx->d_expert_ptrs, d_h_temp, ctx->d_gate,
            expert_intermediate, n_embd, (uint64_t)0);
    } else {
        kernel_q6k_matvec_shared<<<grid_gate, wblock, 0, s>>>(
            ctx->d_expert_ptrs, d_h_temp, ctx->d_gate,
            expert_intermediate, n_embd, (uint64_t)0);
    }

    // Up
    dim3 grid_up((expert_intermediate + RPB - 1) / RPB, n_experts);
    if (up_type == GGML_TYPE_Q4_K_ID) {
        kernel_q4k_matvec_shared<<<grid_up, wblock, 0, s>>>(
            ctx->d_expert_ptrs, d_h_temp, ctx->d_up,
            expert_intermediate, n_embd, expert_gate_size);
    } else {
        kernel_q6k_matvec_shared<<<grid_up, wblock, 0, s>>>(
            ctx->d_expert_ptrs, d_h_temp, ctx->d_up,
            expert_intermediate, n_embd, expert_gate_size);
    }

    // SiLU * up
    int silu_n = n_experts * expert_intermediate;
    kernel_silu_mul<<<(silu_n + 255) / 256, 256, 0, s>>>(ctx->d_gate, ctx->d_up, silu_n);

    // Down
    dim3 grid_down((n_embd + RPB - 1) / RPB, n_experts);
    uint64_t down_offset = expert_gate_size + expert_up_size;
    if (down_type == GGML_TYPE_Q4_K_ID) {
        kernel_q4k_matvec_per_expert<<<grid_down, wblock, 0, s>>>(
            ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
            n_embd, expert_intermediate, down_offset);
    } else if (down_type == GGML_TYPE_Q5_K_ID) {
        kernel_q5k_matvec_per_expert<<<grid_down, wblock, 0, s>>>(
            ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
            n_embd, expert_intermediate, down_offset);
    } else {
        kernel_q6k_matvec_per_expert<<<grid_down, wblock, 0, s>>>(
            ctx->d_expert_ptrs, ctx->d_gate, ctx->d_down,
            n_embd, expert_intermediate, down_offset);
    }

    // Scores + weighted sum
    (void)hipMemcpyAsync(ctx->d_scores, expert_scores, n_experts * sizeof(float),
                         hipMemcpyHostToDevice, s);
    kernel_weighted_sum<<<(n_embd + 255) / 256, 256, 0, s>>>(
        ctx->d_down, ctx->d_scores, ctx->d_ffn_out, n_embd, n_experts);

    // D2H result
    (void)hipMemcpyAsync(ffn_out, ctx->d_ffn_out, n_embd * sizeof(float),
                         hipMemcpyDeviceToHost, s);

    hipError_t err = hipStreamSynchronize(s);
    if (err != hipSuccess) {
        fprintf(stderr, "GPU: stream sync failed: %s\n", hipGetErrorString(err));
        return -1;
    }

    return 0;
}
