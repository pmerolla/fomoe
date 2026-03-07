#ifndef QMOE_GPU_H
#define QMOE_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_ctx gpu_ctx_t;

// Initialize GPU context: select device, create stream, allocate device memory
gpu_ctx_t *gpu_init(void);

// Free GPU context and all device memory
void gpu_free(gpu_ctx_t *ctx);

// Pin existing host buffers for faster H2D transfer
void gpu_register_buffers(gpu_ctx_t *ctx, void **buffers, int n, uint64_t size);

// Unpin previously registered host buffers
void gpu_unregister_buffers(gpu_ctx_t *ctx, void **buffers, int n);

// Upload all non-expert weights to GPU VRAM + allocate KV cache + scratch buffers
// Call once after model_load()
int gpu_upload_model(gpu_ctx_t *ctx, const void *model);

// Set RAM cache size in MB (pinned host memory for expert caching)
// Call before gpu_upload_model(). 0 = disabled, -1 = auto (default)
void gpu_set_ram_cache_mb(gpu_ctx_t *ctx, int mb);

// Set VRAM cache size in MB (GPU memory for expert caching)
// Call before gpu_upload_model(). 0 = disabled, -1 = auto (default)
void gpu_set_vram_cache_mb(gpu_ctx_t *ctx, int mb);

// Full forward pass on GPU (CPU only does NVMe expert I/O)
// Returns pointer to host logits, or NULL on error
float *gpu_forward(gpu_ctx_t *ctx, void *model, int token_id, int pos);

// Reset KV cache and SSM state on GPU (for chunked perplexity evaluation)
void gpu_reset_state(gpu_ctx_t *ctx);

// Seed VRAM cache from frequency profile.
// Loads top experts per layer from NVMe directly into VRAM cache.
// Returns number of experts seeded, or -1 on error.
int gpu_seed_vram_cache(gpu_ctx_t *ctx, void *model, const char *freq_profile_path);

// Enable expert frequency counting for profile generation.
// Call before any forward passes. Allocates [n_layers * n_experts] counter.
void gpu_enable_expert_freq(gpu_ctx_t *ctx);

// Get expert frequency counts. Returns pointer to [n_layers * n_experts] array,
// or NULL if counting not enabled. Caller must NOT free the returned pointer.
const uint32_t *gpu_get_expert_freq(gpu_ctx_t *ctx);

// Get n_layers and n_experts for interpreting the freq array.
void gpu_get_expert_dims(gpu_ctx_t *ctx, int *n_layers, int *n_experts);

// Dispatch batched expert FFN to GPU (legacy, still used by gpu_forward internally)
// Returns 0 on success, -1 to fall back to CPU
int gpu_expert_ffn(gpu_ctx_t *ctx,
                   const float *h,              // [n_embd] input hidden state (CPU)
                   void **expert_buffers,        // n_experts quantized expert data ptrs (pinned CPU)
                   const float *expert_scores,   // n_experts routing weights
                   int n_experts,                // number of active experts (e.g. 8)
                   int n_embd,                   // hidden dimension
                   int expert_intermediate,      // expert FFN intermediate size
                   uint64_t expert_gate_size,    // bytes of gate projection per expert
                   uint64_t expert_up_size,      // bytes of up projection per expert
                   uint64_t expert_stride,       // total bytes per expert in store
                   int gate_type,                // ggml_dtype for gate weights
                   int up_type,                  // ggml_dtype for up weights
                   int down_type,                // ggml_dtype for down weights
                   float *ffn_out);              // [n_embd] output (CPU)

#ifdef __cplusplus
}
#endif

#endif // QMOE_GPU_H
