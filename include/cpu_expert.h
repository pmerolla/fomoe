#ifndef QMOE_CPU_EXPERT_H
#define QMOE_CPU_EXPERT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpu_expert_ctx cpu_expert_ctx_t;

// Work item for async batch submission
typedef struct {
    const void *expert_data;
    float score;
    int gate_type, up_type, down_type;
    uint64_t expert_gate_size, expert_up_size;
} cpu_expert_work_t;

// Create CPU expert context (allocates scratch buffers, starts worker thread)
// n_embd: hidden dimension (e.g., 4096)
// expert_intermediate: expert FFN intermediate size (e.g., 1024)
cpu_expert_ctx_t *cpu_expert_create(int n_embd, int expert_intermediate);

// Free CPU expert context (stops worker thread)
void cpu_expert_free(cpu_expert_ctx_t *ctx);

// Reset accumulated partial result to zero (call before each layer)
void cpu_expert_reset(cpu_expert_ctx_t *ctx);

// Compute one expert's FFN and accumulate weighted result (synchronous).
// expert_data: raw expert bytes (gate|up|down, contiguous)
// input: hidden state [n_embd] (host pointer, from D2H of d_norm)
// score: routing weight for this expert
// gate_type, up_type, down_type: ggml_dtype enum values
// expert_gate_size: byte offset from expert_data start to up weights
// expert_up_size: byte offset from up to down weights
void cpu_expert_ffn(cpu_expert_ctx_t *ctx,
                    const void *expert_data,
                    const float *input,
                    float score,
                    int gate_type, int up_type, int down_type,
                    uint64_t expert_gate_size, uint64_t expert_up_size);

// Submit batch of experts to background worker thread (non-blocking).
// Resets partial accumulator, then computes all work items asynchronously.
// input must remain valid until cpu_expert_wait() returns.
// work array is copied internally — caller's copy can go out of scope.
void cpu_expert_submit_async(cpu_expert_ctx_t *ctx, const float *input,
                             const cpu_expert_work_t *work, int n_work);

// Wait for background worker to finish computing (blocking).
// After return, cpu_expert_result() is valid.
void cpu_expert_wait(cpu_expert_ctx_t *ctx);

// Get pointer to accumulated partial result [n_embd]
// This is the weighted sum of all CPU-computed expert outputs.
float *cpu_expert_result(cpu_expert_ctx_t *ctx);

// Get worker thread timing stats (call after wait)
void cpu_expert_get_timing(cpu_expert_ctx_t *ctx, double *avg_wakeup_us,
                           double *avg_compute_us, int *samples);

#ifdef __cplusplus
}
#endif

#endif // QMOE_CPU_EXPERT_H
