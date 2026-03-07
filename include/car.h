#ifndef QMOE_CAR_H
#define QMOE_CAR_H

#include "expert_cache.h"

// Cache-aware routing state and statistics
typedef struct {
    float    threshold;         // 0.0 = always substitute, 1.0 = never
    int      max_subs_per_layer; // 0 = unlimited, >0 = quality budget
    int      n_layers;          // total layers (for position-aware budget)
    bool     skip_renorm;       // don't renormalize scores after substitution (dampening)
    bool     layer_weight;      // scale budget by layer position (tight early/late, loose middle)
    int      warmup_tokens;    // force CAR=1.0 for this many tokens (0 = disabled)

    // Dynamic accumulated budget (when > 0)
    float    free_ratio;        // subs with ratio >= this are free (don't cost budget)
    float    budget_pool;       // remaining budget across layers (reset per token)
    float    budget_per_layer;  // replenishment per layer
    uint64_t substitutions;
    uint64_t nvme_avoided;
    uint64_t nvme_remaining;
    uint64_t free_subs;         // subs that were "free" (ratio >= free_ratio)
    double   score_ratio_sum;
} car_state_t;

void car_init(car_state_t *car, float threshold);
void car_reset_stats(car_state_t *car);
void car_begin_token(car_state_t *car);  // reset per-token budget pool

// Evaluate CAR substitutions for uncached experts.
//
// For each expert in uncached_ids[0..n_uncached-1], find the highest-scoring
// cached expert (VRAM or RAM) not already in expert_ids, check score ratio
// against threshold, and substitute if acceptable.
//
// On return:
//   - expert_ids[] and expert_scores[] are updated with substitutions
//   - expert_scores[] are renormalized
//   - uncached_ids/uncached_count updated to remaining NVMe-miss experts
//   - newly_cached_ids/slots/count filled with substitutes found in VRAM
//   - newly_ramhit_ids/slots/count filled with substitutes found in RAM
//
// Returns number of substitutions made.
// layer: layer index for VRAM cache lookup (device-local in ping-pong)
// ram_layer: layer index for RAM cache lookup (global); pass -1 to use same as layer
int car_evaluate(
    car_state_t *car,
    int layer,
    int ram_layer,
    // Expert selection arrays (modified in place)
    int *expert_ids,            // [n_experts] selected expert IDs
    float *expert_scores,       // [n_experts] routing scores
    int n_experts,              // number of active experts
    // Full score array for finding substitutes
    const float *all_scores,    // [n_total_experts] softmax scores
    int n_total_experts,
    // Caches to check (vcache1 can be NULL for single-GPU)
    const vram_cache_t *vcache,
    const vram_cache_t *vcache1,
    const ram_cache_t *rcache,
    // Uncached expert tracking (in/out)
    int *uncached_indices,      // [n_uncached] indices into expert_ids[] (in/out)
    int *n_uncached,            // count of uncached (in/out)
    // Output: substitutes found in GPU0 VRAM cache
    int *vram_sub_indices,      // indices into expert_ids[] for VRAM substitutes
    int *vram_sub_slots,        // VRAM cache slot for each
    int *n_vram_subs,
    // Output: substitutes found in GPU1 VRAM cache
    int *vram1_sub_indices,     // indices for GPU1 VRAM substitutes (can be NULL)
    int *vram1_sub_slots,       // GPU1 VRAM cache slot for each (can be NULL)
    int *n_vram1_subs,          // count (can be NULL)
    // Output: substitutes found in RAM cache
    int *ram_sub_indices,       // indices into expert_ids[] for RAM substitutes
    int *ram_sub_slots,         // RAM cache slot for each
    int *n_ram_subs
);

#endif // QMOE_CAR_H
