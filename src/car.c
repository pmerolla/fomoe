#include "car.h"
#include <math.h>
#include <string.h>

void car_init(car_state_t *car, float threshold) {
    memset(car, 0, sizeof(*car));
    car->threshold = threshold;
    car->max_subs_per_layer = 0;  // 0 = unlimited
}

void car_reset_stats(car_state_t *car) {
    car->substitutions = 0;
    car->nvme_avoided = 0;
    car->nvme_remaining = 0;
    car->free_subs = 0;
    car->score_ratio_sum = 0.0;
}

void car_begin_token(car_state_t *car) {
    car->budget_pool = 0.0f;  // will accumulate via budget_per_layer each layer
}

// Candidate substitution (collected before sorting by quality)
typedef struct {
    int uncached_pos;     // position in uncached_indices[]
    int idx;              // index into expert_ids[]
    int orig_eid;
    int best_eid;
    float best_score;
    float ratio;
    int vram_slot;        // >=0 if from GPU0 VRAM, -1 otherwise
    int vram1_slot;       // >=0 if from GPU1 VRAM, -1 otherwise
    int ram_slot;         // >=0 if from RAM, -1 otherwise
} car_candidate_t;

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
) {
    // ram_layer: separate layer index for RAM cache (global layer ID in ping-pong)
    // If -1, use same as VRAM layer index
    int rl = (ram_layer >= 0) ? ram_layer : layer;

    *n_vram_subs = 0;
    if (n_vram1_subs) *n_vram1_subs = 0;
    *n_ram_subs = 0;

    if (car->threshold >= 1.0f || *n_uncached == 0)
        return 0;

    // Replenish budget pool for this layer
    // With layer_weight: edge layers get less replenishment (triangle scaling)
    if (car->budget_per_layer > 0.0f) {
        float replenish = car->budget_per_layer;
        if (car->layer_weight && car->n_layers > 0) {
            int half = car->n_layers / 2;
            int dist = (layer < half) ? layer : (car->n_layers - 1 - layer);
            // Edge layers get 1/n_layers fraction, middle layers get full
            replenish = car->budget_per_layer * (0.1f + 0.9f * dist / half);
        }
        car->budget_pool += replenish;
    }

    // Build set of already-selected experts
    uint8_t selected[64] = {}; // 512 bits
    for (int i = 0; i < n_experts; i++) {
        int eid = expert_ids[i];
        selected[eid >> 3] |= (1 << (eid & 7));
    }

    // Phase 1: Collect all candidate substitutions
    car_candidate_t candidates[16];
    int n_candidates = 0;

    for (int u = 0; u < *n_uncached; u++) {
        int idx = uncached_indices[u];
        int orig_eid = expert_ids[idx];
        float orig_score = all_scores[orig_eid];

        // Find highest-scoring cached expert not already selected
        int best_eid = -1;
        float best_score = 0.0f;
        int best_vram_slot = -1;
        int best_vram1_slot = -1;
        int best_ram_slot = -1;

        for (int e = 0; e < n_total_experts; e++) {
            if (selected[e >> 3] & (1 << (e & 7))) continue;
            float sc = all_scores[e];
            if (sc <= best_score) continue;

            // VRAM cache uses device-local layer index
            if (vcache && vcache->max_slots > 0) {
                int vs = vcache->map[layer * vcache->n_experts + e];
                if (vs >= 0) {
                    best_eid = e;
                    best_score = sc;
                    best_vram_slot = vs;
                    best_vram1_slot = -1;
                    best_ram_slot = -1;
                    continue;
                }
            }
            if (vcache1 && vcache1->max_slots > 0) {
                int vs1 = vcache1->map[layer * vcache1->n_experts + e];
                if (vs1 >= 0) {
                    best_eid = e;
                    best_score = sc;
                    best_vram_slot = -1;
                    best_vram1_slot = vs1;
                    best_ram_slot = -1;
                    continue;
                }
            }
            // RAM cache uses global layer index
            if (rcache && rcache->max_slots > 0) {
                int rs = rcache->map[rl * rcache->n_experts + e];
                if (rs >= 0) {
                    best_eid = e;
                    best_score = sc;
                    best_vram_slot = -1;
                    best_vram1_slot = -1;
                    best_ram_slot = rs;
                }
            }
        }

        float ratio = (orig_score > 0.0f) ? best_score / orig_score : 0.0f;
        if (best_eid >= 0 && ratio >= car->threshold) {
            car_candidate_t *c = &candidates[n_candidates++];
            c->uncached_pos = u;
            c->idx = idx;
            c->orig_eid = orig_eid;
            c->best_eid = best_eid;
            c->best_score = best_score;
            c->ratio = ratio;
            c->vram_slot = best_vram_slot;
            c->vram1_slot = best_vram1_slot;
            c->ram_slot = best_ram_slot;

            // Reserve this substitute so later candidates don't pick it
            selected[best_eid >> 3] |= (1 << (best_eid & 7));
        }
    }

    // Phase 2: Sort candidates by ratio (descending) — best substitutions first
    // Simple insertion sort (n <= 10)
    for (int i = 1; i < n_candidates; i++) {
        car_candidate_t tmp = candidates[i];
        int j = i - 1;
        while (j >= 0 && candidates[j].ratio < tmp.ratio) {
            candidates[j + 1] = candidates[j];
            j--;
        }
        candidates[j + 1] = tmp;
    }

    // Phase 3: Apply substitutions with budget control
    bool use_dynamic = (car->budget_per_layer > 0.0f);

    uint8_t substituted[16] = {};
    int subs = 0;

    for (int c = 0; c < n_candidates; c++) {
        car_candidate_t *cand = &candidates[c];
        bool is_free = use_dynamic && car->free_ratio > 0.0f &&
                       cand->ratio >= car->free_ratio;

        if (use_dynamic && !is_free) {
            if (car->budget_pool < 1.0f)
                break;
            car->budget_pool -= 1.0f;
        } else if (!use_dynamic && car->max_subs_per_layer > 0) {
            int budget = car->max_subs_per_layer;
            if (budget > 1 && car->layer_weight && car->n_layers > 0) {
                int half = car->n_layers / 2;
                int dist = (layer < half) ? layer : (car->n_layers - 1 - layer);
                int scaled = 1 + (budget - 1) * dist / half;
                if (scaled < budget) budget = scaled;
            }
            if (subs >= budget)
                break;
        }

        int idx = cand->idx;
        expert_ids[idx] = cand->best_eid;
        expert_scores[idx] = cand->best_score;

        car->substitutions++;
        car->nvme_avoided++;
        car->score_ratio_sum += cand->ratio;
        if (is_free) car->free_subs++;
        subs++;

        substituted[cand->uncached_pos] = 1;

        if (cand->vram_slot >= 0) {
            vram_sub_indices[*n_vram_subs] = idx;
            vram_sub_slots[*n_vram_subs] = cand->vram_slot;
            (*n_vram_subs)++;
        } else if (cand->vram1_slot >= 0 && vram1_sub_indices && n_vram1_subs) {
            vram1_sub_indices[*n_vram1_subs] = idx;
            vram1_sub_slots[*n_vram1_subs] = cand->vram1_slot;
            (*n_vram1_subs)++;
        } else {
            ram_sub_indices[*n_ram_subs] = idx;
            ram_sub_slots[*n_ram_subs] = cand->ram_slot;
            (*n_ram_subs)++;
        }
    }

    // Phase 4: Rebuild uncached list
    for (int c = subs; c < n_candidates; c++) {
        selected[candidates[c].best_eid >> 3] &= ~(1 << (candidates[c].best_eid & 7));
    }

    int new_uncached = 0;
    for (int u = 0; u < *n_uncached; u++) {
        if (!substituted[u]) {
            uncached_indices[new_uncached++] = uncached_indices[u];
            car->nvme_remaining++;
        }
    }
    *n_uncached = new_uncached;

    // Renormalize scores after substitution (unless dampening mode)
    if (subs > 0 && !car->skip_renorm) {
        float sum = 0.0f;
        for (int i = 0; i < n_experts; i++)
            sum += expert_scores[i];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int i = 0; i < n_experts; i++)
                expert_scores[i] *= inv;
        }
    }

    return subs;
}
