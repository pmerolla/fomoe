#ifndef QMOE_EXPERT_CACHE_H
#define QMOE_EXPERT_CACHE_H

#include <stdint.h>
#include <stdbool.h>

// Forward declarations
struct freq_profile;

// VRAM expert cache (layer-partitioned LRU)
// d_buf must be allocated by GPU code before use
typedef struct {
    void     *d_buf;            // contiguous VRAM allocation (set externally)
    int       max_slots;
    int       slots_per_layer;
    int      *map;              // [n_layers * n_experts] → slot index or -1
    int      *slot_layer;       // [max_slots] → layer (-1 if empty)
    int      *slot_expert;      // [max_slots] → expert_id (-1 if empty)
    uint64_t *slot_ts;          // [max_slots] → access timestamp
    uint64_t  ts;               // global timestamp counter
    int       n_layers;
    int       n_experts;        // per layer (total experts, not active)
    uint64_t  expert_stride;    // bytes per expert
    uint64_t  hits;
    uint64_t  misses;
} vram_cache_t;

// RAM expert cache (layer-partitioned LRU, pinned host memory)
typedef struct {
    void     *buf;              // contiguous host memory (set externally or by init)
    bool      pinned;           // true if hipHostRegister'd
    int       max_slots;
    int       slots_per_layer;
    int      *map;              // [n_layers * n_experts] → slot index or -1
    int      *slot_layer;       // [max_slots]
    int      *slot_expert;      // [max_slots]
    uint64_t *slot_ts;          // [max_slots]
    uint64_t  ts;
    int       n_layers;
    int       n_experts;        // per layer
    uint64_t  expert_stride;
    uint64_t  hits;
    uint64_t  misses;
} ram_cache_t;

// Initialize cache metadata (allocates map/slot arrays, zeroes stats).
// Does NOT allocate d_buf or buf — caller must set those.
int vram_cache_init(vram_cache_t *c, int n_layers, int n_experts,
                    int max_slots, uint64_t expert_stride);
int ram_cache_init(ram_cache_t *c, int n_layers, int n_experts,
                   int max_slots, uint64_t expert_stride);

// Free metadata arrays (does NOT free d_buf or buf)
void vram_cache_free(vram_cache_t *c);
void ram_cache_free(ram_cache_t *c);

// Lookup: returns slot index or -1. Bumps timestamp on hit.
int vram_cache_lookup(vram_cache_t *c, int layer, int expert_id);
int ram_cache_lookup(ram_cache_t *c, int layer, int expert_id);

// Pointer to expert data in a given slot
static inline void *vram_cache_slot_ptr(const vram_cache_t *c, int slot) {
    return (char *)c->d_buf + (uint64_t)slot * c->expert_stride;
}
static inline void *ram_cache_slot_ptr(const ram_cache_t *c, int slot) {
    return (char *)c->buf + (uint64_t)slot * c->expert_stride;
}

// Allocate a slot for (layer, expert_id) by evicting the LRU in that layer.
// Returns slot index. Updates map. Caller must fill the slot with data.
int vram_cache_alloc_slot(vram_cache_t *c, int layer, int expert_id);
int ram_cache_alloc_slot(ram_cache_t *c, int layer, int expert_id);

// Touch: bump timestamp without lookup overhead (already know the slot)
static inline void vram_cache_touch(vram_cache_t *c, int slot) {
    c->slot_ts[slot] = ++c->ts;
}
static inline void ram_cache_touch(ram_cache_t *c, int slot) {
    c->slot_ts[slot] = ++c->ts;
}

// Seed VRAM cache from frequency profile (fills slots with most frequent experts).
// Returns number of entries seeded. Caller must H2D the actual data separately.
int vram_cache_seed(vram_cache_t *c, const struct freq_profile *fp);

// Seed with layer remapping for ping-pong: local_layer -> global_layer via layer_map.
// GPU0 layer_map = {0,2,4,...,58}, GPU1 = {1,3,5,...,59}
int vram_cache_seed_mapped(vram_cache_t *c, const struct freq_profile *fp,
                           const int *layer_map);

// Reset hit/miss counters
void vram_cache_reset_stats(vram_cache_t *c);
void ram_cache_reset_stats(ram_cache_t *c);

#endif // QMOE_EXPERT_CACHE_H
