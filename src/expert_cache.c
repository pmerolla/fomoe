#include "expert_cache.h"
#include "freq_profile.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---- Internal: shared LRU logic for both VRAM and RAM caches ----

// Initialize the metadata arrays for a layer-partitioned LRU cache.
// map: [n_layers * n_experts], slot arrays: [max_slots]
static int cache_meta_init(int **map, int **slot_layer, int **slot_expert,
                           uint64_t **slot_ts, int n_layers, int n_experts,
                           int max_slots) {
    int map_size = n_layers * n_experts;
    *map = malloc(map_size * sizeof(int));
    *slot_layer = malloc(max_slots * sizeof(int));
    *slot_expert = malloc(max_slots * sizeof(int));
    *slot_ts = calloc(max_slots, sizeof(uint64_t));

    if (!*map || !*slot_layer || !*slot_expert || !*slot_ts) {
        free(*map); free(*slot_layer); free(*slot_expert); free(*slot_ts);
        return -1;
    }

    memset(*map, 0xFF, map_size * sizeof(int)); // -1
    memset(*slot_layer, 0xFF, max_slots * sizeof(int));
    memset(*slot_expert, 0xFF, max_slots * sizeof(int));
    return 0;
}

static void cache_meta_free(int *map, int *slot_layer, int *slot_expert,
                            uint64_t *slot_ts) {
    free(map);
    free(slot_layer);
    free(slot_expert);
    free(slot_ts);
}

// Lookup in a layer-partitioned LRU. Returns slot or -1.
static inline int cache_lookup(int *map, uint64_t *slot_ts, uint64_t *ts,
                               int n_experts, int layer, int expert_id) {
    int slot = map[layer * n_experts + expert_id];
    if (slot >= 0)
        slot_ts[slot] = ++(*ts);
    return slot;
}

// Allocate a slot by evicting the LRU within the layer's partition.
// Returns slot index and updates map.
static int cache_alloc_slot(int *map, int *slot_layer, int *slot_expert,
                            uint64_t *slot_ts, uint64_t *ts,
                            int n_experts, int slots_per_layer,
                            int layer, int expert_id) {
    int base = layer * slots_per_layer;
    int end = base + slots_per_layer;

    // Find LRU slot in this layer's partition
    int lru = base;
    uint64_t lru_ts = slot_ts[base];
    for (int s = base + 1; s < end; s++) {
        if (slot_ts[s] < lru_ts) {
            lru_ts = slot_ts[s];
            lru = s;
        }
    }

    // Evict old entry from map
    int old_layer = slot_layer[lru];
    int old_expert = slot_expert[lru];
    if (old_layer >= 0 && old_expert >= 0)
        map[old_layer * n_experts + old_expert] = -1;

    // Install new entry
    slot_layer[lru] = layer;
    slot_expert[lru] = expert_id;
    slot_ts[lru] = ++(*ts);
    map[layer * n_experts + expert_id] = lru;

    return lru;
}

// ---- VRAM cache ----

int vram_cache_init(vram_cache_t *c, int n_layers, int n_experts,
                    int max_slots, uint64_t expert_stride) {
    memset(c, 0, sizeof(*c));
    c->n_layers = n_layers;
    c->n_experts = n_experts;
    c->max_slots = max_slots;
    c->slots_per_layer = max_slots / n_layers;
    c->expert_stride = expert_stride;

    if (c->slots_per_layer < 1) {
        fprintf(stderr, "vram_cache: max_slots=%d too small for %d layers\n",
                max_slots, n_layers);
        return -1;
    }

    // Adjust max_slots to be exact multiple of n_layers
    c->max_slots = c->slots_per_layer * n_layers;

    return cache_meta_init(&c->map, &c->slot_layer, &c->slot_expert,
                           &c->slot_ts, n_layers, n_experts, c->max_slots);
}

void vram_cache_free(vram_cache_t *c) {
    cache_meta_free(c->map, c->slot_layer, c->slot_expert, c->slot_ts);
    memset(c, 0, sizeof(*c));
}

int vram_cache_lookup(vram_cache_t *c, int layer, int expert_id) {
    int slot = cache_lookup(c->map, c->slot_ts, &c->ts,
                            c->n_experts, layer, expert_id);
    if (slot >= 0) c->hits++;
    else c->misses++;
    return slot;
}

int vram_cache_alloc_slot(vram_cache_t *c, int layer, int expert_id) {
    return cache_alloc_slot(c->map, c->slot_layer, c->slot_expert,
                            c->slot_ts, &c->ts, c->n_experts,
                            c->slots_per_layer, layer, expert_id);
}

void vram_cache_reset_stats(vram_cache_t *c) {
    c->hits = 0;
    c->misses = 0;
}

int vram_cache_seed(vram_cache_t *c, const struct freq_profile *fp) {
    if (!fp || !fp->entries) return 0;

    int seeded = 0;
    for (int layer = 0; layer < c->n_layers; layer++) {
        freq_entry_t top[64];
        int n = freq_profile_top_for_layer(fp, layer, top, c->slots_per_layer);

        for (int i = 0; i < n; i++) {
            // Allocate slot (no eviction concern since cache is empty at startup)
            int slot = vram_cache_alloc_slot(c, layer, top[i].expert_id);
            (void)slot;
            seeded++;
        }
    }

    fprintf(stderr, "vram_cache: seeded %d slots from frequency profile\n", seeded);
    return seeded;
}

// Seed with layer remapping: local_layer -> global_layer via layer_map.
// For ping-pong: GPU0 layer_map = {0,2,4,...,58}, GPU1 = {1,3,5,...,59}
int vram_cache_seed_mapped(vram_cache_t *c, const struct freq_profile *fp,
                           const int *layer_map) {
    if (!fp || !fp->entries || !layer_map) return 0;

    int seeded = 0;
    for (int local = 0; local < c->n_layers; local++) {
        int global = layer_map[local];
        freq_entry_t top[64];
        int n = freq_profile_top_for_layer(fp, global, top, c->slots_per_layer);

        for (int i = 0; i < n; i++) {
            int slot = vram_cache_alloc_slot(c, local, top[i].expert_id);
            (void)slot;
            seeded++;
        }
    }

    fprintf(stderr, "vram_cache: seeded %d slots (mapped) from frequency profile\n", seeded);
    return seeded;
}

// ---- RAM cache ----

int ram_cache_init(ram_cache_t *c, int n_layers, int n_experts,
                   int max_slots, uint64_t expert_stride) {
    memset(c, 0, sizeof(*c));
    c->n_layers = n_layers;
    c->n_experts = n_experts;
    c->max_slots = max_slots;
    c->slots_per_layer = max_slots / n_layers;
    c->expert_stride = expert_stride;

    if (c->slots_per_layer < 1) {
        fprintf(stderr, "ram_cache: max_slots=%d too small for %d layers\n",
                max_slots, n_layers);
        return -1;
    }

    c->max_slots = c->slots_per_layer * n_layers;

    return cache_meta_init(&c->map, &c->slot_layer, &c->slot_expert,
                           &c->slot_ts, n_layers, n_experts, c->max_slots);
}

void ram_cache_free(ram_cache_t *c) {
    cache_meta_free(c->map, c->slot_layer, c->slot_expert, c->slot_ts);
    memset(c, 0, sizeof(*c));
}

int ram_cache_lookup(ram_cache_t *c, int layer, int expert_id) {
    int slot = cache_lookup(c->map, c->slot_ts, &c->ts,
                            c->n_experts, layer, expert_id);
    if (slot >= 0) c->hits++;
    else c->misses++;
    return slot;
}

int ram_cache_alloc_slot(ram_cache_t *c, int layer, int expert_id) {
    return cache_alloc_slot(c->map, c->slot_layer, c->slot_expert,
                            c->slot_ts, &c->ts, c->n_experts,
                            c->slots_per_layer, layer, expert_id);
}

void ram_cache_reset_stats(ram_cache_t *c) {
    c->hits = 0;
    c->misses = 0;
}
