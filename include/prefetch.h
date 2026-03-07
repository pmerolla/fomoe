#ifndef QMOE_PREFETCH_H
#define QMOE_PREFETCH_H

#include "expert_cache.h"
#include "nvme_io.h"
#include <stdbool.h>
#include <pthread.h>

#define MAX_PREFETCH_PER_LAYER 32

// Where a prefetched expert is coming from
typedef enum {
    PREFETCH_SRC_VRAM,      // already in VRAM, no action needed
    PREFETCH_SRC_RAM,       // in RAM cache, needs H2D
    PREFETCH_SRC_NVME,      // on NVMe, needs read → RAM copy → H2D
} prefetch_src_t;

// Single prefetch entry
typedef struct {
    int            layer;
    int            expert_id;
    float          score;         // predicted routing score
    prefetch_src_t src;
    int            vram_slot;     // reserved VRAM slot (for RAM and NVMe sources)
    int            ram_slot;      // RAM cache slot (NVMe target or RAM source)
    bool           nvme_complete; // NVMe read + RAM copy done
    bool           h2d_complete;  // H2D transfer done (set by caller)
} prefetch_entry_t;

// Persistent worker thread for async NVMe prefetch
typedef struct prefetch_worker {
    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  cond_work;
    pthread_cond_t  cond_done;
    bool            work_ready;
    bool            work_done;
    bool            shutdown;
    bool            started;

    // Work parameters (set by main thread before signaling)
    nvme_io_t      *nvme_io;
    ram_cache_t    *rcache;       // for post-read RAM copy
    int             layer;
    int             expert_ids[MAX_PREFETCH_PER_LAYER];
    int             ram_slots[MAX_PREFETCH_PER_LAYER];
    int             n_experts;
    int             buffer_offset;
    void           *out_buffers[MAX_PREFETCH_PER_LAYER];
} prefetch_worker_t;

// Prefetch pipeline state
typedef struct prefetch_state {
    prefetch_entry_t entries[MAX_PREFETCH_PER_LAYER];
    int              n_entries;
    int              nvme_budget;      // max NVMe reads per prefetch window
    int              nvme_submitted;
    int              spec_k;           // over-prediction factor (K=18 for 397B)
    int              nvme_buf_offset;  // starting buffer index in NVMe pool

    // Counts per source type (set after classify)
    int              n_vram;
    int              n_ram;
    int              n_nvme;

    prefetch_worker_t worker;

    // Cumulative stats
    uint64_t stat_vram_hits;
    uint64_t stat_ram_hits;
    uint64_t stat_nvme_reads;
    uint64_t stat_budget_skips;  // experts skipped due to budget exhaustion
} prefetch_state_t;

// Initialize prefetch state and start worker thread.
// nvme_io and rcache pointers are stored for the worker.
// nvme_buf_offset: starting index in NVMe buffer pool for prefetch reads.
int prefetch_init(prefetch_state_t *ps, int nvme_budget, int spec_k,
                  nvme_io_t *nvme_io, ram_cache_t *rcache, int nvme_buf_offset);

// Shutdown worker thread and cleanup.
void prefetch_free(prefetch_state_t *ps);

// Classify predicted experts and submit NVMe reads.
// predicted_ids: top-K predicted expert IDs for next layer (sorted by score desc).
// predicted_scores: corresponding routing scores (can be NULL).
//
// For each predicted expert:
//   - VRAM hit → record as SRC_VRAM, touch LRU
//   - RAM hit → reserve VRAM slot, record as SRC_RAM
//   - NVMe + budget → reserve RAM+VRAM slots, submit async read
//   - NVMe + no budget → skip (CAR handles at actual routing)
//
// NVMe reads are dispatched to the worker thread (non-blocking).
// Returns number of NVMe reads submitted.
int prefetch_classify_and_submit(
    prefetch_state_t *ps,
    int next_layer,
    const int *predicted_ids,
    const float *predicted_scores,
    int n_predicted,
    vram_cache_t *vcache,
    vram_cache_t *vcache1,       // GPU1 VRAM cache (can be NULL)
    ram_cache_t *rcache
);

// Wait for NVMe worker to finish all reads for current batch.
// After this returns, all PREFETCH_SRC_NVME entries have nvme_complete=true
// and their data is in the RAM cache slots.
void prefetch_wait_nvme(prefetch_state_t *ps);

// Non-blocking check: returns true if NVMe worker has finished.
// If finished, marks entries as nvme_complete (same as wait, but no blocking).
bool prefetch_try_complete_nvme(prefetch_state_t *ps);

// Check if a specific expert was prefetched.
// Returns pointer to the entry if found (any source), NULL otherwise.
const prefetch_entry_t *prefetch_find(const prefetch_state_t *ps,
                                       int layer, int expert_id);

// Get entries that need H2D transfers (SRC_RAM and SRC_NVME with nvme_complete).
// Writes up to max_out entries to out[]. Returns count.
// Caller performs hipMemcpyAsync from ram_cache_slot_ptr → vram_cache_slot_ptr.
int prefetch_get_h2d_pending(const prefetch_state_t *ps,
                             const prefetch_entry_t **out, int max_out);

// Mark an entry's H2D as complete (called by GPU-side code after hipMemcpy).
void prefetch_mark_h2d_done(prefetch_state_t *ps, int layer, int expert_id);

// Reset state for next layer's predictions.
void prefetch_reset(prefetch_state_t *ps);

// Heap-allocating create/destroy (for opaque pointer usage from GPU code)
prefetch_state_t *prefetch_create(int nvme_budget, int spec_k,
                                   nvme_io_t *nvme_io, ram_cache_t *rcache,
                                   int nvme_buf_offset);
void prefetch_destroy(prefetch_state_t *ps);

// Get RAM→VRAM H2D transfers ready now (SRC_RAM entries after classify).
// Returns count. vram_slots[i] and ram_slots[i] filled for each.
int prefetch_get_ram_transfers(const prefetch_state_t *ps,
                                int *vram_slots, int *ram_slots, int max_out);

// Get NVMe→VRAM H2D transfers ready after wait (SRC_NVME entries).
// Returns count. vram_slots[i] and ram_slots[i] filled for each.
int prefetch_get_nvme_transfers(const prefetch_state_t *ps,
                                 int *vram_slots, int *ram_slots, int max_out);

// Mark all pending H2D transfers as done.
void prefetch_mark_all_h2d_done(prefetch_state_t *ps);

// Total count of entries that received VRAM transfers (RAM + NVMe sources).
int prefetch_vram_transfer_count(const prefetch_state_t *ps);

// Commit RAM cache map entries for completed NVMe prefetch reads.
// Call AFTER prefetch_try_complete_nvme() returns true, to make
// prefetched experts visible as RAM cache hits to the main pipeline.
void prefetch_commit_ram_cache(prefetch_state_t *ps, ram_cache_t *rcache);

#endif // QMOE_PREFETCH_H
