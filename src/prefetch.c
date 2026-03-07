#include "prefetch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---- Worker thread: runs NVMe reads + copies to RAM cache ----

static void *prefetch_worker_fn(void *arg) {
    prefetch_worker_t *w = arg;

    pthread_mutex_lock(&w->mutex);
    while (!w->shutdown) {
        // Wait for work
        while (!w->work_ready && !w->shutdown)
            pthread_cond_wait(&w->cond_work, &w->mutex);

        if (w->shutdown) break;

        // Snapshot work params
        nvme_io_t   *nvme_io = w->nvme_io;
        ram_cache_t *rcache  = w->rcache;
        int          layer   = w->layer;
        int          n       = w->n_experts;
        int          buf_off = w->buffer_offset;

        int expert_ids[MAX_PREFETCH_PER_LAYER];
        int ram_slots[MAX_PREFETCH_PER_LAYER];
        memcpy(expert_ids, w->expert_ids, n * sizeof(int));
        memcpy(ram_slots, w->ram_slots, n * sizeof(int));

        w->work_ready = false;
        pthread_mutex_unlock(&w->mutex);

        // Do NVMe reads using ring2 (concurrent-safe with main pipeline's ring1)
        void *out_buffers[MAX_PREFETCH_PER_LAYER];
        nvme_io_load_experts_prefetch(nvme_io, layer, expert_ids,
                                       n, buf_off, out_buffers);

        // Copy loaded experts to RAM cache slots
        for (int i = 0; i < n; i++) {
            int slot = ram_slots[i];
            if (slot >= 0 && rcache && rcache->buf) {
                void *dest = ram_cache_slot_ptr(rcache, slot);
                memcpy(dest, out_buffers[i], rcache->expert_stride);
            }
        }

        // Store output buffer pointers for potential direct H2D
        pthread_mutex_lock(&w->mutex);
        memcpy(w->out_buffers, out_buffers, n * sizeof(void *));
        w->work_done = true;
        pthread_cond_signal(&w->cond_done);
    }
    pthread_mutex_unlock(&w->mutex);
    return NULL;
}

// ---- Public API ----

int prefetch_init(prefetch_state_t *ps, int nvme_budget, int spec_k,
                  nvme_io_t *nvme_io, ram_cache_t *rcache, int nvme_buf_offset) {
    memset(ps, 0, sizeof(*ps));
    ps->nvme_budget = nvme_budget;
    ps->spec_k = spec_k;
    ps->nvme_buf_offset = nvme_buf_offset;

    // Init worker
    prefetch_worker_t *w = &ps->worker;
    w->nvme_io = nvme_io;
    w->rcache = rcache;
    w->buffer_offset = nvme_buf_offset;
    w->work_done = true;  // so first submit doesn't block

    pthread_mutex_init(&w->mutex, NULL);
    pthread_cond_init(&w->cond_work, NULL);
    pthread_cond_init(&w->cond_done, NULL);

    if (nvme_io) {
        int rc = pthread_create(&w->thread, NULL, prefetch_worker_fn, w);
        if (rc != 0) {
            fprintf(stderr, "prefetch: failed to create worker thread\n");
            return -1;
        }
        w->started = true;
    }

    return 0;
}

void prefetch_free(prefetch_state_t *ps) {
    prefetch_worker_t *w = &ps->worker;
    if (w->started) {
        pthread_mutex_lock(&w->mutex);
        w->shutdown = true;
        pthread_cond_signal(&w->cond_work);
        pthread_mutex_unlock(&w->mutex);
        pthread_join(w->thread, NULL);
    }
    pthread_mutex_destroy(&w->mutex);
    pthread_cond_destroy(&w->cond_work);
    pthread_cond_destroy(&w->cond_done);
}

int prefetch_classify_and_submit(
    prefetch_state_t *ps,
    int next_layer,
    const int *predicted_ids,
    const float *predicted_scores,
    int n_predicted,
    vram_cache_t *vcache,
    vram_cache_t *vcache1,
    ram_cache_t *rcache
) {
    prefetch_reset(ps);

    // Worker NVMe queue
    int nvme_ids[MAX_PREFETCH_PER_LAYER];
    int nvme_ram_slots[MAX_PREFETCH_PER_LAYER];
    int nvme_count = 0;

    int limit = n_predicted < MAX_PREFETCH_PER_LAYER ? n_predicted : MAX_PREFETCH_PER_LAYER;

    for (int i = 0; i < limit; i++) {
        int eid = predicted_ids[i];
        prefetch_entry_t *e = &ps->entries[ps->n_entries];

        e->layer = next_layer;
        e->expert_id = eid;
        e->score = predicted_scores ? predicted_scores[i] : 0.0f;
        e->h2d_complete = false;
        e->nvme_complete = false;
        e->vram_slot = -1;
        e->ram_slot = -1;

        // Check GPU0 VRAM cache
        if (vcache && vcache->max_slots > 0) {
            int vslot = vcache->map[next_layer * vcache->n_experts + eid];
            if (vslot >= 0) {
                e->src = PREFETCH_SRC_VRAM;
                e->vram_slot = vslot;
                vram_cache_touch(vcache, vslot);
                e->nvme_complete = true;
                e->h2d_complete = true;
                ps->n_vram++;
                ps->stat_vram_hits++;
                ps->n_entries++;
                continue;
            }
        }

        // Check GPU1 VRAM cache — expert already cached on GPU1, no prefetch needed
        if (vcache1 && vcache1->max_slots > 0) {
            int vslot1 = vcache1->map[next_layer * vcache1->n_experts + eid];
            if (vslot1 >= 0) {
                e->src = PREFETCH_SRC_VRAM;
                e->vram_slot = vslot1;
                vram_cache_touch(vcache1, vslot1);
                e->nvme_complete = true;
                e->h2d_complete = true;
                ps->n_vram++;
                ps->stat_vram_hits++;
                ps->n_entries++;
                continue;
            }
        }

        // Check RAM cache
        if (rcache && rcache->max_slots > 0) {
            int rslot = rcache->map[next_layer * rcache->n_experts + eid];
            if (rslot >= 0) {
                e->src = PREFETCH_SRC_RAM;
                e->ram_slot = rslot;
                e->vram_slot = vram_cache_alloc_slot(vcache, next_layer, eid);
                ram_cache_touch(rcache, rslot);
                e->nvme_complete = true;  // data in RAM, just needs H2D
                ps->n_ram++;
                ps->stat_ram_hits++;
                ps->n_entries++;
                continue;
            }
        }

        // NVMe needed — check budget
        if (ps->nvme_submitted >= ps->nvme_budget) {
            ps->stat_budget_skips++;
            continue;  // skip, CAR handles at actual routing time
        }

        e->src = PREFETCH_SRC_NVME;
        e->ram_slot = rcache ? ram_cache_alloc_slot(rcache, next_layer, eid) : -1;
        e->vram_slot = vcache ? vram_cache_alloc_slot(vcache, next_layer, eid) : -1;

        // Hide RAM slot from map until NVMe data is written.
        // Prevents main pipeline from seeing stale data as a RAM hit.
        // Restored by prefetch_commit_ram_cache() after NVMe completion.
        if (e->ram_slot >= 0 && rcache)
            rcache->map[next_layer * rcache->n_experts + eid] = -1;

        // Queue for worker
        nvme_ram_slots[nvme_count] = e->ram_slot;
        nvme_ids[nvme_count] = eid;
        nvme_count++;
        ps->nvme_submitted++;
        ps->n_nvme++;
        ps->stat_nvme_reads++;
        ps->n_entries++;
    }

    // Dispatch NVMe reads to worker thread
    if (nvme_count > 0 && ps->worker.started) {
        prefetch_worker_t *w = &ps->worker;

        // Wait for any previous work to complete first
        // Must wait here to avoid io_uring buffer conflicts
        pthread_mutex_lock(&w->mutex);
        while (!w->work_done)
            pthread_cond_wait(&w->cond_done, &w->mutex);

        w->layer = next_layer;
        w->n_experts = nvme_count;
        w->buffer_offset = ps->nvme_buf_offset;
        memcpy(w->expert_ids, nvme_ids, nvme_count * sizeof(int));
        memcpy(w->ram_slots, nvme_ram_slots, nvme_count * sizeof(int));
        w->work_done = false;
        w->work_ready = true;
        pthread_cond_signal(&w->cond_work);
        pthread_mutex_unlock(&w->mutex);
    }

    return nvme_count;
}

void prefetch_wait_nvme(prefetch_state_t *ps) {
    if (ps->nvme_submitted == 0) return;

    prefetch_worker_t *w = &ps->worker;
    if (!w->started) return;

    pthread_mutex_lock(&w->mutex);
    while (!w->work_done)
        pthread_cond_wait(&w->cond_done, &w->mutex);
    pthread_mutex_unlock(&w->mutex);

    // Mark all NVMe entries as complete
    for (int i = 0; i < ps->n_entries; i++) {
        if (ps->entries[i].src == PREFETCH_SRC_NVME)
            ps->entries[i].nvme_complete = true;
    }
}

bool prefetch_try_complete_nvme(prefetch_state_t *ps) {
    if (ps->nvme_submitted == 0) return true;

    prefetch_worker_t *w = &ps->worker;
    if (!w->started) return true;

    pthread_mutex_lock(&w->mutex);
    bool done = w->work_done;
    pthread_mutex_unlock(&w->mutex);

    if (done) {
        for (int i = 0; i < ps->n_entries; i++) {
            if (ps->entries[i].src == PREFETCH_SRC_NVME)
                ps->entries[i].nvme_complete = true;
        }
    }
    return done;
}

const prefetch_entry_t *prefetch_find(const prefetch_state_t *ps,
                                       int layer, int expert_id) {
    for (int i = 0; i < ps->n_entries; i++) {
        const prefetch_entry_t *e = &ps->entries[i];
        if (e->layer == layer && e->expert_id == expert_id)
            return e;
    }
    return NULL;
}

int prefetch_get_h2d_pending(const prefetch_state_t *ps,
                             const prefetch_entry_t **out, int max_out) {
    int count = 0;
    for (int i = 0; i < ps->n_entries && count < max_out; i++) {
        const prefetch_entry_t *e = &ps->entries[i];
        if (e->src == PREFETCH_SRC_VRAM) continue;  // already in VRAM
        if (e->h2d_complete) continue;               // already transferred
        if (!e->nvme_complete) continue;             // NVMe not done yet
        out[count++] = e;
    }
    return count;
}

void prefetch_mark_h2d_done(prefetch_state_t *ps, int layer, int expert_id) {
    for (int i = 0; i < ps->n_entries; i++) {
        prefetch_entry_t *e = &ps->entries[i];
        if (e->layer == layer && e->expert_id == expert_id) {
            e->h2d_complete = true;
            return;
        }
    }
}

void prefetch_reset(prefetch_state_t *ps) {
    ps->n_entries = 0;
    ps->nvme_submitted = 0;
    ps->n_vram = 0;
    ps->n_ram = 0;
    ps->n_nvme = 0;
}

prefetch_state_t *prefetch_create(int nvme_budget, int spec_k,
                                   nvme_io_t *nvme_io, ram_cache_t *rcache,
                                   int nvme_buf_offset) {
    prefetch_state_t *ps = calloc(1, sizeof(prefetch_state_t));
    if (!ps) return NULL;
    if (prefetch_init(ps, nvme_budget, spec_k, nvme_io, rcache, nvme_buf_offset) != 0) {
        free(ps);
        return NULL;
    }
    return ps;
}

void prefetch_destroy(prefetch_state_t *ps) {
    if (!ps) return;
    prefetch_free(ps);
    free(ps);
}

int prefetch_get_ram_transfers(const prefetch_state_t *ps,
                                int *vram_slots, int *ram_slots, int max_out) {
    int count = 0;
    for (int i = 0; i < ps->n_entries && count < max_out; i++) {
        const prefetch_entry_t *e = &ps->entries[i];
        if (e->src == PREFETCH_SRC_RAM && !e->h2d_complete) {
            vram_slots[count] = e->vram_slot;
            ram_slots[count] = e->ram_slot;
            count++;
        }
    }
    return count;
}

int prefetch_get_nvme_transfers(const prefetch_state_t *ps,
                                 int *vram_slots, int *ram_slots, int max_out) {
    int count = 0;
    for (int i = 0; i < ps->n_entries && count < max_out; i++) {
        const prefetch_entry_t *e = &ps->entries[i];
        if (e->src == PREFETCH_SRC_NVME && e->nvme_complete && !e->h2d_complete) {
            vram_slots[count] = e->vram_slot;
            ram_slots[count] = e->ram_slot;
            count++;
        }
    }
    return count;
}

void prefetch_mark_all_h2d_done(prefetch_state_t *ps) {
    for (int i = 0; i < ps->n_entries; i++) {
        if (ps->entries[i].src != PREFETCH_SRC_VRAM)
            ps->entries[i].h2d_complete = true;
    }
}

int prefetch_vram_transfer_count(const prefetch_state_t *ps) {
    return ps->n_ram + ps->n_nvme;
}

void prefetch_commit_ram_cache(prefetch_state_t *ps, ram_cache_t *rcache) {
    if (!rcache) return;
    for (int i = 0; i < ps->n_entries; i++) {
        prefetch_entry_t *e = &ps->entries[i];
        if (e->src == PREFETCH_SRC_NVME && e->nvme_complete && e->ram_slot >= 0) {
            rcache->map[e->layer * rcache->n_experts + e->expert_id] = e->ram_slot;
        }
    }
}
