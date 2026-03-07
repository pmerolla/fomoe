#define _GNU_SOURCE
#include "nvme_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <liburing.h>

// Ring size per drive
#define RING_ENTRIES 32

// Per-drive work package
typedef struct {
    drive_ctx_t    *drive;      // which drive
    struct io_uring *ring;      // which io_uring ring to use (ring1 or ring2)
    read_request_t *requests;   // array of requests for this drive
    int             n_requests;
    uint64_t        buffer_size;
    int             errors;     // output: number of errors
    nvme_expert_cb  cb;         // per-expert completion callback (may be NULL)
    void           *cb_data;
} drive_work_t;

// Thread function: execute io_uring submit+reap for one drive
static void *drive_io_thread(void *arg) {
    drive_work_t *work = (drive_work_t *)arg;
    struct io_uring *ring = work->ring;
    work->errors = 0;

    // Prepare SQEs
    for (int i = 0; i < work->n_requests; i++) {
        read_request_t *req = &work->requests[i];
        struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
        if (!sqe) {
            fprintf(stderr, "nvme_io: failed to get SQE for drive (expert %d)\n",
                    req->expert_id);
            work->errors++;
            return NULL;
        }
        io_uring_prep_read(sqe, work->drive->fd, req->buffer, work->buffer_size, req->offset);
        io_uring_sqe_set_data(sqe, req);
    }

    // Submit all at once
    int ret = io_uring_submit(ring);
    if (ret < 0) {
        fprintf(stderr, "nvme_io: io_uring_submit failed: %s\n", strerror(-ret));
        work->errors = work->n_requests;
        return NULL;
    }

    // Reap all CQEs
    for (int i = 0; i < work->n_requests; i++) {
        struct io_uring_cqe *cqe;
        ret = io_uring_wait_cqe(ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "nvme_io: io_uring_wait_cqe failed: %s\n", strerror(-ret));
            work->errors++;
            continue;
        }

        read_request_t *req = io_uring_cqe_get_data(cqe);
        if (cqe->res < 0) {
            fprintf(stderr, "nvme_io: read failed expert %d: %s\n",
                    req->expert_id, strerror(-cqe->res));
            work->errors++;
        } else if ((uint64_t)cqe->res != req->size) {
            fprintf(stderr, "nvme_io: short read expert %d: got %d of %lu\n",
                    req->expert_id, cqe->res, (unsigned long)req->size);
            work->errors++;
        } else {
            req->done = true;
            if (work->cb) {
                work->cb(req->buffer_idx, req->buffer, work->cb_data);
            }
        }

        io_uring_cqe_seen(ring, cqe);
    }

    return NULL;
}

nvme_io_t *nvme_io_init(const char **store_paths, int n_stores) {
    if (n_stores <= 0 || n_stores > MAX_DRIVES) {
        fprintf(stderr, "nvme_io: invalid number of stores: %d\n", n_stores);
        return NULL;
    }

    nvme_io_t *io = calloc(1, sizeof(nvme_io_t));
    if (!io) return NULL;

    io->n_drives = n_stores;

    // Open each expert store and init io_uring ring per drive
    for (int i = 0; i < n_stores; i++) {
        io->drives[i].store = expert_store_open(store_paths[i]);
        if (!io->drives[i].store) {
            fprintf(stderr, "nvme_io: failed to open store %s\n", store_paths[i]);
            nvme_io_free(io);
            return NULL;
        }
        io->drives[i].fd = io->drives[i].store->fd;
        io->drives[i].path = io->drives[i].store->path;

        int ret = io_uring_queue_init(RING_ENTRIES, &io->drives[i].ring, 0);
        if (ret < 0) {
            fprintf(stderr, "nvme_io: io_uring_queue_init failed for drive %d: %s\n",
                    i, strerror(-ret));
            nvme_io_free(io);
            return NULL;
        }

        // Second ring for concurrent prefetch I/O
        ret = io_uring_queue_init(RING_ENTRIES, &io->drives[i].ring2, 0);
        io->drives[i].ring2_ok = (ret == 0);
    }

    // Get buffer size from first store
    io->buffer_size = io->drives[0].store->header.expert_stride;

    // Allocate aligned buffers for expert reads
    io->n_buffers = MAX_CONCURRENT_READS;
    io->buffers = calloc(io->n_buffers, sizeof(void *));
    if (!io->buffers) { nvme_io_free(io); return NULL; }

    for (int i = 0; i < io->n_buffers; i++) {
        if (posix_memalign(&io->buffers[i], QMOE_ALIGNMENT, io->buffer_size) != 0) {
            fprintf(stderr, "nvme_io: failed to allocate buffer %d\n", i);
            nvme_io_free(io);
            return NULL;
        }
    }

    fprintf(stderr, "nvme_io: initialized %d drives with io_uring, %d buffers of %lu bytes\n",
            io->n_drives, io->n_buffers, (unsigned long)io->buffer_size);

    return io;
}

void nvme_io_free(nvme_io_t *io) {
    if (!io) return;

    for (int i = 0; i < io->n_drives; i++) {
        if (io->drives[i].store) {
            io_uring_queue_exit(&io->drives[i].ring);
            if (io->drives[i].ring2_ok)
                io_uring_queue_exit(&io->drives[i].ring2);
            expert_store_close(io->drives[i].store);
        }
    }

    if (io->buffers) {
        for (int i = 0; i < io->n_buffers; i++) {
            free(io->buffers[i]);
        }
        free(io->buffers);
    }

    free(io);
}

int nvme_io_load_experts_at_cb(nvme_io_t *io, int layer, const int *expert_ids,
                                int n_experts, int buffer_offset, void **out_buffers,
                                nvme_expert_cb cb, void *cb_data) {
    if (buffer_offset + n_experts > io->n_buffers) {
        fprintf(stderr, "nvme_io: too many experts (%d+%d > %d buffers)\n",
                buffer_offset, n_experts, io->n_buffers);
        return -1;
    }

    // Per-drive request arrays (stack allocated)
    read_request_t drive_requests[MAX_DRIVES][MAX_CONCURRENT_READS];
    int drive_n_reqs[MAX_DRIVES] = {0};

    // Distribute experts round-robin across drives and build request arrays
    for (int i = 0; i < n_experts; i++) {
        int d = i % io->n_drives;
        int idx = drive_n_reqs[d];

        drive_requests[d][idx].drive_idx  = d;
        drive_requests[d][idx].buffer_idx = i;
        drive_requests[d][idx].layer      = layer;
        drive_requests[d][idx].expert_id  = expert_ids[i];
        drive_requests[d][idx].buffer     = io->buffers[buffer_offset + i];
        drive_requests[d][idx].offset     = expert_store_offset(io->drives[d].store, layer, expert_ids[i]);
        drive_requests[d][idx].size       = io->buffer_size;
        drive_requests[d][idx].done       = false;

        out_buffers[i] = io->buffers[buffer_offset + i];
        drive_n_reqs[d]++;
    }

    // Launch one thread per drive for parallel I/O
    drive_work_t work[MAX_DRIVES];
    pthread_t threads[MAX_DRIVES];
    bool thread_active[MAX_DRIVES] = {false};

    for (int d = 0; d < io->n_drives; d++) {
        if (drive_n_reqs[d] == 0) continue;

        work[d].drive       = &io->drives[d];
        work[d].ring        = &io->drives[d].ring;
        work[d].requests    = drive_requests[d];
        work[d].n_requests  = drive_n_reqs[d];
        work[d].buffer_size = io->buffer_size;
        work[d].errors      = 0;
        work[d].cb          = cb;
        work[d].cb_data     = cb_data;

        if (pthread_create(&threads[d], NULL, drive_io_thread, &work[d]) == 0) {
            thread_active[d] = true;
        } else {
            // Fallback: run inline
            drive_io_thread(&work[d]);
        }
    }

    // Wait for all drives to complete
    int errors = 0;
    for (int d = 0; d < io->n_drives; d++) {
        if (thread_active[d]) {
            pthread_join(threads[d], NULL);
        }
        if (drive_n_reqs[d] > 0) {
            errors += work[d].errors;
        }
    }

    return errors ? -1 : 0;
}

int nvme_io_load_experts_at(nvme_io_t *io, int layer, const int *expert_ids,
                             int n_experts, int buffer_offset, void **out_buffers) {
    return nvme_io_load_experts_at_cb(io, layer, expert_ids, n_experts,
                                      buffer_offset, out_buffers, NULL, NULL);
}

int nvme_io_load_experts_prefetch(nvme_io_t *io, int layer, const int *expert_ids,
                                  int n_experts, int buffer_offset, void **out_buffers) {
    if (buffer_offset + n_experts > io->n_buffers) return -1;

    // Same logic as nvme_io_load_experts_at_cb but uses ring2 (concurrent-safe)
    read_request_t drive_requests[MAX_DRIVES][MAX_CONCURRENT_READS];
    int drive_n_reqs[MAX_DRIVES] = {0};

    for (int i = 0; i < n_experts; i++) {
        int d = i % io->n_drives;
        int idx = drive_n_reqs[d];
        drive_requests[d][idx].drive_idx  = d;
        drive_requests[d][idx].buffer_idx = i;
        drive_requests[d][idx].layer      = layer;
        drive_requests[d][idx].expert_id  = expert_ids[i];
        drive_requests[d][idx].buffer     = io->buffers[buffer_offset + i];
        drive_requests[d][idx].offset     = expert_store_offset(io->drives[d].store, layer, expert_ids[i]);
        drive_requests[d][idx].size       = io->buffer_size;
        drive_requests[d][idx].done       = false;
        out_buffers[i] = io->buffers[buffer_offset + i];
        drive_n_reqs[d]++;
    }

    drive_work_t work[MAX_DRIVES];
    pthread_t threads[MAX_DRIVES];
    bool thread_active[MAX_DRIVES] = {false};

    for (int d = 0; d < io->n_drives; d++) {
        if (drive_n_reqs[d] == 0) continue;
        if (!io->drives[d].ring2_ok) return -1;  // ring2 not available

        work[d].drive       = &io->drives[d];
        work[d].ring        = &io->drives[d].ring2;  // use ring2!
        work[d].requests    = drive_requests[d];
        work[d].n_requests  = drive_n_reqs[d];
        work[d].buffer_size = io->buffer_size;
        work[d].errors      = 0;
        work[d].cb          = NULL;
        work[d].cb_data     = NULL;

        if (pthread_create(&threads[d], NULL, drive_io_thread, &work[d]) == 0) {
            thread_active[d] = true;
        } else {
            drive_io_thread(&work[d]);
        }
    }

    int errors = 0;
    for (int d = 0; d < io->n_drives; d++) {
        if (thread_active[d]) pthread_join(threads[d], NULL);
        if (drive_n_reqs[d] > 0) errors += work[d].errors;
    }
    return errors ? -1 : 0;
}

int nvme_io_load_experts(nvme_io_t *io, int layer, const int *expert_ids,
                         int n_experts, void **out_buffers) {
    return nvme_io_load_experts_at(io, layer, expert_ids, n_experts, 0, out_buffers);
}

// Blocking pread of a single expert into caller-supplied buffer (must be page-aligned).
// No io_uring — safe to call from any thread without contention.
int nvme_io_pread_expert(nvme_io_t *io, int layer, int expert_id, void *buf) {
    if (!io || io->n_drives == 0) return -1;
    int d = layer % io->n_drives;
    drive_ctx_t *drv = &io->drives[d];
    uint64_t off = expert_store_offset(drv->store, layer, expert_id);
    uint64_t remaining = io->buffer_size;
    uint64_t pos = 0;
    while (remaining > 0) {
        ssize_t rd = pread(drv->fd, (char *)buf + pos, remaining, off + pos);
        if (rd <= 0) return -1;
        pos += rd;
        remaining -= rd;
    }
    return 0;
}

void nvme_io_benchmark(nvme_io_t *io, int n_experts_to_load, int iterations) {
    if (n_experts_to_load > io->n_buffers) n_experts_to_load = io->n_buffers;

    int expert_ids[MAX_CONCURRENT_READS];
    void *out_buffers[MAX_CONCURRENT_READS];

    int n_experts_total = io->drives[0].store->header.n_experts;
    int n_moe_layers = io->drives[0].store->header.n_moe_layers;

    // Warmup
    for (int i = 0; i < n_experts_to_load; i++) expert_ids[i] = i;
    nvme_io_load_experts(io, 0, expert_ids, n_experts_to_load, out_buffers);

    struct timespec start, end;
    double total_time = 0;
    double min_time = 1e9, max_time = 0;
    uint64_t drive_bytes[MAX_DRIVES] = {0};

    for (int iter = 0; iter < iterations; iter++) {
        int layer = iter % n_moe_layers;
        for (int i = 0; i < n_experts_to_load; i++) {
            expert_ids[i] = ((iter * 7 + i * 13) ^ (iter >> 2)) % n_experts_total;
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        int rc = nvme_io_load_experts(io, layer, expert_ids, n_experts_to_load, out_buffers);
        clock_gettime(CLOCK_MONOTONIC, &end);

        if (rc != 0) {
            fprintf(stderr, "nvme_io: benchmark read failed on iteration %d\n", iter);
            return;
        }

        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_nsec - start.tv_nsec) / 1e6;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;

        for (int i = 0; i < n_experts_to_load; i++) {
            drive_bytes[i % io->n_drives] += io->buffer_size;
        }
    }

    double avg_time = total_time / iterations;
    double total_bytes = (double)n_experts_to_load * io->buffer_size * iterations;
    double agg_throughput_gbs = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);

    fprintf(stderr, "\n=== NVMe I/O Benchmark (io_uring) ===\n");
    fprintf(stderr, "  Drives:       %d\n", io->n_drives);
    fprintf(stderr, "  Experts/load: %d (%.2f MB each)\n",
            n_experts_to_load, io->buffer_size / (1024.0 * 1024.0));
    fprintf(stderr, "  Layers:       random across %d MoE layers\n", n_moe_layers);
    fprintf(stderr, "  Expert IDs:   random (simulating real workload)\n");
    fprintf(stderr, "  Iterations:   %d\n", iterations);
    fprintf(stderr, "\n  Per-drive results:\n");
    for (int d = 0; d < io->n_drives; d++) {
        double drive_gbs = (drive_bytes[d] / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
        int experts_per_drive = 0;
        for (int i = 0; i < n_experts_to_load; i++) {
            if (i % io->n_drives == d) experts_per_drive++;
        }
        fprintf(stderr, "    Drive %d (%s): %.1f GB/s (QD avg: %.1f)\n",
                d, io->drives[d].path, drive_gbs, (double)experts_per_drive);
    }
    fprintf(stderr, "\n  Aggregate:    %.1f GB/s\n", agg_throughput_gbs);
    fprintf(stderr, "  Avg latency:  %.2f ms\n", avg_time);
    fprintf(stderr, "  Min latency:  %.2f ms\n", min_time);
    fprintf(stderr, "  Max latency:  %.2f ms\n", max_time);
    fprintf(stderr, "  Total data:   %.2f MB per load\n",
            (double)n_experts_to_load * io->buffer_size / (1024.0 * 1024.0));
    fprintf(stderr, "================================================\n\n");
}
