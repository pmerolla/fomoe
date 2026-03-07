#ifndef QMOE_NVME_IO_H
#define QMOE_NVME_IO_H

#include "expert_store.h"
#include <stdint.h>
#include <stdbool.h>
#include <liburing.h>

// Maximum number of NVMe drives
#define MAX_DRIVES 8

// Maximum number of concurrent expert reads
#define MAX_CONCURRENT_READS 32

// Single drive context with io_uring ring
typedef struct {
    int            fd;       // O_DIRECT fd from expert_store
    const char    *path;
    expert_store_t *store;
    struct io_uring ring;    // one io_uring ring per drive
    struct io_uring ring2;   // second ring for concurrent prefetch I/O
    bool           ring2_ok; // ring2 initialized successfully
} drive_ctx_t;

// Read request (used as userdata for CQE tracking)
typedef struct {
    int       drive_idx;     // which drive to read from
    int       buffer_idx;    // index in out_buffers array
    int       layer;         // MoE layer index
    int       expert_id;     // expert number
    void     *buffer;        // destination (must be page-aligned for O_DIRECT)
    uint64_t  offset;        // file offset
    uint64_t  size;          // bytes to read
    bool      done;
} read_request_t;

// NVMe I/O manager
typedef struct {
    int          n_drives;
    drive_ctx_t  drives[MAX_DRIVES];

    // Pre-allocated aligned buffers for expert reads
    int          n_buffers;
    void       **buffers;        // array of page-aligned buffers
    uint64_t     buffer_size;    // size of each buffer (expert_stride)
} nvme_io_t;

// Initialize NVMe I/O with a list of expert store paths
nvme_io_t *nvme_io_init(const char **store_paths, int n_stores);

// Cleanup
void nvme_io_free(nvme_io_t *io);

// Load multiple experts in parallel across drives
// expert_ids: array of expert indices to load
// layer: which MoE layer
// n_experts: how many experts to load (typically 8)
// out_buffers: receives pointers to the loaded expert data
// Returns 0 on success
int nvme_io_load_experts(nvme_io_t *io, int layer, const int *expert_ids,
                         int n_experts, void **out_buffers);

// Load experts using buffers starting at buffer_offset in the pool
// Allows two batches to use non-overlapping host buffers for pipelining
int nvme_io_load_experts_at(nvme_io_t *io, int layer, const int *expert_ids,
                             int n_experts, int buffer_offset, void **out_buffers);

// Callback fired per expert when its NVMe read completes successfully
// buffer_idx: index in the out_buffers array, buffer: pointer to read data
typedef void (*nvme_expert_cb)(int buffer_idx, void *buffer, void *user_data);

// Load experts with per-expert completion callback
// cb fires from drive I/O threads — must be thread-safe
int nvme_io_load_experts_at_cb(nvme_io_t *io, int layer, const int *expert_ids,
                                int n_experts, int buffer_offset, void **out_buffers,
                                nvme_expert_cb cb, void *cb_data);

// Load experts using separate ring2 io_uring rings (safe to run concurrently with ring1 loads)
// Used for speculative prefetch while main pipeline NVMe loads are in-flight
int nvme_io_load_experts_prefetch(nvme_io_t *io, int layer, const int *expert_ids,
                                   int n_experts, int buffer_offset, void **out_buffers);

// Blocking pread of a single expert into caller-supplied buffer (page-aligned).
// No io_uring — safe to call from any thread without contention.
int nvme_io_pread_expert(nvme_io_t *io, int layer, int expert_id, void *buf);

// Benchmark: read a set of experts and report timing
void nvme_io_benchmark(nvme_io_t *io, int n_experts_to_load, int iterations);

#endif // QMOE_NVME_IO_H
