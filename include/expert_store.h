#ifndef QMOE_EXPERT_STORE_H
#define QMOE_EXPERT_STORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Expert store binary format constants
#define QMOE_MAGIC      0x454F4D51  // "QMOE" little-endian
#define QMOE_VERSION    2
#define QMOE_ALIGNMENT  4096        // alignment for O_DIRECT

// Round up to alignment
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

// Header (64 bytes, padded)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_moe_layers;
    uint32_t n_experts;
    uint32_t quant_type;       // ggml_dtype of expert weights
    uint32_t alignment;        // 4096 for O_DIRECT
    uint64_t expert_gate_size; // bytes for gate_proj per expert (before alignment)
    uint64_t expert_up_size;   // bytes for up_proj per expert
    uint64_t expert_down_size; // bytes for down_proj per expert
    uint64_t expert_stride;    // aligned total bytes per expert (gate+up+down, aligned)
    // (pad to 64 bytes if needed - total used = 56 bytes, 8 bytes reserved)
} qmoe_header_t;

_Static_assert(sizeof(qmoe_header_t) <= 64, "header too large");

// Per-layer entry in the layer index
typedef struct {
    uint64_t data_offset;   // absolute file offset to first expert of this layer
    uint64_t expert_stride; // bytes per expert (aligned), can override header default
    uint64_t down_size;     // actual bytes for down_proj per expert (may differ from header)
    uint32_t down_type;     // ggml_dtype for down projection in this layer
    uint32_t pad;
} qmoe_layer_entry_t;  // 32 bytes

// Runtime handle for reading from an expert store file
typedef struct {
    int       fd;              // file descriptor (opened with O_RDONLY | O_DIRECT)
    char     *path;            // file path (owned)

    qmoe_header_t       header;
    qmoe_layer_entry_t *layer_index;
} expert_store_t;

// Open an expert store file for reading
expert_store_t *expert_store_open(const char *path);

// Close and free
void expert_store_close(expert_store_t *store);

// Get the absolute file offset for a specific expert in a specific layer
uint64_t expert_store_offset(const expert_store_t *store, int layer, int expert_id);

// Get the read size for one expert
uint64_t expert_store_expert_size(const expert_store_t *store, int layer);

// ---- Writing (for prepare_experts tool) ----

// Create a new expert store file (truncates if exists)
// Returns fd or -1 on error
int expert_store_create(const char *path, const qmoe_header_t *header,
                        const qmoe_layer_entry_t *layers);

// Write expert data at the correct offset
// data must be aligned to QMOE_ALIGNMENT, size must be aligned
bool expert_store_write_expert(int fd, const qmoe_layer_entry_t *layer,
                               int expert_id, uint64_t expert_stride,
                               const void *data, uint64_t size);

#endif // QMOE_EXPERT_STORE_H
