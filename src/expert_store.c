#define _GNU_SOURCE
#include "expert_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

expert_store_t *expert_store_open(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        // Fall back to non-direct if O_DIRECT fails (e.g. tmpfs)
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "expert_store: cannot open %s: %s\n", path, strerror(errno));
            return NULL;
        }
        fprintf(stderr, "expert_store: O_DIRECT not supported for %s, using buffered I/O\n", path);
    }

    // Read header - need aligned buffer for O_DIRECT
    void *hdr_buf = NULL;
    if (posix_memalign(&hdr_buf, QMOE_ALIGNMENT, QMOE_ALIGNMENT) != 0) {
        close(fd);
        return NULL;
    }

    ssize_t n = pread(fd, hdr_buf, QMOE_ALIGNMENT, 0);
    if (n < (ssize_t)sizeof(qmoe_header_t)) {
        fprintf(stderr, "expert_store: failed reading header from %s\n", path);
        free(hdr_buf);
        close(fd);
        return NULL;
    }

    qmoe_header_t *hdr = (qmoe_header_t *)hdr_buf;
    if (hdr->magic != QMOE_MAGIC) {
        fprintf(stderr, "expert_store: bad magic in %s (got 0x%08x)\n", path, hdr->magic);
        free(hdr_buf);
        close(fd);
        return NULL;
    }
    if (hdr->version < QMOE_VERSION) {
        fprintf(stderr, "expert_store: old format version %u in %s (need %u), please re-run prepare_experts\n",
                hdr->version, path, QMOE_VERSION);
        free(hdr_buf);
        close(fd);
        return NULL;
    }

    expert_store_t *store = calloc(1, sizeof(expert_store_t));
    if (!store) { free(hdr_buf); close(fd); return NULL; }

    store->fd = fd;
    store->path = strdup(path);
    memcpy(&store->header, hdr, sizeof(qmoe_header_t));

    // Read layer index
    // Layer index starts right after the 64-byte header
    uint32_t n_layers = store->header.n_moe_layers;
    size_t index_size = n_layers * sizeof(qmoe_layer_entry_t);
    size_t index_read_size = ALIGN_UP(64 + index_size, QMOE_ALIGNMENT);

    void *idx_buf = NULL;
    if (posix_memalign(&idx_buf, QMOE_ALIGNMENT, index_read_size) != 0) {
        expert_store_close(store);
        free(hdr_buf);
        return NULL;
    }

    n = pread(fd, idx_buf, index_read_size, 0);
    if (n < (ssize_t)(64 + index_size)) {
        fprintf(stderr, "expert_store: failed reading layer index from %s\n", path);
        free(idx_buf);
        expert_store_close(store);
        free(hdr_buf);
        return NULL;
    }

    store->layer_index = malloc(index_size);
    if (!store->layer_index) {
        free(idx_buf);
        expert_store_close(store);
        free(hdr_buf);
        return NULL;
    }
    memcpy(store->layer_index, (uint8_t *)idx_buf + 64, index_size);

    free(idx_buf);
    free(hdr_buf);

    fprintf(stderr, "expert_store: opened %s (%u layers, %u experts, stride=%lu)\n",
            path, store->header.n_moe_layers, store->header.n_experts,
            (unsigned long)store->header.expert_stride);

    return store;
}

void expert_store_close(expert_store_t *store) {
    if (!store) return;
    if (store->fd >= 0) close(store->fd);
    free(store->path);
    free(store->layer_index);
    free(store);
}

uint64_t expert_store_offset(const expert_store_t *store, int layer, int expert_id) {
    if (layer < 0 || (uint32_t)layer >= store->header.n_moe_layers) return 0;
    if (expert_id < 0 || (uint32_t)expert_id >= store->header.n_experts) return 0;

    uint64_t stride = store->layer_index[layer].expert_stride;
    if (stride == 0) stride = store->header.expert_stride;

    return store->layer_index[layer].data_offset + (uint64_t)expert_id * stride;
}

uint64_t expert_store_expert_size(const expert_store_t *store, int layer) {
    if (layer < 0 || (uint32_t)layer >= store->header.n_moe_layers) return 0;

    uint64_t stride = store->layer_index[layer].expert_stride;
    if (stride == 0) stride = store->header.expert_stride;
    return stride;
}

// ---- Writing ----

int expert_store_create(const char *path, const qmoe_header_t *header,
                        const qmoe_layer_entry_t *layers) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        fprintf(stderr, "expert_store: cannot create %s: %s\n", path, strerror(errno));
        return -1;
    }

    // Write header (64 bytes)
    uint8_t hdr_buf[64];
    memset(hdr_buf, 0, sizeof(hdr_buf));
    memcpy(hdr_buf, header, sizeof(qmoe_header_t));
    if (write(fd, hdr_buf, 64) != 64) {
        fprintf(stderr, "expert_store: failed writing header\n");
        close(fd);
        return -1;
    }

    // Write layer index
    size_t index_size = header->n_moe_layers * sizeof(qmoe_layer_entry_t);
    if (write(fd, layers, index_size) != (ssize_t)index_size) {
        fprintf(stderr, "expert_store: failed writing layer index\n");
        close(fd);
        return -1;
    }

    return fd;
}

bool expert_store_write_expert(int fd, const qmoe_layer_entry_t *layer,
                               int expert_id, uint64_t expert_stride,
                               const void *data, uint64_t size) {
    uint64_t offset = layer->data_offset + (uint64_t)expert_id * expert_stride;

    ssize_t n = pwrite(fd, data, size, offset);
    if (n < 0 || (uint64_t)n != size) {
        fprintf(stderr, "expert_store: write failed at offset %lu: %s\n",
                (unsigned long)offset, strerror(errno));
        return false;
    }
    return true;
}
