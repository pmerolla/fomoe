#define _GNU_SOURCE
#include "gguf.h"
#include "expert_store.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <store.qmoe> [layer] [expert]\n", argv[0]);
        return 1;
    }

    int layer = argc > 3 ? atoi(argv[3]) : 0;
    int expert = argc > 4 ? atoi(argv[4]) : 0;

    // Open GGUF
    gguf_ctx_t *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    // Open expert store
    expert_store_t *store = expert_store_open(argv[2]);
    if (!store) { gguf_close(ctx); return 1; }

    printf("Store: gate_size=%lu, up_size=%lu, stride=%lu\n",
           (unsigned long)store->header.expert_gate_size,
           (unsigned long)store->header.expert_up_size,
           (unsigned long)store->header.expert_stride);

    // Read gate_exps tensor from GGUF
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) { fprintf(stderr, "Tensor not found: %s\n", name); return 1; }

    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    printf("GGUF tensor: %s, type=%s, dims=[%ld,%ld,%ld], size=%zu\n",
           name, ggml_type_name(ti->type),
           ti->dims[0], ti->dims[1], ti->dims[2], ti->size);

    uint64_t per_expert = ti->size / ti->dims[2];  // size / n_experts
    printf("Per-expert gate size from GGUF: %lu (store says: %lu)\n",
           (unsigned long)per_expert, (unsigned long)store->header.expert_gate_size);

    // Read just the expert's gate data from GGUF
    void *gguf_expert = malloc(per_expert);
    void *full_tensor = malloc(ti->size);
    if (!gguf_expert || !full_tensor) { fprintf(stderr, "OOM\n"); return 1; }

    gguf_read_tensor(ctx, tid, full_tensor, ti->size);
    memcpy(gguf_expert, (uint8_t *)full_tensor + (uint64_t)expert * per_expert, per_expert);
    free(full_tensor);

    // Read same expert from .qmoe store
    uint64_t qmoe_offset = expert_store_offset(store, layer, expert);
    printf("QMOE offset for layer %d expert %d: %lu\n", layer, expert, (unsigned long)qmoe_offset);

    // Read with regular I/O (not O_DIRECT)
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("open qmoe"); return 1; }

    void *qmoe_expert = malloc(store->header.expert_stride);
    if (!qmoe_expert) { fprintf(stderr, "OOM\n"); return 1; }

    ssize_t n = pread(fd, qmoe_expert, store->header.expert_stride, qmoe_offset);
    if (n != (ssize_t)store->header.expert_stride) {
        fprintf(stderr, "Short read: got %zd, expected %lu\n", n, (unsigned long)store->header.expert_stride);
    }
    close(fd);

    // Compare gate section
    printf("\n=== Comparing gate weight (first %lu bytes) ===\n", (unsigned long)per_expert);
    int mismatches = 0;
    int first_mismatch = -1;
    for (uint64_t i = 0; i < per_expert; i++) {
        if (((uint8_t *)gguf_expert)[i] != ((uint8_t *)qmoe_expert)[i]) {
            if (first_mismatch < 0) first_mismatch = (int)i;
            mismatches++;
        }
    }
    printf("Gate: %d mismatches out of %lu bytes", mismatches, (unsigned long)per_expert);
    if (first_mismatch >= 0) printf(" (first at byte %d)", first_mismatch);
    printf("\n");

    // Print first 64 bytes from each
    printf("GGUF gate first 64 bytes: ");
    for (int i = 0; i < 64 && i < (int)per_expert; i++) printf("%02x", ((uint8_t *)gguf_expert)[i]);
    printf("\nQMOE gate first 64 bytes: ");
    for (int i = 0; i < 64 && i < (int)per_expert; i++) printf("%02x", ((uint8_t *)qmoe_expert)[i]);
    printf("\n");

    // Also compare up section
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
    tid = gguf_find_tensor(ctx, name);
    if (tid >= 0) {
        const gguf_tensor_info_t *ui = &ctx->tensors[tid];
        uint64_t up_per_expert = ui->size / ui->dims[2];

        void *up_full = malloc(ui->size);
        gguf_read_tensor(ctx, tid, up_full, ui->size);
        void *gguf_up = (uint8_t *)up_full + (uint64_t)expert * up_per_expert;
        void *qmoe_up = (uint8_t *)qmoe_expert + store->header.expert_gate_size;

        printf("\n=== Comparing up weight (first %lu bytes) ===\n", (unsigned long)up_per_expert);
        mismatches = 0;
        first_mismatch = -1;
        for (uint64_t i = 0; i < up_per_expert; i++) {
            if (((uint8_t *)gguf_up)[i] != ((uint8_t *)qmoe_up)[i]) {
                if (first_mismatch < 0) first_mismatch = (int)i;
                mismatches++;
            }
        }
        printf("Up: %d mismatches out of %lu bytes", mismatches, (unsigned long)up_per_expert);
        if (first_mismatch >= 0) printf(" (first at byte %d)", first_mismatch);
        printf("\n");

        free(up_full);
    }

    // Also compare down section
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
    tid = gguf_find_tensor(ctx, name);
    if (tid >= 0) {
        const gguf_tensor_info_t *di = &ctx->tensors[tid];
        uint64_t down_per_expert = di->size / di->dims[2];

        void *down_full = malloc(di->size);
        gguf_read_tensor(ctx, tid, down_full, di->size);
        void *gguf_down = (uint8_t *)down_full + (uint64_t)expert * down_per_expert;
        void *qmoe_down = (uint8_t *)qmoe_expert + store->header.expert_gate_size + store->header.expert_up_size;

        printf("\n=== Comparing down weight (first %lu bytes) ===\n", (unsigned long)down_per_expert);
        mismatches = 0;
        first_mismatch = -1;
        for (uint64_t i = 0; i < down_per_expert; i++) {
            if (((uint8_t *)gguf_down)[i] != ((uint8_t *)qmoe_down)[i]) {
                if (first_mismatch < 0) first_mismatch = (int)i;
                mismatches++;
            }
        }
        printf("Down: %d mismatches out of %lu bytes", mismatches, (unsigned long)down_per_expert);
        if (first_mismatch >= 0) printf(" (first at byte %d)", first_mismatch);
        printf("\n");

        free(down_full);
    }

    // Dequant first row of gate from both sources and print
    printf("\n=== Dequantized first row of gate (first 16 values) ===\n");
    float gguf_f32[256], qmoe_f32[256];
    dequantize_row_q4_K((const block_q4_K *)gguf_expert, gguf_f32, 256);
    dequantize_row_q4_K((const block_q4_K *)qmoe_expert, qmoe_f32, 256);
    printf("GGUF: ");
    for (int i = 0; i < 16; i++) printf("%.4f ", gguf_f32[i]);
    printf("\nQMOE: ");
    for (int i = 0; i < 16; i++) printf("%.4f ", qmoe_f32[i]);
    printf("\n");

    free(gguf_expert);
    free(qmoe_expert);
    expert_store_close(store);
    gguf_close(ctx);
    return 0;
}
