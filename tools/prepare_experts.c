#define _GNU_SOURCE
#include "gguf.h"
#include "expert_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> <output_dir1> [output_dir2 ...]\n", prog);
    fprintf(stderr, "\nExtracts MoE expert weights from a GGUF model and writes them\n");
    fprintf(stderr, "as expert store files (one replica per output directory).\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s ~/models/Qwen3-30B-A3B-Q4_K_M.gguf /mnt/nvme_shard_a /mnt/nvme_shard_b\n", prog);
}

// Progress tracking
static void print_progress(const char *label, int current, int total) {
    int pct = (current * 100) / total;
    fprintf(stderr, "\r  %s: %d/%d (%d%%)", label, current, total, pct);
    if (current == total) fprintf(stderr, "\n");
    fflush(stderr);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_outputs = argc - 2;
    const char **output_dirs = (const char **)&argv[2];

    // Open GGUF
    fprintf(stderr, "Opening %s...\n", model_path);
    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    // Read model params
    const char *arch = gguf_get_str(ctx, "general.architecture");
    if (!arch) {
        fprintf(stderr, "Error: cannot find general.architecture\n");
        gguf_close(ctx);
        return 1;
    }

    // Get model name for output filename
    const char *general_name = gguf_get_str(ctx, "general.name");
    // Build a short safe filename from the name
    char safe_name[64] = "model";
    if (general_name) {
        // Copy, replacing spaces/slashes with dashes, lowercase
        int j = 0;
        for (int i = 0; general_name[i] && j < 60; i++) {
            char c = general_name[i];
            if (c == ' ' || c == '/' || c == '\\') c = '-';
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_')
                safe_name[j++] = c;
        }
        safe_name[j] = '\0';
        if (j == 0) strcpy(safe_name, "model");
    }

    char key_buf[256];
    snprintf(key_buf, sizeof(key_buf), "%s.block_count", arch);
    uint32_t n_layers = gguf_get_u32(ctx, key_buf);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_count", arch);
    uint32_t n_experts = gguf_get_u32(ctx, key_buf);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_used_count", arch);
    uint32_t n_experts_used = gguf_get_u32(ctx, key_buf);

    fprintf(stderr, "Architecture: %s\n", arch);
    fprintf(stderr, "Layers: %u, Experts: %u, Used: %u\n", n_layers, n_experts, n_experts_used);

    // Count MoE layers and get tensor sizes
    // Expert tensors: blk.N.ffn_gate_exps.weight, blk.N.ffn_up_exps.weight, blk.N.ffn_down_exps.weight
    int n_moe_layers = 0;
    int *moe_layer_ids = calloc(n_layers, sizeof(int));

    for (uint32_t i = 0; i < n_layers; i++) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", i);
        if (gguf_find_tensor(ctx, name) >= 0) {
            moe_layer_ids[n_moe_layers++] = i;
        }
    }

    if (n_moe_layers == 0) {
        fprintf(stderr, "Error: no MoE layers found\n");
        free(moe_layer_ids);
        gguf_close(ctx);
        return 1;
    }

    fprintf(stderr, "Found %d MoE layers\n", n_moe_layers);

    // Get sizes from the first MoE layer (gate/up are uniform, down varies)
    char name_gate[128], name_up[128], name_down[128];
    snprintf(name_gate, sizeof(name_gate), "blk.%d.ffn_gate_exps.weight", moe_layer_ids[0]);
    snprintf(name_up, sizeof(name_up), "blk.%d.ffn_up_exps.weight", moe_layer_ids[0]);

    int64_t gate_tid = gguf_find_tensor(ctx, name_gate);
    int64_t up_tid   = gguf_find_tensor(ctx, name_up);

    const gguf_tensor_info_t *gate_info = &ctx->tensors[gate_tid];
    const gguf_tensor_info_t *up_info   = &ctx->tensors[up_tid];

    uint64_t gate_per_expert = gate_info->size / n_experts;
    uint64_t up_per_expert   = up_info->size / n_experts;

    // Find the MAX down_per_expert across all layers (down type varies: Q4_K or Q6_K)
    uint64_t max_down_per_expert = 0;
    for (int i = 0; i < n_moe_layers; i++) {
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down_exps.weight", moe_layer_ids[i]);
        int64_t dtid = gguf_find_tensor(ctx, name_down);
        if (dtid >= 0) {
            uint64_t dpe = ctx->tensors[dtid].size / n_experts;
            if (dpe > max_down_per_expert) max_down_per_expert = dpe;
            fprintf(stderr, "  Layer %d down: %s, %lu bytes/expert\n",
                    moe_layer_ids[i], ggml_type_name(ctx->tensors[dtid].type),
                    (unsigned long)dpe);
        }
    }

    uint64_t expert_stride = ALIGN_UP(gate_per_expert + up_per_expert + max_down_per_expert, QMOE_ALIGNMENT);

    fprintf(stderr, "Expert sizes: gate=%lu, up=%lu, max_down=%lu (gate/up type: %s)\n",
            (unsigned long)gate_per_expert, (unsigned long)up_per_expert,
            (unsigned long)max_down_per_expert,
            ggml_type_name(gate_info->type));
    fprintf(stderr, "Expert stride (aligned): %lu bytes (%.2f MB)\n",
            (unsigned long)expert_stride, expert_stride / (1024.0 * 1024.0));

    // Build header
    qmoe_header_t header = {
        .magic           = QMOE_MAGIC,
        .version         = QMOE_VERSION,
        .n_moe_layers    = n_moe_layers,
        .n_experts       = n_experts,
        .quant_type      = gate_info->type,  // primary type
        .alignment        = QMOE_ALIGNMENT,
        .expert_gate_size = gate_per_expert,
        .expert_up_size   = up_per_expert,
        .expert_down_size = max_down_per_expert,
        .expert_stride    = expert_stride,
    };

    // Build layer index with per-layer down size and type
    uint64_t index_end = 64 + n_moe_layers * sizeof(qmoe_layer_entry_t);
    uint64_t data_start = ALIGN_UP(index_end, QMOE_ALIGNMENT);

    qmoe_layer_entry_t *layers = calloc(n_moe_layers, sizeof(qmoe_layer_entry_t));
    for (int i = 0; i < n_moe_layers; i++) {
        layers[i].data_offset   = data_start + (uint64_t)i * n_experts * expert_stride;
        layers[i].expert_stride = expert_stride;

        // Per-layer down projection info
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down_exps.weight", moe_layer_ids[i]);
        int64_t dtid = gguf_find_tensor(ctx, name_down);
        if (dtid >= 0) {
            layers[i].down_size = ctx->tensors[dtid].size / n_experts;
            layers[i].down_type = ctx->tensors[dtid].type;
        }
    }

    // Total store size varies since stride is fixed at max
    uint64_t total_store = data_start + (uint64_t)n_moe_layers * n_experts * expert_stride;
    fprintf(stderr, "Total store size: %.2f GB\n", total_store / (1024.0 * 1024.0 * 1024.0));

    // Allocate buffers for reading full expert tensors
    size_t max_tensor_size = gate_info->size;
    if (up_info->size > max_tensor_size) max_tensor_size = up_info->size;
    // Find max down tensor size across layers
    for (int i = 0; i < n_moe_layers; i++) {
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down_exps.weight", moe_layer_ids[i]);
        int64_t dtid = gguf_find_tensor(ctx, name_down);
        if (dtid >= 0 && ctx->tensors[dtid].size > max_tensor_size)
            max_tensor_size = ctx->tensors[dtid].size;
    }

    void *tensor_buf = malloc(max_tensor_size);
    void *expert_buf = NULL;
    if (posix_memalign(&expert_buf, QMOE_ALIGNMENT, expert_stride) != 0) {
        fprintf(stderr, "Error: failed to allocate expert buffer\n");
        free(tensor_buf);
        free(layers);
        free(moe_layer_ids);
        gguf_close(ctx);
        return 1;
    }

    // Create output files
    int *out_fds = calloc(n_outputs, sizeof(int));
    for (int i = 0; i < n_outputs; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/experts-%s.qmoe", output_dirs[i], safe_name);

        // Ensure directory exists
        struct stat st;
        if (stat(output_dirs[i], &st) != 0) {
            fprintf(stderr, "Error: output directory %s does not exist\n", output_dirs[i]);
            goto cleanup;
        }

        fprintf(stderr, "Creating %s...\n", path);
        out_fds[i] = expert_store_create(path, &header, layers);
        if (out_fds[i] < 0) goto cleanup;
    }

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Process each MoE layer
    for (int li = 0; li < n_moe_layers; li++) {
        int layer_id = moe_layer_ids[li];
        fprintf(stderr, "\nProcessing layer %d/%d (blk.%d)...\n", li + 1, n_moe_layers, layer_id);

        // Read the three expert tensors for this layer
        char tname[128];
        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", layer_id);
        int64_t tid_gate = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", layer_id);
        int64_t tid_up = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", layer_id);
        int64_t tid_down = gguf_find_tensor(ctx, tname);

        if (tid_gate < 0 || tid_up < 0 || tid_down < 0) {
            fprintf(stderr, "Error: missing expert tensor in layer %d\n", layer_id);
            goto cleanup;
        }

        // Get per-layer down size
        uint64_t layer_down_per_expert = ctx->tensors[tid_down].size / n_experts;

        // Read expert tensors
        void *gate_data = malloc(ctx->tensors[tid_gate].size);
        void *up_data   = malloc(ctx->tensors[tid_up].size);
        void *down_data = malloc(ctx->tensors[tid_down].size);

        if (!gate_data || !up_data || !down_data) {
            fprintf(stderr, "Error: out of memory reading layer %d\n", layer_id);
            free(gate_data); free(up_data); free(down_data);
            goto cleanup;
        }

        fprintf(stderr, "  Reading tensors (down: %s, %lu bytes/expert)...\n",
                ggml_type_name(ctx->tensors[tid_down].type),
                (unsigned long)layer_down_per_expert);
        gguf_read_tensor(ctx, tid_gate, gate_data, ctx->tensors[tid_gate].size);
        gguf_read_tensor(ctx, tid_up,   up_data,   ctx->tensors[tid_up].size);
        gguf_read_tensor(ctx, tid_down, down_data,  ctx->tensors[tid_down].size);

        // For each expert, extract its slice and write to all output files
        for (uint32_t ei = 0; ei < n_experts; ei++) {
            // Zero the expert buffer (handles alignment padding for smaller down projections)
            memset(expert_buf, 0, expert_stride);

            // Copy gate_proj for expert ei
            memcpy(expert_buf,
                   (uint8_t *)gate_data + ei * gate_per_expert,
                   gate_per_expert);

            // Copy up_proj for expert ei
            memcpy((uint8_t *)expert_buf + gate_per_expert,
                   (uint8_t *)up_data + ei * up_per_expert,
                   up_per_expert);

            // Copy down_proj for expert ei (size varies per layer!)
            memcpy((uint8_t *)expert_buf + gate_per_expert + up_per_expert,
                   (uint8_t *)down_data + ei * layer_down_per_expert,
                   layer_down_per_expert);

            // Write to all output files
            for (int oi = 0; oi < n_outputs; oi++) {
                if (!expert_store_write_expert(out_fds[oi], &layers[li],
                                               ei, expert_stride,
                                               expert_buf, expert_stride)) {
                    fprintf(stderr, "Error: write failed for layer %d expert %u output %d\n",
                            layer_id, ei, oi);
                    free(gate_data); free(up_data); free(down_data);
                    goto cleanup;
                }
            }

            if ((ei + 1) % 16 == 0 || ei == n_experts - 1) {
                print_progress("Experts", ei + 1, n_experts);
            }
        }

        free(gate_data);
        free(up_data);
        free(down_data);

        // Progress timing
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        double layers_per_sec = (li + 1) / elapsed;
        double eta = (n_moe_layers - li - 1) / layers_per_sec;
        fprintf(stderr, "  Layer done (%.1f s elapsed, ETA: %.0f s)\n", elapsed, eta);
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    double total_time = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
    fprintf(stderr, "\nDone! %d layers, %u experts each, %d replicas in %.1f seconds\n",
            n_moe_layers, n_experts, n_outputs, total_time);

    // Cleanup
    for (int i = 0; i < n_outputs; i++) {
        if (out_fds[i] >= 0) {
            fsync(out_fds[i]);
            close(out_fds[i]);
        }
    }

    free(expert_buf);
    free(tensor_buf);
    free(layers);
    free(out_fds);
    free(moe_layer_ids);
    gguf_close(ctx);

    return 0;

cleanup:
    for (int i = 0; i < n_outputs; i++) {
        if (out_fds[i] >= 0) close(out_fds[i]);
    }
    free(expert_buf);
    free(tensor_buf);
    free(layers);
    free(out_fds);
    free(moe_layer_ids);
    gguf_close(ctx);
    return 1;
}
