#include "gguf.h"
#include "quant.h"
#include "tensor.h"
#include "nvme_io.h"
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "sampler.h"
#include "freq_profile.h"
#ifdef QMOE_GPU
#include "gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <command> [args...]\n", prog);
    fprintf(stderr, "\nCommands:\n");
    fprintf(stderr, "  info      <model.gguf>              - Show model metadata and tensor list\n");
    fprintf(stderr, "  dequant   <model.gguf> <tensor>     - Dequantize and print first values\n");
    fprintf(stderr, "  benchmark <store1> [store2 ...]     - Benchmark NVMe expert loading\n");
    fprintf(stderr, "  generate  [options] <model.gguf> <store1> [store2 ...] -- \"prompt\"\n");
    fprintf(stderr, "                                      - Generate text from a prompt\n");
    fprintf(stderr, "  chat      [options] <model.gguf> <store1> [store2 ...]\n");
    fprintf(stderr, "                                      - Interactive multi-turn chat\n");
    fprintf(stderr, "  profile   [options] <model.gguf> <store1> [store2 ...] -- \"prompt\" <output.freq>\n");
    fprintf(stderr, "                                      - Generate frequency profile from expert usage\n");
    fprintf(stderr, "  ppl       [options] <model.gguf> <store1> [store2 ...] -- \"text\" or @file\n");
    fprintf(stderr, "                                      - Measure perplexity (llama.cpp-compatible)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --ram-cache MB         Pinned RAM cache size (0=off, -1=auto, default=auto)\n");
    fprintf(stderr, "  --vram-cache MB        VRAM expert cache size (0=off, -1=auto, default=auto)\n");
    fprintf(stderr, "  --max-tokens N         Maximum tokens to generate\n");
    fprintf(stderr, "  --no-eos               Don't stop on EOS/im_end tokens\n");
    fprintf(stderr, "  --freq-profile PATH    Seed VRAM cache from frequency profile at startup\n");
    fprintf(stderr, "  --car-threshold FLOAT  Cache-aware routing threshold (0.0-1.0, default=0.7)\n");
    fprintf(stderr, "  --spec-k INT           Over-prediction K for speculative routing (default=auto)\n");
    fprintf(stderr, "  --prefetch-budget INT  Max NVMe reads per prefetch window (default=auto)\n");
    fprintf(stderr, "  --ppl-ctx INT          Context size for perplexity chunks (default=512)\n");
    fprintf(stderr, "  --ppl-chunks INT       Max chunks to evaluate (default=all)\n");
    fprintf(stderr, "  --ppl-resume FILE      Checkpoint file for resumable ppl (appends per-chunk NLL)\n");
}

// ---- info command: dump GGUF metadata and tensors ----

static int cmd_info(const char *model_path) {
    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    printf("=== GGUF Info: %s ===\n", model_path);
    printf("Version:     %u\n", ctx->version);
    printf("Alignment:   %lu\n", (unsigned long)ctx->alignment);
    printf("Data offset: %lu\n", (unsigned long)ctx->data_offset);
    printf("KV pairs:    %ld\n", (long)ctx->n_kv);
    printf("Tensors:     %ld\n", (long)ctx->n_tensors);

    printf("\n--- Key-Value Pairs ---\n");
    for (int64_t i = 0; i < ctx->n_kv; i++) {
        const gguf_kv_t *kv = &ctx->kv[i];
        printf("  %-50s ", kv->key);
        switch (kv->type) {
            case GGUF_TYPE_UINT8:   printf("[u8]    %u\n", kv->value.u8); break;
            case GGUF_TYPE_INT8:    printf("[i8]    %d\n", kv->value.i8); break;
            case GGUF_TYPE_UINT16:  printf("[u16]   %u\n", kv->value.u16); break;
            case GGUF_TYPE_INT16:   printf("[i16]   %d\n", kv->value.i16); break;
            case GGUF_TYPE_UINT32:  printf("[u32]   %u\n", kv->value.u32); break;
            case GGUF_TYPE_INT32:   printf("[i32]   %d\n", kv->value.i32); break;
            case GGUF_TYPE_FLOAT32: printf("[f32]   %g\n", kv->value.f32); break;
            case GGUF_TYPE_BOOL:    printf("[bool]  %s\n", kv->value.b ? "true" : "false"); break;
            case GGUF_TYPE_STRING:  printf("[str]   \"%s\"\n", kv->value.str); break;
            case GGUF_TYPE_UINT64:  printf("[u64]   %lu\n", (unsigned long)kv->value.u64); break;
            case GGUF_TYPE_INT64:   printf("[i64]   %ld\n", (long)kv->value.i64); break;
            case GGUF_TYPE_FLOAT64: printf("[f64]   %g\n", kv->value.f64); break;
            case GGUF_TYPE_ARRAY:
                printf("[arr]   type=%d, n=%lu\n", kv->value.arr.type, (unsigned long)kv->value.arr.n);
                break;
        }
    }

    printf("\n--- Tensors ---\n");
    int64_t total_size = 0;
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        const gguf_tensor_info_t *ti = &ctx->tensors[i];
        printf("  %4ld: %-60s %5s  [", (long)i, ti->name, ggml_type_name(ti->type));
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%ld", (long)ti->dims[d]);
        }
        printf("]  %zu bytes\n", ti->size);
        total_size += ti->size;
    }
    printf("\nTotal tensor data: %.2f GB\n", total_size / (1024.0 * 1024.0 * 1024.0));

    // Print key model parameters if present
    printf("\n--- Model Parameters ---\n");
    const char *arch = gguf_get_str(ctx, "general.architecture");
    if (arch) printf("  Architecture:        %s\n", arch);

    const char *name = gguf_get_str(ctx, "general.name");
    if (name) printf("  Name:                %s\n", name);

    int64_t idx;
    idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        char key_buf[256];
        const char *a = ctx->kv[idx].value.str;

        #define PRINT_U32(suffix) do { \
            snprintf(key_buf, sizeof(key_buf), "%s.%s", a, suffix); \
            int64_t ki = gguf_find_key(ctx, key_buf); \
            if (ki >= 0) printf("  %-22s %u\n", suffix ":", ctx->kv[ki].value.u32); \
        } while(0)

        #define PRINT_F32(suffix) do { \
            snprintf(key_buf, sizeof(key_buf), "%s.%s", a, suffix); \
            int64_t ki = gguf_find_key(ctx, key_buf); \
            if (ki >= 0) printf("  %-22s %g\n", suffix ":", ctx->kv[ki].value.f32); \
        } while(0)

        PRINT_U32("block_count");
        PRINT_U32("embedding_length");
        PRINT_U32("feed_forward_length");
        PRINT_U32("attention.head_count");
        PRINT_U32("attention.head_count_kv");
        PRINT_U32("expert_count");
        PRINT_U32("expert_used_count");
        PRINT_U32("expert_feed_forward_length");
        PRINT_U32("shared_expert_feed_forward_length");
        PRINT_F32("rope.freq_base");
        PRINT_F32("attention.layer_norm_rms_epsilon");
        PRINT_F32("rope.dimension_count_fraction");
        PRINT_U32("rope.dimension_count");
        PRINT_U32("attention.full_attention_interval");
        PRINT_U32("attention.key_length");
        PRINT_U32("linear_attention.key_length");
        PRINT_U32("linear_attention.head_count");
        PRINT_U32("linear_attention.value_length");
        PRINT_U32("linear_attention.head_count_value");

        #undef PRINT_U32
        #undef PRINT_F32
    }

    // Count expert tensors
    int n_expert_layers = 0;
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "ffn_gate_exps")) {
            n_expert_layers++;
        }
    }
    if (n_expert_layers > 0) {
        printf("  MoE layers found:    %d\n", n_expert_layers);

        // Show first expert tensor dimensions
        for (int64_t i = 0; i < ctx->n_tensors; i++) {
            if (strstr(ctx->tensors[i].name, "ffn_gate_exps")) {
                const gguf_tensor_info_t *ti = &ctx->tensors[i];
                printf("  Expert tensor shape: [");
                for (uint32_t d = 0; d < ti->n_dims; d++) {
                    if (d > 0) printf(", ");
                    printf("%ld", (long)ti->dims[d]);
                }
                printf("]  (type: %s, size: %.2f MB)\n",
                       ggml_type_name(ti->type), ti->size / (1024.0 * 1024.0));
                break;
            }
        }
    }

    gguf_close(ctx);
    return 0;
}

// ---- dequant command: read and dequantize a tensor ----

static int cmd_dequant(const char *model_path, const char *tensor_name) {
    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    int64_t tid = gguf_find_tensor(ctx, tensor_name);
    if (tid < 0) {
        fprintf(stderr, "Tensor '%s' not found\n", tensor_name);
        gguf_close(ctx);
        return 1;
    }

    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    printf("Tensor: %s\n", ti->name);
    printf("Type:   %s\n", ggml_type_name(ti->type));
    printf("Dims:   [");
    for (uint32_t d = 0; d < ti->n_dims; d++) {
        if (d > 0) printf(", ");
        printf("%ld", (long)ti->dims[d]);
    }
    printf("]\n");
    printf("Size:   %zu bytes\n", ti->size);

    // Read raw tensor data
    void *raw = malloc(ti->size);
    if (!raw) { gguf_close(ctx); return 1; }

    size_t n_read = gguf_read_tensor(ctx, tid, raw, ti->size);
    if (n_read == 0) { free(raw); gguf_close(ctx); return 1; }

    // Compute element count
    int64_t nelements = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) {
        nelements *= ti->dims[d];
    }

    // Dequantize
    if (ti->type == GGML_TYPE_Q4_K) {
        int64_t n_show = nelements < 32 ? nelements : 32;
        float *f32 = malloc(n_show * sizeof(float));
        if (!f32) { free(raw); gguf_close(ctx); return 1; }

        // Dequantize at least one block (QK_K = 256 values)
        int64_t n_dequant = nelements < QK_K ? nelements : QK_K;
        float *full_block = malloc(n_dequant * sizeof(float));
        dequantize_row_q4_K((const block_q4_K *)raw, full_block, n_dequant);

        printf("\nFirst %ld dequantized values:\n", (long)n_show);
        for (int64_t i = 0; i < n_show; i++) {
            printf("  [%3ld] = %12.6f\n", (long)i, full_block[i]);
        }

        free(full_block);
        free(f32);
    } else if (ti->type == GGML_TYPE_F32) {
        const float *fp = (const float *)raw;
        int64_t n_show = nelements < 32 ? nelements : 32;
        printf("\nFirst %ld values:\n", (long)n_show);
        for (int64_t i = 0; i < n_show; i++) {
            printf("  [%3ld] = %12.6f\n", (long)i, fp[i]);
        }
    } else if (ti->type == GGML_TYPE_F16) {
        const uint16_t *hp = (const uint16_t *)raw;
        int64_t n_show = nelements < 32 ? nelements : 32;
        printf("\nFirst %ld values (converted from F16):\n", (long)n_show);
        for (int64_t i = 0; i < n_show; i++) {
            printf("  [%3ld] = %12.6f\n", (long)i, fp16_to_fp32(hp[i]));
        }
    } else {
        printf("\nDequantization not implemented for type %s\n", ggml_type_name(ti->type));
    }

    free(raw);
    gguf_close(ctx);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "info") == 0) {
        if (argc < 3) { fprintf(stderr, "Usage: %s info <model.gguf>\n", argv[0]); return 1; }
        return cmd_info(argv[2]);
    }

    if (strcmp(cmd, "dequant") == 0) {
        if (argc < 4) { fprintf(stderr, "Usage: %s dequant <model.gguf> <tensor_name>\n", argv[0]); return 1; }
        return cmd_dequant(argv[2], argv[3]);
    }

    if (strcmp(cmd, "benchmark") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s benchmark <store1.qmoe> [store2.qmoe ...]\n", argv[0]);
            return 1;
        }
        int n_stores = argc - 2;
        const char **paths = (const char **)&argv[2];

        nvme_io_t *io = nvme_io_init(paths, n_stores);
        if (!io) return 1;

        // Benchmark with 8 experts (typical MoE load)
        fprintf(stderr, "Benchmarking with 8 experts across %d drives...\n", n_stores);
        nvme_io_benchmark(io, 8, 100);

        // Also benchmark with 1, 2, 4 experts for comparison
        for (int n = 1; n <= 4; n *= 2) {
            nvme_io_benchmark(io, n, 100);
        }

        nvme_io_free(io);
        return 0;
    }

    if (strcmp(cmd, "generate") == 0) {
        // Parse: generate [--ram-cache MB] <model.gguf> <store1> [store2 ...] -- "prompt"
        if (argc < 5) {
            fprintf(stderr, "Usage: %s generate [--ram-cache MB] <model.gguf> <store1> [store2 ...] -- \"prompt\"\n", argv[0]);
            return 1;
        }

        // Parse optional flags before positional args
        int arg_start = 2;
        int cli_max_gen = -1;
        for (int i = 2; i < argc - 1; i++) {
            if (strcmp(argv[i], "--ram-cache") == 0) {
                setenv("QMOE_RAM_CACHE_MB", argv[i + 1], 1);
                fprintf(stderr, "RAM cache: set to %s MB\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--vram-cache") == 0) {
                setenv("QMOE_VRAM_CACHE_MB", argv[i + 1], 1);
                fprintf(stderr, "VRAM cache: set to %s MB\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--max-tokens") == 0) {
                cli_max_gen = atoi(argv[i + 1]);
                fprintf(stderr, "Max tokens: %d\n", cli_max_gen);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--no-eos") == 0) {
                setenv("QMOE_NO_EOS", "1", 1);
                fprintf(stderr, "EOS stopping disabled\n");
                arg_start = i + 1; continue;
            }
            if (strcmp(argv[i], "--freq-profile") == 0) {
                setenv("QMOE_FREQ_PROFILE", argv[i + 1], 1);
                fprintf(stderr, "Freq profile: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--car-threshold") == 0) {
                setenv("QMOE_CAR_THRESHOLD", argv[i + 1], 1);
                fprintf(stderr, "CAR threshold: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--spec-k") == 0) {
                setenv("QMOE_SPEC_K", argv[i + 1], 1);
                fprintf(stderr, "Speculative K: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--prefetch-budget") == 0) {
                setenv("QMOE_PREFETCH_BUDGET", argv[i + 1], 1);
                fprintf(stderr, "Prefetch budget: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            // Stop looking for flags at first non-flag arg
            if (argv[i][0] != '-' || strcmp(argv[i], "--") == 0) break;
        }

        const char *model_path = argv[arg_start];
        const char *prompt = NULL;
        int n_stores = 0;
        const char *store_paths[MAX_DRIVES];

        // Find the "--" separator
        int sep_idx = -1;
        for (int i = arg_start + 1; i < argc; i++) {
            if (strcmp(argv[i], "--") == 0) {
                sep_idx = i;
                break;
            }
        }

        if (sep_idx < 0 || sep_idx + 1 >= argc) {
            fprintf(stderr, "Error: missing '--' separator and prompt\n");
            fprintf(stderr, "Usage: %s generate [--ram-cache MB] <model.gguf> <store1> [store2 ...] -- \"prompt\"\n", argv[0]);
            return 1;
        }

        n_stores = sep_idx - (arg_start + 1);
        for (int i = 0; i < n_stores; i++) {
            store_paths[i] = argv[arg_start + 1 + i];
        }
        prompt = argv[sep_idx + 1];

        // Load model
        model_t *model = model_load(model_path, store_paths, n_stores);
        if (!model) return 1;

        // Load tokenizer
        tokenizer_t *tok = tokenizer_load(model->gguf);
        if (!tok) { model_free(model); return 1; }

        // Encode prompt - with or without chat template
        int max_tokens = 4096;
        int *tokens = malloc(max_tokens * sizeof(int));
        int n_prompt = 0;

        // Find special token IDs dynamically from vocab
        int im_start_id = -1, im_end_id = -1;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_start|>") == 0) im_start_id = i;
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_end|>") == 0) im_end_id = i;
        }
        if (im_start_id < 0 || im_end_id < 0) {
            fprintf(stderr, "Warning: could not find im_start/im_end tokens, using fallback\n");
            if (im_start_id < 0) im_start_id = 151644;
            if (im_end_id < 0) im_end_id = tok->eos_id;
        }
        fprintf(stderr, "Special tokens: im_start=%d, im_end=%d, eos=%d\n",
                im_start_id, im_end_id, tok->eos_id);

        if (getenv("QMOE_RAW")) {
            // Raw mode: just tokenize the prompt directly, no chat template
            n_prompt = tokenizer_encode(tok, prompt,
                                        tokens, max_tokens);
        } else {
            // Chat template
            // Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
            //         <|im_start|>user\n{prompt}<|im_end|>\n
            //         <|im_start|>assistant\n

            // System message
            tokens[n_prompt++] = im_start_id;
            n_prompt += tokenizer_encode(tok, "system\nYou are a helpful assistant.",
                                         tokens + n_prompt, max_tokens - n_prompt);
            tokens[n_prompt++] = im_end_id;
            n_prompt += tokenizer_encode(tok, "\n",
                                         tokens + n_prompt, max_tokens - n_prompt);

            // User message
            tokens[n_prompt++] = im_start_id;
            {
                char user_msg[4096];
                snprintf(user_msg, sizeof(user_msg), "user\n%s", prompt);
                n_prompt += tokenizer_encode(tok, user_msg,
                                             tokens + n_prompt, max_tokens - n_prompt);
            }
            tokens[n_prompt++] = im_end_id;
            n_prompt += tokenizer_encode(tok, "\n",
                                         tokens + n_prompt, max_tokens - n_prompt);

            // Assistant prefix
            tokens[n_prompt++] = im_start_id;
            n_prompt += tokenizer_encode(tok, "assistant\n",
                                         tokens + n_prompt, max_tokens - n_prompt);
        }

        fprintf(stderr, "\nPrompt (%d tokens): \"%s\"\n", n_prompt, prompt);
        fprintf(stderr, "Token IDs: ");
        for (int i = 0; i < n_prompt && i < 30; i++) {
            fprintf(stderr, "%d ", tokens[i]);
        }
        if (n_prompt > 30) fprintf(stderr, "...");
        fprintf(stderr, "\n\nGenerating...\n\n");

        // Generate
        int max_gen = (cli_max_gen > 0) ? cli_max_gen : 200;
        float temperature = 0.7f;
        float top_p = 0.9f;
        float rep_penalty = 1.1f;
        int rep_window = 64;

        srand((unsigned int)time(NULL));

        struct timespec t_start, t_now;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        // Process prompt tokens
        float *logits = NULL;
        for (int i = 0; i < n_prompt; i++) {
            struct timespec t_tok_start, t_tok_end;
            clock_gettime(CLOCK_MONOTONIC, &t_tok_start);
            logits = forward(model, tokens[i], i);
            clock_gettime(CLOCK_MONOTONIC, &t_tok_end);
            double tok_ms = (t_tok_end.tv_sec - t_tok_start.tv_sec) * 1000.0 +
                           (t_tok_end.tv_nsec - t_tok_start.tv_nsec) / 1e6;
            fprintf(stderr, "  prompt[%d/%d] tok=%d  %.0fms\n", i+1, n_prompt, tokens[i], tok_ms);
            fflush(stderr);
            if (!logits) {
                fprintf(stderr, "Error: forward pass failed at prompt token %d\n", i);
                break;
            }
        }

        int pos = n_prompt;
        int gen_count = 0;

        struct timespec t_gen_start;
        clock_gettime(CLOCK_MONOTONIC, &t_gen_start);

        // Track recent tokens for repetition penalty
        int *all_tokens = malloc((n_prompt + max_gen) * sizeof(int));
        memcpy(all_tokens, tokens, n_prompt * sizeof(int));
        int n_all = n_prompt;

        // Auto-regressive generation
        while (logits && gen_count < max_gen) {
            // Print top-5 predictions for first 3 generated tokens
            if (gen_count < 3) {
                typedef struct { float v; int i; } lp;
                lp top5[5] = {{-1e30,0},{-1e30,0},{-1e30,0},{-1e30,0},{-1e30,0}};
                for (int i = 0; i < model->hparams.vocab_size; i++) {
                    for (int j = 0; j < 5; j++) {
                        if (logits[i] > top5[j].v) {
                            for (int k = 4; k > j; k--) top5[k] = top5[k-1];
                            top5[j].v = logits[i]; top5[j].i = i;
                            break;
                        }
                    }
                }
                fprintf(stderr, "  gen[%d] top5:", gen_count);
                for (int j = 0; j < 5; j++) {
                    fprintf(stderr, " %d(\"%s\",%.2f)",
                            top5[j].i, tok->tokens[top5[j].i], top5[j].v);
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }

            // Apply repetition penalty
            int pen_start = n_all - rep_window;
            if (pen_start < 0) pen_start = 0;
            apply_repetition_penalty(logits, model->hparams.vocab_size,
                                     all_tokens + pen_start, n_all - pen_start,
                                     rep_penalty);

            int next_token = sample_token(logits, model->hparams.vocab_size,
                                          temperature, top_p);

            if (!getenv("QMOE_NO_EOS") &&
                (next_token == tok->eos_id || next_token == im_end_id)) {
                fprintf(stderr, "\n[EOS]\n");
                break;
            }

            all_tokens[n_all++] = next_token;

            const char *piece = tokenizer_decode(tok, next_token);
            printf("%s", piece);
            fflush(stdout);

            logits = forward(model, next_token, pos);
            pos++;
            gen_count++;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                         (t_now.tv_nsec - t_start.tv_nsec) / 1e9;

        printf("\n");
        double prompt_time = (t_gen_start.tv_sec - t_start.tv_sec) +
                             (t_gen_start.tv_nsec - t_start.tv_nsec) / 1e9;
        double gen_time = (t_now.tv_sec - t_gen_start.tv_sec) +
                          (t_now.tv_nsec - t_gen_start.tv_nsec) / 1e9;
        fprintf(stderr, "\n--- Stats ---\n");
        fprintf(stderr, "  Prompt tokens: %d (%.2f s, %.1f tok/s)\n",
                n_prompt, prompt_time, n_prompt > 0 ? n_prompt / prompt_time : 0.0);
        fprintf(stderr, "  Generated:     %d tokens (%.2f s)\n", gen_count, gen_time);
        fprintf(stderr, "  Total time:    %.2f s\n", elapsed);
        if (gen_count > 0) {
            fprintf(stderr, "  Speed:         %.2f tokens/s (generation only)\n", gen_count / gen_time);
        }
#ifdef QMOE_GPU
        if (model->gpu_ctx) {
            uint64_t relay_hops = 0, relay_payload = 0, relay_d2h = 0, relay_h2d = 0, relay_dma = 0;
            gpu_get_pingpong_relay_stats(model->gpu_ctx,
                                         &relay_hops,
                                         &relay_payload,
                                         &relay_d2h,
                                         &relay_h2d,
                                         &relay_dma);
            if (relay_hops > 0) {
                fprintf(stderr,
                        "  Relay total:   %llu hops, %.2f MiB payload, %.2f MiB DMA"
                        " (%.2f MiB D2H + %.2f MiB H2D)\n",
                        (unsigned long long)relay_hops,
                        (double)relay_payload / (1024.0 * 1024.0),
                        (double)relay_dma / (1024.0 * 1024.0),
                        (double)relay_d2h / (1024.0 * 1024.0),
                        (double)relay_h2d / (1024.0 * 1024.0));
            }
        }
#endif

        free(all_tokens);
        free(tokens);
        tokenizer_free(tok);
        model_free(model);
        return 0;
    }

    if (strcmp(cmd, "chat") == 0) {
        // Parse: chat [--ram-cache MB] <model.gguf> <store1> [store2 ...]
        if (argc < 4) {
            fprintf(stderr, "Usage: %s chat [--ram-cache MB] <model.gguf> <store1> [store2 ...]\n", argv[0]);
            return 1;
        }

        // Suppress profiling output
        setenv("QMOE_DEBUG", "0", 1);

        int arg_start = 2;
        int cli_max_gen = -1;
        for (int i = 2; i < argc - 1; i++) {
            if (strcmp(argv[i], "--ram-cache") == 0) {
                setenv("QMOE_RAM_CACHE_MB", argv[i + 1], 1);
                fprintf(stderr, "RAM cache: set to %s MB\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--vram-cache") == 0) {
                setenv("QMOE_VRAM_CACHE_MB", argv[i + 1], 1);
                fprintf(stderr, "VRAM cache: set to %s MB\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--max-tokens") == 0) {
                cli_max_gen = atoi(argv[i + 1]);
                fprintf(stderr, "Max tokens: %d\n", cli_max_gen);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--no-eos") == 0) {
                setenv("QMOE_NO_EOS", "1", 1);
                fprintf(stderr, "EOS stopping disabled\n");
                arg_start = i + 1; continue;
            }
            if (strcmp(argv[i], "--freq-profile") == 0) {
                setenv("QMOE_FREQ_PROFILE", argv[i + 1], 1);
                fprintf(stderr, "Freq profile: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--car-threshold") == 0) {
                setenv("QMOE_CAR_THRESHOLD", argv[i + 1], 1);
                fprintf(stderr, "CAR threshold: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--spec-k") == 0) {
                setenv("QMOE_SPEC_K", argv[i + 1], 1);
                fprintf(stderr, "Speculative K: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--prefetch-budget") == 0) {
                setenv("QMOE_PREFETCH_BUDGET", argv[i + 1], 1);
                fprintf(stderr, "Prefetch budget: %s\n", argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (argv[i][0] != '-') break;
        }

        const char *model_path = argv[arg_start];
        int n_stores = argc - (arg_start + 1);
        const char *store_paths[MAX_DRIVES];
        for (int i = 0; i < n_stores && i < MAX_DRIVES; i++)
            store_paths[i] = argv[arg_start + 1 + i];

        model_t *model = model_load(model_path, store_paths, n_stores);
        if (!model) return 1;

        tokenizer_t *tok = tokenizer_load(model->gguf);
        if (!tok) { model_free(model); return 1; }

        int im_start_id = -1, im_end_id = -1;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_start|>") == 0) im_start_id = i;
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_end|>") == 0) im_end_id = i;
        }
        if (im_start_id < 0) im_start_id = 151644;
        if (im_end_id < 0) im_end_id = tok->eos_id;

        float temperature = 0.7f;
        float top_p = 0.9f;
        float rep_penalty = 1.1f;
        int rep_window = 64;
        int max_gen = (cli_max_gen > 0) ? cli_max_gen : 2048;
        int max_ctx = 262144;
        srand((unsigned int)time(NULL));

        // Token buffer for full conversation
        int *all_tokens = malloc(max_ctx * sizeof(int));
        int n_all = 0;
        int pos = 0;

        // System prompt
        int tmp_buf_sz = 8192;
        int *tmp = malloc(tmp_buf_sz * sizeof(int));

        all_tokens[n_all++] = im_start_id;
        int n_enc = tokenizer_encode(tok, "system\nYou are a helpful assistant.",
                                      tmp, tmp_buf_sz);
        memcpy(all_tokens + n_all, tmp, n_enc * sizeof(int));
        n_all += n_enc;
        all_tokens[n_all++] = im_end_id;
        n_enc = tokenizer_encode(tok, "\n", tmp, tmp_buf_sz);
        memcpy(all_tokens + n_all, tmp, n_enc * sizeof(int));
        n_all += n_enc;

        fprintf(stderr, "\nModel loaded. Type your message (or /quit to exit).\n\n");

        char input_buf[4096];
        float *logits = NULL;

        while (1) {
            // Prompt
            fprintf(stdout, "\033[1;32m> \033[0m");
            fflush(stdout);

            if (!fgets(input_buf, sizeof(input_buf), stdin)) break;

            // Strip trailing newline
            size_t len = strlen(input_buf);
            while (len > 0 && (input_buf[len-1] == '\n' || input_buf[len-1] == '\r'))
                input_buf[--len] = '\0';

            if (len == 0) continue;
            if (strcmp(input_buf, "/quit") == 0 || strcmp(input_buf, "/exit") == 0) break;

            // Build user turn: <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
            all_tokens[n_all++] = im_start_id;
            {
                char user_msg[4096 + 8];
                snprintf(user_msg, sizeof(user_msg), "user\n%s", input_buf);
                n_enc = tokenizer_encode(tok, user_msg, tmp, tmp_buf_sz);
                memcpy(all_tokens + n_all, tmp, n_enc * sizeof(int));
                n_all += n_enc;
            }
            all_tokens[n_all++] = im_end_id;
            n_enc = tokenizer_encode(tok, "\n", tmp, tmp_buf_sz);
            memcpy(all_tokens + n_all, tmp, n_enc * sizeof(int));
            n_all += n_enc;
            all_tokens[n_all++] = im_start_id;
            n_enc = tokenizer_encode(tok, "assistant\n", tmp, tmp_buf_sz);
            memcpy(all_tokens + n_all, tmp, n_enc * sizeof(int));
            n_all += n_enc;

            // Process new tokens (from pos to n_all)
            struct timespec t0, t1;
            int new_prompt_tokens = n_all - pos;
            fprintf(stderr, "[processing %d tokens...]\r", new_prompt_tokens);
            fflush(stderr);

            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = pos; i < n_all; i++) {
                logits = forward(model, all_tokens[i], i);
                if (!logits) {
                    fprintf(stderr, "\nError: forward pass failed\n");
                    break;
                }
            }
            pos = n_all;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double prompt_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                               (t1.tv_nsec - t0.tv_nsec) / 1e6;

            // Clear the "processing" line
            fprintf(stderr, "                              \r");

            if (!logits) continue;

            // Generate response
            printf("\033[1;34m");  // blue for assistant
            fflush(stdout);

            int gen_count = 0;
            int thinking = 0;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            while (logits && gen_count < max_gen) {
                // Repetition penalty
                int pen_start = n_all - rep_window;
                if (pen_start < 0) pen_start = 0;
                apply_repetition_penalty(logits, model->hparams.vocab_size,
                                         all_tokens + pen_start, n_all - pen_start,
                                         rep_penalty);

                int next_token = sample_token(logits, model->hparams.vocab_size,
                                              temperature, top_p);

                if (!getenv("QMOE_NO_EOS") &&
                    (next_token == tok->eos_id || next_token == im_end_id)) break;

                all_tokens[n_all++] = next_token;

                const char *piece = tokenizer_decode(tok, next_token);

                // Handle <think>...</think> tags — hide thinking
                if (piece && strcmp(piece, "<think>") == 0) {
                    thinking = 1;
                } else if (piece && strcmp(piece, "</think>") == 0) {
                    thinking = 0;
                } else if (!thinking && piece) {
                    printf("%s", piece);
                    fflush(stdout);
                }

                logits = forward(model, next_token, pos);
                pos++;
                gen_count++;
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                            (t1.tv_nsec - t0.tv_nsec) / 1e6;

            printf("\033[0m\n");  // reset color

            // Append end tokens to history
            all_tokens[n_all++] = im_end_id;
            int nl_enc = tokenizer_encode(tok, "\n", tmp, tmp_buf_sz);
            memcpy(all_tokens + n_all, tmp, nl_enc * sizeof(int));
            n_all += nl_enc;
            // Don't need to process these through forward — they'll be processed next turn

            // Stats line
            fprintf(stderr, "\033[2m[%d tokens, %.1f tok/s | prompt: %.1fs]\033[0m\n\n",
                    gen_count,
                    gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0,
                    prompt_ms / 1000.0);
        }

        free(tmp);
        free(all_tokens);
        tokenizer_free(tok);
        model_free(model);
        return 0;
    }

    if (strcmp(cmd, "profile") == 0) {
        // Parse: profile [options] <model.gguf> <store1> [store2 ...] -- "prompt" <output.freq>
        if (argc < 6) {
            fprintf(stderr, "Usage: %s profile [options] <model.gguf> <store1> [store2 ...] -- \"prompt\" <output.freq>\n", argv[0]);
            return 1;
        }

        int arg_start = 2;
        int cli_max_gen = -1;
        for (int i = 2; i < argc - 1; i++) {
            if (strcmp(argv[i], "--ram-cache") == 0) {
                setenv("QMOE_RAM_CACHE_MB", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--max-tokens") == 0) {
                cli_max_gen = atoi(argv[i + 1]);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--no-eos") == 0) {
                setenv("QMOE_NO_EOS", "1", 1);
                arg_start = i + 1; continue;
            }
            if (argv[i][0] != '-' || strcmp(argv[i], "--") == 0) break;
        }

        const char *model_path = argv[arg_start];
        const char *prompt = NULL;
        const char *output_path = NULL;
        int n_stores = 0;
        const char *store_paths[MAX_DRIVES];

        int sep_idx = -1;
        for (int i = arg_start + 1; i < argc; i++) {
            if (strcmp(argv[i], "--") == 0) { sep_idx = i; break; }
        }

        if (sep_idx < 0 || sep_idx + 2 >= argc) {
            fprintf(stderr, "Error: expected -- \"prompt\" <output.freq>\n");
            return 1;
        }

        n_stores = sep_idx - (arg_start + 1);
        for (int i = 0; i < n_stores; i++)
            store_paths[i] = argv[arg_start + 1 + i];
        prompt = argv[sep_idx + 1];
        output_path = argv[sep_idx + 2];

        model_t *model = model_load(model_path, store_paths, n_stores);
        if (!model) return 1;

#ifdef QMOE_GPU
        if (model->gpu_ctx)
            gpu_enable_expert_freq(model->gpu_ctx);
#endif

        tokenizer_t *tok = tokenizer_load(model->gguf);
        if (!tok) { model_free(model); return 1; }

        // Encode with chat template
        int max_tokens = 4096;
        int *tokens = malloc(max_tokens * sizeof(int));
        int n_prompt = 0;

        int im_start_id = -1, im_end_id = -1;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_start|>") == 0) im_start_id = i;
            if (tok->tokens[i] && strcmp(tok->tokens[i], "<|im_end|>") == 0) im_end_id = i;
        }
        if (im_start_id < 0) im_start_id = 151644;
        if (im_end_id < 0) im_end_id = tok->eos_id;

        tokens[n_prompt++] = im_start_id;
        n_prompt += tokenizer_encode(tok, "system\nYou are a helpful assistant.",
                                     tokens + n_prompt, max_tokens - n_prompt);
        tokens[n_prompt++] = im_end_id;
        n_prompt += tokenizer_encode(tok, "\n", tokens + n_prompt, max_tokens - n_prompt);
        tokens[n_prompt++] = im_start_id;
        {
            char user_msg[4096];
            snprintf(user_msg, sizeof(user_msg), "user\n%s", prompt);
            n_prompt += tokenizer_encode(tok, user_msg, tokens + n_prompt, max_tokens - n_prompt);
        }
        tokens[n_prompt++] = im_end_id;
        n_prompt += tokenizer_encode(tok, "\n", tokens + n_prompt, max_tokens - n_prompt);
        tokens[n_prompt++] = im_start_id;
        n_prompt += tokenizer_encode(tok, "assistant\n", tokens + n_prompt, max_tokens - n_prompt);

        fprintf(stderr, "Profiling: %d prompt tokens, generating up to %d tokens...\n",
                n_prompt, cli_max_gen > 0 ? cli_max_gen : 200);

        // Forward pass: prompt + generation
        int max_gen = (cli_max_gen > 0) ? cli_max_gen : 200;
        float *logits = NULL;

        for (int i = 0; i < n_prompt; i++) {
            logits = forward(model, tokens[i], i);
            if (!logits) break;
        }

        int pos = n_prompt;
        int gen_count = 0;
        srand((unsigned int)time(NULL));

        while (logits && gen_count < max_gen) {
            int next_token = sample_token(logits, model->hparams.vocab_size, 0.7f, 0.9f);
            if (!getenv("QMOE_NO_EOS") &&
                (next_token == tok->eos_id || next_token == im_end_id))
                break;

            const char *piece = tokenizer_decode(tok, next_token);
            fprintf(stderr, "\r  gen %d: \"%s\"  ", gen_count, piece ? piece : "?");

            logits = forward(model, next_token, pos);
            pos++;
            gen_count++;
        }
        fprintf(stderr, "\nGenerated %d tokens.\n", gen_count);

        // Save frequency profile
        int total_tokens = n_prompt + gen_count;
#ifdef QMOE_GPU
        if (model->gpu_ctx) {
            int n_layers = 0, n_experts = 0;
            gpu_get_expert_dims(model->gpu_ctx, &n_layers, &n_experts);
            const uint32_t *freq = gpu_get_expert_freq(model->gpu_ctx);

            if (freq && n_layers > 0 && n_experts > 0) {
                freq_profile_t *fp = freq_profile_from_counts(freq, n_layers, n_experts, total_tokens);
                if (fp) {
                    freq_profile_save(output_path, fp);
                    fprintf(stderr, "Saved frequency profile to %s (%d entries, %d tokens)\n",
                            output_path, fp->n_entries, total_tokens);
                    freq_profile_free(fp);
                } else {
                    fprintf(stderr, "Error: failed to create frequency profile\n");
                }
            } else {
                fprintf(stderr, "Error: no frequency data collected\n");
            }
        }
#else
        fprintf(stderr, "Error: profile command requires GPU build (USE_GPU=1)\n");
        (void)total_tokens;
        (void)output_path;
#endif

        free(tokens);
        tokenizer_free(tok);
        model_free(model);
        return 0;
    }

    if (strcmp(cmd, "ppl") == 0) {
        // Parse: ppl [options] <model.gguf> <store1> [store2 ...] -- "text"
        if (argc < 5) {
            fprintf(stderr, "Usage: %s ppl [options] <model.gguf> <store1> [store2 ...] -- \"text\"\n", argv[0]);
            return 1;
        }

        int arg_start = 2;
        for (int i = 2; i < argc - 1; i++) {
            if (strcmp(argv[i], "--ram-cache") == 0) {
                setenv("QMOE_RAM_CACHE_MB", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--vram-cache") == 0) {
                setenv("QMOE_VRAM_CACHE_MB", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--freq-profile") == 0) {
                setenv("QMOE_FREQ_PROFILE", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--car-threshold") == 0) {
                setenv("QMOE_CAR_THRESHOLD", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--spec-k") == 0) {
                setenv("QMOE_SPEC_K", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--prefetch-budget") == 0) {
                setenv("QMOE_PREFETCH_BUDGET", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--ppl-ctx") == 0) {
                setenv("QMOE_PPL_CTX", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--ppl-chunks") == 0) {
                setenv("QMOE_PPL_CHUNKS", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--ppl-resume") == 0) {
                setenv("QMOE_PPL_RESUME", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (strcmp(argv[i], "--freq-output") == 0) {
                setenv("QMOE_FREQ_OUTPUT", argv[i + 1], 1);
                arg_start = i + 2; i++; continue;
            }
            if (argv[i][0] != '-' || strcmp(argv[i], "--") == 0) break;
        }

        const char *model_path = argv[arg_start];
        const char *text = NULL;
        int n_stores = 0;
        const char *store_paths[MAX_DRIVES];

        int sep_idx = -1;
        for (int i = arg_start + 1; i < argc; i++) {
            if (strcmp(argv[i], "--") == 0) { sep_idx = i; break; }
        }

        if (sep_idx < 0 || sep_idx + 1 >= argc) {
            fprintf(stderr, "Error: missing '--' separator and text\n");
            return 1;
        }

        n_stores = sep_idx - (arg_start + 1);
        for (int i = 0; i < n_stores; i++)
            store_paths[i] = argv[arg_start + 1 + i];
        text = argv[sep_idx + 1];

        // Optionally read text from file if it starts with @
        char *file_text = NULL;
        if (text[0] == '@') {
            FILE *f = fopen(text + 1, "r");
            if (!f) { fprintf(stderr, "Error: cannot open %s\n", text + 1); return 1; }
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            file_text = malloc(fsize + 1);
            fread(file_text, 1, fsize, f);
            file_text[fsize] = '\0';
            fclose(f);
            text = file_text;
        }

        // Clamp context length to ppl chunk size — the default 262144
        // allocates ~15GB of CPU KV cache that PPL never uses.
        {
            int ppl_ctx = 512;
            const char *ce = getenv("QMOE_PPL_CTX");
            if (ce) ppl_ctx = atoi(ce);
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", ppl_ctx);
            setenv("QMOE_CTX_SIZE", buf, 0);  // don't override if already set
        }

        model_t *model = model_load(model_path, store_paths, n_stores);
        if (!model) { free(file_text); return 1; }

#ifdef QMOE_GPU
        // Enable freq counting if --freq-output requested
        if (getenv("QMOE_FREQ_OUTPUT") && model->gpu_ctx)
            gpu_enable_expert_freq(model->gpu_ctx);
#endif

        tokenizer_t *tok = tokenizer_load(model->gguf);
        if (!tok) { model_free(model); free(file_text); return 1; }

        // Tokenize input text line-by-line (BPE tokenizer is O(n^2), so we
        // must avoid feeding the entire file as one sequence)
        int max_tokens = 512000;
        int *tokens = malloc(max_tokens * sizeof(int));

        // Add BOS at position 0 (matching llama.cpp: common_tokenize with add_bos=true)
        tokens[0] = tok->bos_id;
        int n_tokens = 1;

        // Tokenize line by line to avoid O(n^2) BPE on large files
        const char *line_start = text;
        while (*line_start && n_tokens < max_tokens - 1024) {
            // Find end of line
            const char *line_end = line_start;
            while (*line_end && *line_end != '\n') line_end++;

            // Tokenize this line (include the newline if present)
            size_t line_len = line_end - line_start + (*line_end == '\n' ? 1 : 0);
            if (line_len > 0) {
                char *line_buf = malloc(line_len + 1);
                memcpy(line_buf, line_start, line_len);
                line_buf[line_len] = '\0';
                n_tokens += tokenizer_encode(tok, line_buf, tokens + n_tokens,
                                              max_tokens - n_tokens);
                free(line_buf);
            }

            line_start = line_end;
            if (*line_start == '\n') line_start++;
        }
        fprintf(stderr, "PPL eval: tokenized %d tokens from %zu chars\n",
                n_tokens, strlen(text));

        // llama.cpp-compatible chunked perplexity evaluation
        // Default: 512-token non-overlapping chunks, score only last half
        int n_ctx = 512;
        const char *ctx_env = getenv("QMOE_PPL_CTX");
        if (ctx_env) n_ctx = atoi(ctx_env);
        if (n_ctx < 64) n_ctx = 64;

        int first = n_ctx / 2;  // only score positions [first, n_ctx-2]
        int n_chunks = n_tokens / n_ctx;

        // Optional chunk limit (for quick validation runs)
        const char *chunks_env = getenv("QMOE_PPL_CHUNKS");
        if (chunks_env) {
            int max_chunks = atoi(chunks_env);
            if (max_chunks > 0 && max_chunks < n_chunks) n_chunks = max_chunks;
        }
        int total_input = n_chunks * n_ctx;  // truncate to whole chunks

        fprintf(stderr, "PPL eval: %d tokens, %d chunks of %d (scoring last %d per chunk)\n",
                n_tokens, n_chunks, n_ctx, n_ctx - first - 1);
        fprintf(stderr, "  Tokens used: %d of %d (remainder %d discarded)\n",
                total_input, n_tokens, n_tokens - total_input);

        if (n_chunks < 1) {
            fprintf(stderr, "Error: need at least %d tokens for perplexity (got %d)\n",
                    n_ctx, n_tokens);
            free(tokens); free(file_text); tokenizer_free(tok); model_free(model);
            return 1;
        }

        double nll_sum = 0.0;
        double nll2_sum = 0.0;
        int n_eval = 0;
        int start_chunk = 0;

        // Resume support: load per-chunk NLL from checkpoint file
        const char *resume_path = getenv("QMOE_PPL_RESUME");
        if (resume_path) {
            FILE *rf = fopen(resume_path, "r");
            if (rf) {
                double chunk_nll, chunk_nll2;
                int chunk_n;
                int skipped = 0;
                while (fscanf(rf, "%lf %lf %d", &chunk_nll, &chunk_nll2, &chunk_n) == 3) {
                    if (chunk_n > 0) {
                        nll_sum += chunk_nll;
                        nll2_sum += chunk_nll2;
                        n_eval += chunk_n;
                    } else {
                        skipped++;
                    }
                    start_chunk++;
                }
                fclose(rf);
                if (start_chunk > 0) {
                    double resumed_ppl = n_eval > 0 ? exp(nll_sum / n_eval) : 0.0;
                    fprintf(stderr, "PPL resume: loaded %d chunks from %s (ppl=%.4f, %d scored, %d skipped)\n",
                            start_chunk, resume_path, resumed_ppl, n_eval, skipped);
                }
            }
            if (start_chunk >= n_chunks) {
                fprintf(stderr, "PPL resume: all %d chunks already complete\n", n_chunks);
                goto ppl_done;
            }
        }

        struct timespec t_start;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int consecutive_nan = 0;

        for (int c = start_chunk; c < n_chunks; c++) {
            int chunk_start = c * n_ctx;

            // Reset KV cache + SSM state for fresh context per chunk
            model_reset_state(model);

            double chunk_nll = 0.0;
            double chunk_nll2 = 0.0;
            int chunk_scored = 0;

            // Process all positions in this chunk
            for (int pos = 0; pos < n_ctx; pos++) {
                // Position 0: feed BOS (matching llama.cpp behavior)
                int input_tok = (pos == 0) ? tok->bos_id : tokens[chunk_start + pos];
                float *logits = forward(model, input_tok, pos);

                if (!logits) {
                    fprintf(stderr, "Warning: forward failed at chunk %d pos %d — skipping chunk\n", c, pos);
                    goto next_chunk;
                }

                // Check for NaN in logits (numerical stability diagnostic)
                if (isnan(logits[0]) || isnan(logits[1])) {
                    fprintf(stderr, "Warning: NaN logits at chunk %d pos %d (token %d) — skipping chunk\n",
                            c, pos, input_tok);
                    consecutive_nan++;
                    if (consecutive_nan >= 3) {
                        fprintf(stderr, "Error: %d consecutive NaN chunks — likely weight corruption, aborting\n",
                                consecutive_nan);
                        goto ppl_done;
                    }
                    goto next_chunk;
                }

                // Only score positions in [first, n_ctx-2]
                // logits at position pos predict token at chunk_start + pos + 1
                if (pos >= first && pos < n_ctx - 1) {
                    int target = tokens[chunk_start + pos + 1];

                    float max_logit = logits[0];
                    for (int v = 1; v < model->hparams.vocab_size; v++)
                        if (logits[v] > max_logit) max_logit = logits[v];

                    double sum_exp = 0.0;
                    for (int v = 0; v < model->hparams.vocab_size; v++)
                        sum_exp += exp((double)(logits[v] - max_logit));

                    double log_prob = (double)(logits[target] - max_logit) - log(sum_exp);
                    double nll = -log_prob;
                    if (isnan(nll) || isinf(nll)) {
                        fprintf(stderr, "Warning: NaN/Inf NLL at chunk %d pos %d target=%d "
                                "logit=%.4f max=%.4f sum_exp=%.6g\n",
                                c, pos, target, logits[target], max_logit, sum_exp);
                        // Skip this token rather than corrupting the whole run
                        continue;
                    }
                    chunk_nll += nll;
                    chunk_nll2 += nll * nll;
                    chunk_scored++;
                }
            }

            next_chunk:
            // Checkpoint: always write an entry so resume advances past this chunk.
            // Entries with n_scored=0 are skipped chunks (NaN/forward failure).
            if (resume_path) {
                FILE *wf = fopen(resume_path, "a");
                if (wf) {
                    fprintf(wf, "%.17g %.17g %d\n", chunk_nll, chunk_nll2, chunk_scored);
                    fclose(wf);
                }
            }

            if (chunk_scored == 0) {
                fprintf(stderr, "  [chunk %d/%d] SKIPPED (NaN/forward failure)\n",
                        c + 1, n_chunks);
                continue;
            }

            // Accumulate into running totals (successful chunk resets NaN counter)
            consecutive_nan = 0;
            nll_sum += chunk_nll;
            nll2_sum += chunk_nll2;
            n_eval += chunk_scored;

            double running_ppl = exp(nll_sum / n_eval);
            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
            int chunks_this_run = c - start_chunk + 1;
            int fwd_this_run = chunks_this_run * n_ctx;
            fprintf(stderr, "  [chunk %d/%d] ppl=%.4f +/- %.4f  (%d scored, %.1f tok/s)\n",
                    c + 1, n_chunks, running_ppl,
                    running_ppl * sqrt((nll2_sum / n_eval - (nll_sum / n_eval) * (nll_sum / n_eval)) / (n_eval - 1)),
                    n_eval, fwd_this_run / elapsed);
            fflush(stderr);
        }

        ppl_done: ;
        double ppl = exp(nll_sum / n_eval);

        // Uncertainty: std error of mean NLL, propagated through exp()
        double mean_nll = nll_sum / n_eval;
        double var_nll = nll2_sum / n_eval - mean_nll * mean_nll;
        double se_nll = sqrt(var_nll / (n_eval - 1));
        double ppl_err = ppl * se_nll;

        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double total_time = (t_end.tv_sec - t_start.tv_sec) +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        int total_chunks_done = n_eval / (n_ctx - first - 1);
        fprintf(stderr, "\n=== Perplexity (llama.cpp-compatible) ===\n");
        fprintf(stderr, "  Method:      %d-token chunks, score last %d\n", n_ctx, n_ctx - first - 1);
        fprintf(stderr, "  Chunks:      %d / %d\n", total_chunks_done, n_chunks);
        fprintf(stderr, "  Tokens:      %d scored of %d total\n", n_eval, total_input);
        fprintf(stderr, "  PPL:         %.4f +/- %.5f\n", ppl, ppl_err);
        fprintf(stderr, "  NLL/token:   %.4f +/- %.5f\n", mean_nll, se_nll);
        fprintf(stderr, "  Time:        %.1f s (%.1f tok/s)\n", total_time,
                (total_chunks_done - start_chunk) > 0
                    ? (double)(total_chunks_done - start_chunk) * n_ctx / total_time : 0.0);
        if (resume_path)
            fprintf(stderr, "  Checkpoint:  %s (%d chunks saved)\n", resume_path, total_chunks_done);

        // Print to stdout for easy parsing
        printf("%.4f\n", ppl);

#ifdef QMOE_GPU
        // Save frequency profile if requested
        const char *freq_out = getenv("QMOE_FREQ_OUTPUT");
        if (freq_out && model->gpu_ctx) {
            int n_layers_f = 0, n_experts_f = 0;
            gpu_get_expert_dims(model->gpu_ctx, &n_layers_f, &n_experts_f);
            const uint32_t *freq = gpu_get_expert_freq(model->gpu_ctx);
            if (freq && n_layers_f > 0) {
                freq_profile_t *fp = freq_profile_from_counts(freq, n_layers_f,
                                                               n_experts_f, n_eval);
                if (fp) {
                    freq_profile_save(freq_out, fp);
                    fprintf(stderr, "  Freq profile: %s (%d entries)\n",
                            freq_out, fp->n_entries);
                    freq_profile_free(fp);
                }
            }
        }
#endif

        free(tokens);
        free(file_text);
        tokenizer_free(tok);
        model_free(model);
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd);
    print_usage(argv[0]);
    return 1;
}
