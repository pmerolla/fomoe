#include "gguf.h"
#include "quant.h"
#include "tensor.h"
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "expert_store.h"
#include "nvme_io.h"
#include "sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// ---- Diagnostic helpers ----

static void check_floats(const char *label, const float *x, int64_t n) {
    float min_v = x[0], max_v = x[0], sum = 0, sum2 = 0;
    int nan_count = 0, inf_count = 0, zero_count = 0;

    for (int64_t i = 0; i < n; i++) {
        if (isnan(x[i])) { nan_count++; continue; }
        if (isinf(x[i])) { inf_count++; continue; }
        if (x[i] == 0.0f) zero_count++;
        if (x[i] < min_v) min_v = x[i];
        if (x[i] > max_v) max_v = x[i];
        sum += x[i];
        sum2 += x[i] * x[i];
    }

    int valid = n - nan_count - inf_count;
    float mean = valid > 0 ? sum / valid : 0;
    float rms = valid > 0 ? sqrtf(sum2 / valid) : 0;

    printf("  %-35s n=%-6ld min=%11.6f  max=%11.6f  mean=%11.6f  rms=%11.6f",
           label, (long)n, min_v, max_v, mean, rms);
    if (nan_count) printf("  NaN=%d", nan_count);
    if (inf_count) printf("  Inf=%d", inf_count);
    if (zero_count == n) printf("  ALL ZEROS!");
    printf("\n");

    // Print first 8 values
    printf("    first 8: ");
    for (int i = 0; i < 8 && i < n; i++) printf("%.6f ", x[i]);
    printf("\n");
}

// ---- Test 1: Tokenizer ----

static int test_tokenizer(const char *model_path) {
    printf("\n=== Test 1: Tokenizer ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    tokenizer_t *tok = tokenizer_load(ctx);
    if (!tok) { gguf_close(ctx); return 1; }

    // Print some special tokens
    printf("  vocab_size: %d\n", tok->vocab_size);
    printf("  n_merges: %d\n", tok->n_merges);
    printf("  BOS token (%d): \"%s\"\n", tok->bos_id, tokenizer_decode(tok, tok->bos_id));
    printf("  EOS token (%d): \"%s\"\n", tok->eos_id, tokenizer_decode(tok, tok->eos_id));
    printf("  PAD token (%d): \"%s\"\n", tok->pad_id, tokenizer_decode(tok, tok->pad_id));

    // Check what token 151935 is
    if (tok->vocab_size > 151935) {
        printf("  Token 151935: \"%s\"\n", tokenizer_decode(tok, 151935));
    }
    if (tok->vocab_size > 151934) {
        printf("  Token 151934: \"%s\"\n", tokenizer_decode(tok, 151934));
    }

    // Print a few sample tokens to understand the vocab
    printf("  Token 0: \"%s\"\n", tokenizer_decode(tok, 0));
    printf("  Token 1: \"%s\"\n", tokenizer_decode(tok, 1));
    printf("  Token 2: \"%s\"\n", tokenizer_decode(tok, 2));
    printf("  Token 220: \"%s\"\n", tokenizer_decode(tok, 220));
    printf("  Token 9707: \"%s\"\n", tokenizer_decode(tok, 9707));

    // Test encoding some simple strings
    const char *tests[] = { "Hello", "Hello, my name is", "1+1=", " the", "The" };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < n_tests; t++) {
        int ids[256];
        int n = tokenizer_encode(tok, tests[t], ids, 256);
        printf("  Encode \"%s\" -> %d tokens: [", tests[t], n);
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(", ");
            printf("%d(\"%s\")", ids[i], tokenizer_decode(tok, ids[i]));
        }
        printf("]\n");
    }

    // Verify first few merges
    printf("  First 5 merges:\n");
    for (int i = 0; i < 5 && i < tok->n_merges; i++) {
        printf("    %d: \"%s\" + \"%s\"\n", i,
               tok->merge_a[i] ? tok->merge_a[i] : "(null)",
               tok->merge_b[i] ? tok->merge_b[i] : "(null)");
    }

    tokenizer_free(tok);
    gguf_close(ctx);
    return 0;
}

// ---- Test 2: Dequantization correctness ----

static int test_dequant(const char *model_path) {
    printf("\n=== Test 2: Dequantization ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    // Test F32 tensor (attn_norm)
    {
        int64_t tid = gguf_find_tensor(ctx, "blk.0.attn_norm.weight");
        if (tid >= 0) {
            const gguf_tensor_info_t *ti = &ctx->tensors[tid];
            printf("  attn_norm: type=%s, dims=[", ggml_type_name(ti->type));
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%ld", (long)ti->dims[d]);
            }
            printf("], size=%zu\n", ti->size);

            float *data = malloc(ti->size);
            gguf_read_tensor(ctx, tid, data, ti->size);
            check_floats("attn_norm (F32)", data, ti->dims[0]);
            free(data);
        }
    }

    // Test Q4_K tensor (wq weight)
    {
        int64_t tid = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
        if (tid >= 0) {
            const gguf_tensor_info_t *ti = &ctx->tensors[tid];
            printf("  wq: type=%s, dims=[", ggml_type_name(ti->type));
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%ld", (long)ti->dims[d]);
            }
            printf("], size=%zu\n", ti->size);

            void *raw = malloc(ti->size);
            gguf_read_tensor(ctx, tid, raw, ti->size);

            // Dequantize first row (dims[0] elements)
            int row_elems = ti->dims[0];
            float *row = malloc(row_elems * sizeof(float));
            if (ti->type == GGML_TYPE_Q4_K) {
                dequantize_row_q4_K((const block_q4_K *)raw, row, row_elems);
            } else if (ti->type == GGML_TYPE_Q6_K) {
                dequantize_row_q6_K((const block_q6_K *)raw, row, row_elems);
            }
            check_floats("wq row 0 (dequant)", row, row_elems);

            // Also test vec_dot consistency: dequant-then-dot vs quantized-dot
            // Using a simple all-ones input vector
            float *ones = calloc(row_elems, sizeof(float));
            for (int i = 0; i < row_elems; i++) ones[i] = 1.0f;

            float sum_dequant = 0;
            for (int i = 0; i < row_elems; i++) sum_dequant += row[i];

            float sum_qdot = 0;
            if (ti->type == GGML_TYPE_Q4_K)
                sum_qdot = vec_dot_q4_K_f32((const block_q4_K *)raw, ones, row_elems);
            else if (ti->type == GGML_TYPE_Q6_K)
                sum_qdot = vec_dot_q6_K_f32((const block_q6_K *)raw, ones, row_elems);

            printf("  vec_dot consistency: dequant_sum=%.6f  qdot=%.6f  diff=%.9f\n",
                   sum_dequant, sum_qdot, fabsf(sum_dequant - sum_qdot));

            free(ones);
            free(row);
            free(raw);
        }
    }

    // Print types of key tensors
    printf("\n  Key tensor types:\n");
    const char *names[] = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_inp.weight",
    };
    for (int i = 0; i < (int)(sizeof(names)/sizeof(names[0])); i++) {
        int64_t tid = gguf_find_tensor(ctx, names[i]);
        if (tid >= 0) {
            const gguf_tensor_info_t *ti = &ctx->tensors[tid];
            printf("    %-40s %5s  [", names[i], ggml_type_name(ti->type));
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%ld", (long)ti->dims[d]);
            }
            printf("]\n");
        }
    }

    gguf_close(ctx);
    return 0;
}

// ---- Test 3: Embedding lookup ----

static int test_embedding(const char *model_path) {
    printf("\n=== Test 3: Embedding Lookup ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    int64_t tid = gguf_find_tensor(ctx, "token_embd.weight");
    if (tid < 0) { gguf_close(ctx); return 1; }

    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    printf("  token_embd: type=%s, dims=[%ld, %ld], size=%zu\n",
           ggml_type_name(ti->type), (long)ti->dims[0], (long)ti->dims[1], ti->size);

    void *embd_data = malloc(ti->size);
    if (!embd_data) { gguf_close(ctx); return 1; }
    gguf_read_tensor(ctx, tid, embd_data, ti->size);

    int n_embd = ti->dims[0];
    float *vec = malloc(n_embd * sizeof(float));

    int test_tokens[] = { 0, 1, 9707, 151935 };
    for (int t = 0; t < 4; t++) {
        int token_id = test_tokens[t];
        if (ti->type == GGML_TYPE_Q4_K) {
            int blocks_per_row = n_embd / QK_K;
            dequantize_row_q4_K((const block_q4_K *)embd_data + token_id * blocks_per_row,
                                vec, n_embd);
        } else if (ti->type == GGML_TYPE_F32) {
            memcpy(vec, (float *)embd_data + (size_t)token_id * n_embd, n_embd * sizeof(float));
        } else if (ti->type == GGML_TYPE_F16) {
            const uint16_t *fp16 = (const uint16_t *)embd_data;
            for (int i = 0; i < n_embd; i++)
                vec[i] = fp16_to_fp32(fp16[(size_t)token_id * n_embd + i]);
        }
        char label[64];
        snprintf(label, sizeof(label), "embedding token %d", token_id);
        check_floats(label, vec, n_embd);
    }

    free(vec);
    free(embd_data);
    gguf_close(ctx);
    return 0;
}

// ---- Test 4: Mat-vec multiply ----

static int test_matvec(const char *model_path) {
    printf("\n=== Test 4: Matrix-Vector Multiply ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    int64_t wq_tid = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    int64_t norm_tid = gguf_find_tensor(ctx, "blk.0.attn_norm.weight");
    int64_t embd_tid = gguf_find_tensor(ctx, "token_embd.weight");

    if (wq_tid < 0 || norm_tid < 0 || embd_tid < 0) {
        gguf_close(ctx);
        return 1;
    }

    const gguf_tensor_info_t *wq_ti = &ctx->tensors[wq_tid];
    int n_embd = ctx->tensors[embd_tid].dims[0];
    int q_dim = wq_ti->dims[1]; // output dim

    printf("  wq dims: [%ld, %ld] type=%s\n",
           (long)wq_ti->dims[0], (long)wq_ti->dims[1], ggml_type_name(wq_ti->type));
    printf("  n_embd=%d, q_dim=%d (M=%d, K=%d)\n", n_embd, q_dim, q_dim, n_embd);

    // Load weights
    void *wq_data = malloc(ctx->tensors[wq_tid].size);
    gguf_read_tensor(ctx, wq_tid, wq_data, ctx->tensors[wq_tid].size);

    float *norm_data = malloc(ctx->tensors[norm_tid].size);
    gguf_read_tensor(ctx, norm_tid, norm_data, ctx->tensors[norm_tid].size);

    void *embd_data = malloc(ctx->tensors[embd_tid].size);
    gguf_read_tensor(ctx, embd_tid, embd_data, ctx->tensors[embd_tid].size);

    // Get embedding for token 9707 (more interesting than token 0)
    float *x = calloc(n_embd, sizeof(float));
    if (ctx->tensors[embd_tid].type == GGML_TYPE_Q4_K) {
        int blocks_per_row = n_embd / QK_K;
        dequantize_row_q4_K((const block_q4_K *)embd_data + 9707 * blocks_per_row, x, n_embd);
    }
    check_floats("input (token 9707 embedding)", x, n_embd);

    // Apply RMS norm
    float *h = calloc(n_embd, sizeof(float));
    rms_norm(h, x, norm_data, n_embd, 1e-6f);
    check_floats("after RMS norm", h, n_embd);

    // Apply wq
    float *q = calloc(q_dim, sizeof(float));
    mat_vec_mul(q, wq_data, wq_ti->type, h, q_dim, n_embd);
    check_floats("Q = Wq @ h", q, q_dim);

    free(q);
    free(h);
    free(x);
    free(embd_data);
    free(norm_data);
    free(wq_data);
    gguf_close(ctx);
    return 0;
}

// ---- Test 4b: Q6_K vec_dot consistency ----

static int test_q6k_dot(const char *model_path) {
    printf("\n=== Test 4b: Q6_K vec_dot Consistency ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    // Find a Q6_K tensor - use attn_v which is Q6_K
    int64_t tid = gguf_find_tensor(ctx, "blk.0.attn_v.weight");
    if (tid < 0) {
        printf("  No Q6_K tensor found\n");
        gguf_close(ctx);
        return 0;
    }

    const gguf_tensor_info_t *ti = &ctx->tensors[tid];
    printf("  Using tensor: %s, type=%s, dims=[%ld, %ld]\n",
           ti->name, ggml_type_name(ti->type),
           (long)ti->dims[0], (long)ti->dims[1]);

    if (ti->type != GGML_TYPE_Q6_K) {
        printf("  Not Q6_K, skipping\n");
        gguf_close(ctx);
        return 0;
    }

    void *raw = malloc(ti->size);
    gguf_read_tensor(ctx, tid, raw, ti->size);

    int K = ti->dims[0];  // inner dim (n_embd = 2048)
    int M = ti->dims[1];  // outer dim (n_kv_heads * head_dim = 512)

    // Create a random-ish input vector
    float *vec = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) {
        vec[i] = sinf(i * 0.123f) * 0.1f;  // small deterministic values
    }

    // Method 1: dequantize full row, then dot product
    float *dequant_row = malloc(K * sizeof(float));
    float *results_dequant = malloc(M * sizeof(float));
    float *results_qdot = malloc(M * sizeof(float));

    int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; row++) {
        // Dequantize this row
        dequantize_row_q6_K((const block_q6_K *)raw + row * blocks_per_row,
                            dequant_row, K);

        // Manual dot product
        float sum = 0;
        for (int j = 0; j < K; j++) {
            sum += dequant_row[j] * vec[j];
        }
        results_dequant[row] = sum;

        // Quantized dot product
        results_qdot[row] = vec_dot_q6_K_f32(
            (const block_q6_K *)raw + row * blocks_per_row, vec, K);
    }

    // Compare
    float max_diff = 0, max_reldiff = 0;
    int nan_dq = 0, nan_qd = 0;
    for (int row = 0; row < M; row++) {
        if (isnan(results_dequant[row])) nan_dq++;
        if (isnan(results_qdot[row])) nan_qd++;
        float diff = fabsf(results_dequant[row] - results_qdot[row]);
        if (diff > max_diff) max_diff = diff;
        float denom = fabsf(results_dequant[row]);
        if (denom > 1e-8) {
            float reldiff = diff / denom;
            if (reldiff > max_reldiff) max_reldiff = reldiff;
        }
    }

    check_floats("dequant-then-dot", results_dequant, M);
    check_floats("quantized dot", results_qdot, M);
    printf("  Max abs diff: %.9f\n", max_diff);
    printf("  Max rel diff: %.9f\n", max_reldiff);
    printf("  NaN: dequant=%d, qdot=%d\n", nan_dq, nan_qd);

    if (max_diff > 0.001f) {
        printf("  WARNING: Q6_K vec_dot has significant error!\n");
        // Find first row with large diff
        for (int row = 0; row < M; row++) {
            float diff = fabsf(results_dequant[row] - results_qdot[row]);
            if (diff > 0.001f) {
                printf("  Row %d: dequant=%.6f qdot=%.6f diff=%.6f\n",
                       row, results_dequant[row], results_qdot[row], diff);
                break;
            }
        }
    }

    free(results_qdot);
    free(results_dequant);
    free(dequant_row);
    free(vec);
    free(raw);
    gguf_close(ctx);
    return 0;
}

// ---- Test 5: Expert store round-trip (multi-layer, all projections) ----

static int test_expert_store(const char *model_path, const char *store_path) {
    printf("\n=== Test 5: Expert Store Round-Trip (Multi-Layer) ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    expert_store_t *store = expert_store_open(store_path);
    if (!store) { gguf_close(ctx); return 1; }

    printf("  Store header:\n");
    printf("    n_moe_layers: %u\n", store->header.n_moe_layers);
    printf("    n_experts: %u\n", store->header.n_experts);
    printf("    expert_gate_size: %lu\n", (unsigned long)store->header.expert_gate_size);
    printf("    expert_up_size: %lu\n", (unsigned long)store->header.expert_up_size);
    printf("    expert_down_size: %lu\n", (unsigned long)store->header.expert_down_size);
    printf("    expert_stride: %lu\n", (unsigned long)store->header.expert_stride);

    // Verify layers 0, 1, 5, 6, 10, 47 with experts 0 and 63
    int test_layers[] = { 0, 1, 5, 6, 10, 47 };
    int test_experts[] = { 0, 63 };
    int n_test_layers = sizeof(test_layers) / sizeof(test_layers[0]);
    int n_test_experts = sizeof(test_experts) / sizeof(test_experts[0]);

    void *store_buf = NULL;
    if (posix_memalign(&store_buf, 4096, store->header.expert_stride) != 0) {
        expert_store_close(store);
        gguf_close(ctx);
        return 1;
    }

    int total_pass = 0, total_fail = 0;

    for (int li = 0; li < n_test_layers; li++) {
        int layer = test_layers[li];
        if ((uint32_t)layer >= store->header.n_moe_layers) continue;

        // Find GGUF tensors for this layer
        char tname[128];
        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", layer);
        int64_t gate_tid = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", layer);
        int64_t up_tid = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", layer);
        int64_t down_tid = gguf_find_tensor(ctx, tname);

        if (gate_tid < 0 || up_tid < 0 || down_tid < 0) {
            printf("  Layer %d: missing GGUF tensors, skip\n", layer);
            continue;
        }

        uint64_t gate_per_expert = ctx->tensors[gate_tid].size / store->header.n_experts;
        uint64_t up_per_expert   = ctx->tensors[up_tid].size / store->header.n_experts;
        uint64_t down_per_expert = ctx->tensors[down_tid].size / store->header.n_experts;

        for (int ei = 0; ei < n_test_experts; ei++) {
            int expert_id = test_experts[ei];

            // Read expert from store
            uint64_t offset = expert_store_offset(store, layer, expert_id);
            ssize_t nr = pread(store->fd, store_buf, store->header.expert_stride, offset);
            if (nr < (ssize_t)store->header.expert_stride) {
                printf("  Layer %d expert %d: pread failed (%zd)\n", layer, expert_id, nr);
                total_fail++;
                continue;
            }

            // Verify gate (use split-aware fd/offset for split GGUF)
            int gate_si = ctx->tensors[gate_tid].split_idx;
            void *gguf_slice = malloc(gate_per_expert);
            off_t gguf_off = ctx->splits[gate_si].data_offset + ctx->tensors[gate_tid].offset
                             + (uint64_t)expert_id * gate_per_expert;
            pread(ctx->splits[gate_si].fd, gguf_slice, gate_per_expert, gguf_off);

            int gate_ok = (memcmp(gguf_slice, store_buf, gate_per_expert) == 0);
            free(gguf_slice);

            // Verify up
            int up_si = ctx->tensors[up_tid].split_idx;
            gguf_slice = malloc(up_per_expert);
            gguf_off = ctx->splits[up_si].data_offset + ctx->tensors[up_tid].offset
                       + (uint64_t)expert_id * up_per_expert;
            pread(ctx->splits[up_si].fd, gguf_slice, up_per_expert, gguf_off);

            int up_ok = (memcmp(gguf_slice,
                                (uint8_t *)store_buf + gate_per_expert,
                                up_per_expert) == 0);
            free(gguf_slice);

            // Verify down
            int down_si = ctx->tensors[down_tid].split_idx;
            gguf_slice = malloc(down_per_expert);
            gguf_off = ctx->splits[down_si].data_offset + ctx->tensors[down_tid].offset
                       + (uint64_t)expert_id * down_per_expert;
            pread(ctx->splits[down_si].fd, gguf_slice, down_per_expert, gguf_off);

            int down_ok = (memcmp(gguf_slice,
                                  (uint8_t *)store_buf + gate_per_expert + up_per_expert,
                                  down_per_expert) == 0);

            if (gate_ok && up_ok && down_ok) {
                printf("  Layer %2d expert %3d: gate=PASS up=PASS down=PASS\n",
                       layer, expert_id);
                total_pass++;
            } else {
                printf("  Layer %2d expert %3d: gate=%s up=%s down=%s  *** FAIL ***\n",
                       layer, expert_id,
                       gate_ok ? "PASS" : "FAIL",
                       up_ok ? "PASS" : "FAIL",
                       down_ok ? "PASS" : "FAIL");
                total_fail++;

                // Show details for the first failure
                if (!down_ok) {
                    uint8_t *a = (uint8_t *)gguf_slice;
                    uint8_t *b = (uint8_t *)store_buf + gate_per_expert + up_per_expert;
                    int diff_count = 0;
                    uint64_t first_diff = 0;
                    for (uint64_t i = 0; i < down_per_expert; i++) {
                        if (a[i] != b[i]) {
                            if (diff_count == 0) first_diff = i;
                            diff_count++;
                        }
                    }
                    printf("    down: %d bytes differ, first at byte %lu\n",
                           diff_count, (unsigned long)first_diff);
                }
            }

            free(gguf_slice);
        }
    }

    printf("  Summary: %d passed, %d failed\n", total_pass, total_fail);

    // If any failures, cross-check using gguf_read_tensor (same method as prepare_experts)
    if (total_fail > 0) {
        printf("\n  Cross-checking failures using gguf_read_tensor (prepare_experts method)...\n");

        // Re-test the failed cases by reading the full tensor and slicing
        for (int li = 0; li < n_test_layers; li++) {
            int layer = test_layers[li];
            if ((uint32_t)layer >= store->header.n_moe_layers) continue;

            char tname2[128];
            snprintf(tname2, sizeof(tname2), "blk.%d.ffn_down_exps.weight", layer);
            int64_t dtid = gguf_find_tensor(ctx, tname2);
            if (dtid < 0) continue;

            const gguf_tensor_info_t *dti = &ctx->tensors[dtid];
            uint64_t dpe = dti->size / store->header.n_experts;

            // Read full tensor
            void *full_tensor = malloc(dti->size);
            if (!full_tensor) continue;
            size_t nr2 = gguf_read_tensor(ctx, dtid, full_tensor, dti->size);
            if (nr2 != dti->size) {
                printf("    Layer %d: gguf_read_tensor failed (got %zu of %zu)\n",
                       layer, nr2, dti->size);
                free(full_tensor);
                continue;
            }

            for (int ei = 0; ei < n_test_experts; ei++) {
                int eid = test_experts[ei];

                // Slice from full tensor (same as prepare_experts)
                uint8_t *gguf_down = (uint8_t *)full_tensor + (uint64_t)eid * dpe;

                // Read from store
                uint64_t soff = expert_store_offset(store, layer, eid);
                pread(store->fd, store_buf, store->header.expert_stride, soff);
                uint8_t *store_down = (uint8_t *)store_buf
                                      + store->header.expert_gate_size
                                      + store->header.expert_up_size;

                int match2 = (memcmp(gguf_down, store_down, dpe) == 0);
                printf("    Layer %2d expert %3d down (full-tensor method): %s\n",
                       layer, eid, match2 ? "PASS" : "FAIL");

                if (!match2) {
                    // Also check: does the pread method give same data as full-tensor slice?
                    int dsi = dti->split_idx;
                    void *pread_slice = malloc(dpe);
                    off_t poff = ctx->splits[dsi].data_offset + dti->offset + (uint64_t)eid * dpe;
                    ssize_t pn = pread(ctx->splits[dsi].fd, pread_slice, dpe, poff);
                    int pread_match = (memcmp(pread_slice, gguf_down, dpe) == 0);
                    printf("      pread vs full_tensor slice: %s (read %zd bytes at offset %lu)\n",
                           pread_match ? "MATCH" : "DIFFER", pn, (unsigned long)poff);

                    // Print first bytes of each to compare
                    printf("      GGUF slice first 16 bytes: ");
                    for (int b = 0; b < 16; b++) printf("%02x ", gguf_down[b]);
                    printf("\n");
                    printf("      Store down first 16 bytes: ");
                    for (int b = 0; b < 16; b++) printf("%02x ", store_down[b]);
                    printf("\n");

                    // Try to identify what's at the store offset - maybe wrong expert?
                    // Check if store_down matches any other expert's down data
                    for (uint32_t check_eid = 0; check_eid < store->header.n_experts; check_eid++) {
                        uint8_t *check_down = (uint8_t *)full_tensor + (uint64_t)check_eid * dpe;
                        if (memcmp(check_down, store_down, dpe) == 0) {
                            printf("      Store data matches GGUF expert %d (not %d)!\n",
                                   check_eid, eid);
                            break;
                        }
                    }
                    // Also check if store_down matches gate or up data
                    snprintf(tname2, sizeof(tname2), "blk.%d.ffn_gate_exps.weight", layer);
                    int64_t gtid = gguf_find_tensor(ctx, tname2);
                    if (gtid >= 0) {
                        void *gate_full = malloc(ctx->tensors[gtid].size);
                        if (gate_full && gguf_read_tensor(ctx, gtid, gate_full, ctx->tensors[gtid].size)) {
                            uint64_t gpe = ctx->tensors[gtid].size / store->header.n_experts;
                            for (uint32_t check_eid = 0; check_eid < store->header.n_experts; check_eid++) {
                                // Check if store_down data matches this expert's gate data
                                // (could indicate offset mixup between gate/up/down)
                                if (dpe <= gpe &&
                                    memcmp((uint8_t *)gate_full + (uint64_t)check_eid * gpe,
                                           store_down, 256) == 0) {
                                    printf("      Store down first 256B matches gate expert %d!\n",
                                           check_eid);
                                    break;
                                }
                            }
                        }
                        free(gate_full);
                    }

                    free(pread_slice);
                }
            }
            free(full_tensor);
        }
    }

    free(store_buf);
    expert_store_close(store);
    gguf_close(ctx);
    return 0;
}

// ---- Test 5b: Check for shared/missing expert tensors ----

static int test_tensor_survey(const char *model_path) {
    printf("\n=== Test 5b: Tensor Survey (shared experts, missing tensors) ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    // Check for shared expert tensors
    const char *shared_patterns[] = {
        "blk.0.ffn_gate_shexp", "blk.0.ffn_up_shexp", "blk.0.ffn_down_shexp",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.0.ffn_gate_exps_shared", "blk.0.ffn_norm_exps",
    };

    printf("  Checking for shared/additional expert tensors:\n");
    for (int i = 0; i < (int)(sizeof(shared_patterns)/sizeof(shared_patterns[0])); i++) {
        int64_t tid = gguf_find_tensor(ctx, shared_patterns[i]);
        if (tid >= 0) {
            const gguf_tensor_info_t *ti = &ctx->tensors[tid];
            printf("    FOUND: %-40s %5s [", ti->name, ggml_type_name(ti->type));
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%ld", (long)ti->dims[d]);
            }
            printf("]\n");
        } else {
            printf("    not found: %s\n", shared_patterns[i]);
        }
    }

    // List ALL tensors for layer 0 to find anything we might be missing
    printf("\n  All tensors with 'blk.0' prefix:\n");
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, "blk.0.", 6) == 0) {
            const gguf_tensor_info_t *ti = &ctx->tensors[i];
            printf("    %-50s %5s [", ti->name, ggml_type_name(ti->type));
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%ld", (long)ti->dims[d]);
            }
            printf("]  %zu bytes\n", ti->size);
        }
    }

    gguf_close(ctx);
    return 0;
}

// ---- Helper functions for test_forward (duplicated from static inference.c) ----

static void per_head_rms_norm_test(float *out, const float *x, const float *weight,
                                    int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + h * head_dim;
        float *oh = out + h * head_dim;
        rms_norm(oh, xh, weight, head_dim, eps);
    }
}

static void apply_rope_test(float *q, float *k, int pos, int head_dim, int n_heads,
                             int n_kv_heads, float freq_base) {
    for (int h = 0; h < n_heads + n_kv_heads; h++) {
        float *vec = (h < n_heads) ? q + h * head_dim : k + (h - n_heads) * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(freq_base, (float)i / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * cos_a - v1 * sin_a;
            vec[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// ---- Test 6: Forward pass diagnostics (first full-attention layer) ----

static int test_forward_layer0(const char *model_path, const char *store_path) {
    printf("\n=== Test 6: Forward Pass Diagnostics (First Full Attn Layer) ===\n");

    const char *store_paths[] = { store_path };
    model_t *m = model_load(model_path, store_paths, 1);
    if (!m) return 1;

    const model_hparams_t *hp = &m->hparams;

    printf("  Model loaded: %u layers, %u embd, %u heads, %u kv_heads, %u head_dim\n",
           hp->n_layers, hp->n_embd, hp->n_heads, hp->n_kv_heads, hp->head_dim);
    printf("  experts=%u used=%u, expert_intermediate=%u\n",
           hp->n_experts, hp->n_experts_used, hp->expert_intermediate);
    printf("  gate_type=%d, up_type=%d, L0 down_type=%d\n",
           hp->gate_type, hp->up_type, m->layers[0].down_type);
    printf("  expert_gate_size=%lu, expert_up_size=%lu, L0 down_size=%lu\n",
           (unsigned long)hp->expert_gate_size, (unsigned long)hp->expert_up_size,
           (unsigned long)m->layers[0].expert_down_size);

    // Find first full attention layer (SSM layers don't have wq/wk/wv/wo)
    uint32_t test_layer = 0;
    for (uint32_t i = 0; i < hp->n_layers; i++) {
        if (m->layers[i].attn_type == LAYER_ATTN_FULL) {
            test_layer = i;
            break;
        }
    }
    printf("  Using layer %u (first full attention layer)\n", test_layer);

    // Use token 9707 as input
    int token_id = 9707;
    int pos = 0;
    float *x = m->buf_x;
    float *h = m->buf_h;
    layer_weights_t *l = &m->layers[test_layer];

    // Embedding
    if (m->token_embd_type == GGML_TYPE_Q4_K) {
        size_t row_blocks = hp->n_embd / QK_K;
        const block_q4_K *embd = (const block_q4_K *)m->token_embd;
        dequantize_row_q4_K(embd + (size_t)token_id * row_blocks, x, hp->n_embd);
    } else if (m->token_embd_type == GGML_TYPE_F32) {
        memcpy(x, (float *)m->token_embd + (size_t)token_id * hp->n_embd,
               hp->n_embd * sizeof(float));
    }
    check_floats("embedding", x, hp->n_embd);

    // Attention norm
    rms_norm(h, x, l->attn_norm, hp->n_embd, hp->rms_norm_eps);
    check_floats("attn_norm(x)", h, hp->n_embd);

    // Q projection (fused Q + gate: full_q_dim = 2 * n_heads * head_dim)
    printf("  wq_type=%d wk_type=%d wv_type=%d wo_type=%d\n",
           l->wq_type, l->wk_type, l->wv_type, l->wo_type);

    mat_vec_mul(m->buf_qkv, l->wq, l->wq_type, h,
                hp->full_q_dim, hp->n_embd);
    float *q = m->buf_qkv;
    check_floats("Q+gate = Wq @ h", q, hp->n_heads * hp->head_dim);

    // K projection
    mat_vec_mul(m->buf_k, l->wk, l->wk_type, h,
                hp->n_kv_heads * hp->head_dim, hp->n_embd);
    check_floats("K = Wk @ h", m->buf_k, hp->n_kv_heads * hp->head_dim);

    // V projection
    mat_vec_mul(m->buf_v, l->wv, l->wv_type, h,
                hp->n_kv_heads * hp->head_dim, hp->n_embd);
    check_floats("V = Wv @ h", m->buf_v, hp->n_kv_heads * hp->head_dim);

    // Per-head norms
    if (l->q_norm) {
        per_head_rms_norm_test(q, q, l->q_norm,
                               hp->n_heads, hp->head_dim, hp->rms_norm_eps);
        check_floats("Q normed", q, hp->n_heads * hp->head_dim);
    }

    if (l->k_norm) {
        per_head_rms_norm_test(m->buf_k, m->buf_k, l->k_norm,
                               hp->n_kv_heads, hp->head_dim, hp->rms_norm_eps);
        check_floats("K normed", m->buf_k, hp->n_kv_heads * hp->head_dim);
    }

    // Partial RoPE (only first rope_dim dimensions per head)
    apply_rope_test(q, m->buf_k, pos, hp->head_dim, hp->n_heads,
                    hp->n_kv_heads, hp->rope_freq_base);
    check_floats("Q after RoPE", q, hp->n_heads * hp->head_dim);
    check_floats("K after RoPE", m->buf_k, hp->n_kv_heads * hp->head_dim);

    // At pos=0, attention is just V (single position -> softmax([score]) = [1.0])
    // So attn_out for each Q head = V of corresponding KV head
    memcpy(m->kv_cache[test_layer].k, m->buf_k, hp->n_kv_heads * hp->head_dim * sizeof(float));
    memcpy(m->kv_cache[test_layer].v, m->buf_v, hp->n_kv_heads * hp->head_dim * sizeof(float));

    memset(m->buf_attn, 0, hp->n_heads * hp->head_dim * sizeof(float));
    int heads_per_kv = hp->n_heads / hp->n_kv_heads;
    for (uint32_t hh = 0; hh < hp->n_heads; hh++) {
        int kv_h = hh / heads_per_kv;
        memcpy(m->buf_attn + hh * hp->head_dim,
               m->buf_v + kv_h * hp->head_dim,
               hp->head_dim * sizeof(float));
    }
    check_floats("attn out (V via GQA)", m->buf_attn, hp->n_heads * hp->head_dim);

    // Wo projection
    mat_vec_mul(m->buf_ffn, l->wo, l->wo_type, m->buf_attn,
                hp->n_embd, hp->n_heads * hp->head_dim);
    check_floats("Wo @ attn_out", m->buf_ffn, hp->n_embd);

    // Residual
    vec_add(x, x, m->buf_ffn, hp->n_embd);
    check_floats("x + attn (residual)", x, hp->n_embd);

    // FFN norm
    rms_norm(h, x, l->ffn_norm, hp->n_embd, hp->rms_norm_eps);
    check_floats("ffn_norm(x)", h, hp->n_embd);

    // Router
    mat_vec_mul(m->buf_router, l->router, GGML_TYPE_F32, h,
                hp->n_experts, hp->n_embd);
    check_floats("router logits", m->buf_router, hp->n_experts);

    // Softmax
    softmax(m->buf_router, hp->n_experts);
    check_floats("router probs", m->buf_router, hp->n_experts);

    // Top-k
    int expert_ids[16];
    float expert_scores[16];
    top_k(m->buf_router, hp->n_experts, hp->n_experts_used,
          expert_ids, expert_scores);

    printf("  Selected experts: ");
    for (uint32_t i = 0; i < hp->n_experts_used; i++) {
        printf("%d(%.4f) ", expert_ids[i], expert_scores[i]);
    }
    printf("\n");

    // Load and test one expert
    void *expert_buffers[16];
    if (m->nvme_io) {
        if (nvme_io_load_experts(m->nvme_io, test_layer, expert_ids,
                                 hp->n_experts_used, expert_buffers) != 0) {
            fprintf(stderr, "  Failed to load experts\n");
        } else {
            printf("  Experts loaded successfully\n");

            // Test first expert FFN
            uint8_t *expert_data = (uint8_t *)expert_buffers[0];
            void *gate_w = expert_data;
            void *up_w   = expert_data + hp->expert_gate_size;
            void *down_w = expert_data + hp->expert_gate_size + hp->expert_up_size;

            mat_vec_mul(m->buf_gate, gate_w, hp->gate_type, h,
                        hp->expert_intermediate, hp->n_embd);
            check_floats("expert 0 gate (pre-SiLU)", m->buf_gate, hp->expert_intermediate);

            silu(m->buf_gate, hp->expert_intermediate);
            check_floats("expert 0 gate (post-SiLU)", m->buf_gate, hp->expert_intermediate);

            mat_vec_mul(m->buf_up, up_w, hp->up_type, h,
                        hp->expert_intermediate, hp->n_embd);
            check_floats("expert 0 up", m->buf_up, hp->expert_intermediate);

            vec_mul(m->buf_gate, m->buf_gate, m->buf_up, hp->expert_intermediate);
            check_floats("expert 0 gate*up", m->buf_gate, hp->expert_intermediate);

            mat_vec_mul(m->buf_down, down_w, l->down_type,
                        m->buf_gate, hp->n_embd, hp->expert_intermediate);
            check_floats("expert 0 down", m->buf_down, hp->n_embd);
        }
    }

    model_free(m);
    return 0;
}

// ---- Test 7: Full forward pass (all layers) ----

static int test_full_forward(const char *model_path, const char *store_path) {
    printf("\n=== Test 7: Full Forward Pass ===\n");

    const char *store_paths[] = { store_path };
    model_t *m = model_load(model_path, store_paths, 1);
    if (!m) return 1;

    // Test with token 9707 at position 0
    int token_id = 9707;
    printf("  Running forward(token=%d, pos=0)...\n", token_id);

    float *logits = forward(m, token_id, 0);
    if (!logits) {
        printf("  forward() returned NULL!\n");
        model_free(m);
        return 1;
    }

    check_floats("logits", logits, m->hparams.vocab_size);

    // Check for NaN/Inf in logits
    int nan_count = 0, inf_count = 0;
    for (uint32_t i = 0; i < m->hparams.vocab_size; i++) {
        if (isnan(logits[i])) nan_count++;
        if (isinf(logits[i])) inf_count++;
    }
    printf("  Logits: NaN=%d, Inf=%d out of %u\n",
           nan_count, inf_count, m->hparams.vocab_size);

    if (nan_count == 0 && inf_count == 0) {
        // Find top 10 tokens by logit value
        printf("  Top 10 tokens by logit:\n");
        int top_ids[10];
        float top_vals[10];
        top_k(logits, m->hparams.vocab_size, 10, top_ids, top_vals);

        tokenizer_t *tok = tokenizer_load(m->gguf);
        for (int i = 0; i < 10; i++) {
            const char *piece = tok ? tokenizer_decode(tok, top_ids[i]) : "?";
            printf("    %2d: token %6d (logit=%10.4f) \"%s\"\n",
                   i, top_ids[i], top_vals[i], piece);
        }
        if (tok) tokenizer_free(tok);

        // Test sampling
        printf("  Argmax sample: %d\n", sample_argmax(logits, m->hparams.vocab_size));
        printf("  Sampled (T=0.7, p=0.9): %d\n",
               sample_token(logits, m->hparams.vocab_size, 0.7f, 0.9f));
    }

    // Now test with token 0 (to compare)
    printf("\n  Running forward(token=0, pos=0) [reset KV cache]...\n");
    // Reset KV cache (only full attention layers have allocated KV cache)
    for (uint32_t i = 0; i < m->hparams.n_layers; i++) {
        if (!m->kv_cache[i].k) continue;  // SSM layers have NULL kv_cache
        memset(m->kv_cache[i].k, 0, m->hparams.ctx_len * m->hparams.n_kv_heads *
               m->hparams.head_dim * sizeof(float));
        memset(m->kv_cache[i].v, 0, m->hparams.ctx_len * m->hparams.n_kv_heads *
               m->hparams.head_dim * sizeof(float));
    }

    logits = forward(m, 0, 0);
    if (logits) {
        check_floats("logits (token 0)", logits, m->hparams.vocab_size);
    }

    model_free(m);
    return 0;
}

// ---- Test 8: NVMe io_uring Data Integrity Check ----
// Loads experts via nvme_io_load_experts() and compares byte-for-byte against
// the original GGUF tensors to verify io_uring reads are correct.

static int test_nvme_integrity(const char *model_path, const char **store_paths, int n_stores) {
    printf("\n=== Test 8: NVMe io_uring Data Integrity ===\n");

    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) return 1;

    nvme_io_t *io = nvme_io_init(store_paths, n_stores);
    if (!io) { gguf_close(ctx); return 1; }

    int test_layers[] = { 0, 23, 47 };
    int test_experts[] = { 0, 63, 127 };
    int n_test_layers = sizeof(test_layers) / sizeof(test_layers[0]);
    int n_test_experts = sizeof(test_experts) / sizeof(test_experts[0]);

    int total_pass = 0, total_fail = 0;

    for (int li = 0; li < n_test_layers; li++) {
        int layer = test_layers[li];

        // Find GGUF tensors for this layer
        char tname[128];
        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", layer);
        int64_t gate_tid = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", layer);
        int64_t up_tid = gguf_find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", layer);
        int64_t down_tid = gguf_find_tensor(ctx, tname);

        if (gate_tid < 0 || up_tid < 0 || down_tid < 0) {
            printf("  Layer %d: missing GGUF tensors, skip\n", layer);
            continue;
        }

        uint32_t n_exp = io->drives[0].store->header.n_experts;
        uint64_t gate_per_expert = ctx->tensors[gate_tid].size / n_exp;
        uint64_t up_per_expert   = ctx->tensors[up_tid].size / n_exp;
        uint64_t down_per_expert = ctx->tensors[down_tid].size / n_exp;

        for (int ei = 0; ei < n_test_experts; ei++) {
            int expert_id = test_experts[ei];

            // Load expert via io_uring
            int ids[] = { expert_id };
            void *out_bufs[1];
            int rc = nvme_io_load_experts(io, layer, ids, 1, out_bufs);
            if (rc != 0) {
                printf("  Layer %2d expert %3d: io_uring load FAILED\n", layer, expert_id);
                total_fail++;
                continue;
            }

            uint8_t *expert_data = (uint8_t *)out_bufs[0];

            // Compare gate (use split-aware fd/offset)
            int gate_si = ctx->tensors[gate_tid].split_idx;
            void *gguf_slice = malloc(gate_per_expert);
            off_t gguf_off = ctx->splits[gate_si].data_offset + ctx->tensors[gate_tid].offset
                             + (uint64_t)expert_id * gate_per_expert;
            pread(ctx->splits[gate_si].fd, gguf_slice, gate_per_expert, gguf_off);
            int gate_ok = (memcmp(gguf_slice, expert_data, gate_per_expert) == 0);
            free(gguf_slice);

            // Compare up
            int up_si = ctx->tensors[up_tid].split_idx;
            gguf_slice = malloc(up_per_expert);
            gguf_off = ctx->splits[up_si].data_offset + ctx->tensors[up_tid].offset
                       + (uint64_t)expert_id * up_per_expert;
            pread(ctx->splits[up_si].fd, gguf_slice, up_per_expert, gguf_off);
            int up_ok = (memcmp(gguf_slice, expert_data + gate_per_expert, up_per_expert) == 0);
            free(gguf_slice);

            // Compare down
            int down_si = ctx->tensors[down_tid].split_idx;
            gguf_slice = malloc(down_per_expert);
            gguf_off = ctx->splits[down_si].data_offset + ctx->tensors[down_tid].offset
                       + (uint64_t)expert_id * down_per_expert;
            pread(ctx->splits[down_si].fd, gguf_slice, down_per_expert, gguf_off);
            int down_ok = (memcmp(gguf_slice, expert_data + gate_per_expert + up_per_expert,
                                  down_per_expert) == 0);
            free(gguf_slice);

            if (gate_ok && up_ok && down_ok) {
                printf("  Layer %2d expert %3d: PASS (gate/up/down all match GGUF)\n",
                       layer, expert_id);
                total_pass++;
            } else {
                printf("  Layer %2d expert %3d: FAIL gate=%s up=%s down=%s\n",
                       layer, expert_id,
                       gate_ok ? "ok" : "MISMATCH",
                       up_ok ? "ok" : "MISMATCH",
                       down_ok ? "ok" : "MISMATCH");
                total_fail++;
            }
        }
    }

    printf("  Summary: %d passed, %d failed\n", total_pass, total_fail);

    nvme_io_free(io);
    gguf_close(ctx);
    return total_fail > 0 ? 1 : 0;
}

// ---- Test 9: Cross-Drive Consistency Check ----
// Loads the same expert from each drive independently and verifies all copies match.

static int test_cross_drive_consistency(const char **store_paths, int n_stores) {
    printf("\n=== Test 9: Cross-Drive Consistency ===\n");

    if (n_stores < 2) {
        printf("  Need at least 2 stores for cross-drive check, have %d\n", n_stores);
        return 0;
    }

    // Open each store individually to read the same expert from each
    expert_store_t *stores[MAX_DRIVES];
    for (int i = 0; i < n_stores; i++) {
        stores[i] = expert_store_open(store_paths[i]);
        if (!stores[i]) {
            fprintf(stderr, "  Failed to open store %s\n", store_paths[i]);
            for (int j = 0; j < i; j++) expert_store_close(stores[j]);
            return 1;
        }
    }

    uint64_t stride = stores[0]->header.expert_stride;
    void *buf0 = NULL, *buf1 = NULL;
    posix_memalign(&buf0, QMOE_ALIGNMENT, stride);
    posix_memalign(&buf1, QMOE_ALIGNMENT, stride);

    int test_layers[] = { 0, 23, 47 };
    int test_experts[] = { 0, 63, 127 };
    int total_pass = 0, total_fail = 0;

    for (int li = 0; li < 3; li++) {
        int layer = test_layers[li];
        if ((uint32_t)layer >= stores[0]->header.n_moe_layers) continue;

        for (int ei = 0; ei < 3; ei++) {
            int eid = test_experts[ei];

            // Read from drive 0 as reference
            uint64_t offset = expert_store_offset(stores[0], layer, eid);
            pread(stores[0]->fd, buf0, stride, offset);

            // Compare against all other drives
            int all_match = 1;
            for (int d = 1; d < n_stores; d++) {
                offset = expert_store_offset(stores[d], layer, eid);
                pread(stores[d]->fd, buf1, stride, offset);

                if (memcmp(buf0, buf1, stride) != 0) {
                    printf("  Layer %2d expert %3d: drive 0 vs drive %d MISMATCH\n",
                           layer, eid, d);
                    all_match = 0;
                }
            }

            if (all_match) {
                printf("  Layer %2d expert %3d: all %d drives identical\n",
                       layer, eid, n_stores);
                total_pass++;
            } else {
                total_fail++;
            }
        }
    }

    printf("  Summary: %d passed, %d failed\n", total_pass, total_fail);

    free(buf0);
    free(buf1);
    for (int i = 0; i < n_stores; i++) expert_store_close(stores[i]);
    return total_fail > 0 ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [store1.qmoe] [store2.qmoe] ...\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *store_path = argc > 2 ? argv[2] : NULL;
    int n_stores = argc > 2 ? argc - 2 : 0;
    const char **store_paths = n_stores > 0 ? (const char **)&argv[2] : NULL;

    test_tokenizer(model_path);
    test_dequant(model_path);
    test_embedding(model_path);
    test_matvec(model_path);
    test_q6k_dot(model_path);

    if (store_path) {
        test_expert_store(model_path, store_path);
        test_tensor_survey(model_path);
        test_forward_layer0(model_path, store_path);
        test_full_forward(model_path, store_path);

        // NVMe io_uring sanity checks
        test_nvme_integrity(model_path, store_paths, n_stores);
        if (n_stores >= 2) {
            test_cross_drive_consistency(store_paths, n_stores);
        }
    } else {
        printf("\n(Skipping tests 5-9: no expert store path provided)\n");
    }

    printf("\n=== All tests completed ===\n");
    return 0;
}
