/*
 * tools/expert_cluster.c - Expert clustering and shared-basis analysis
 *
 * Tests whether experts within a layer cluster into groups whose centroids
 * approximate individual expert FFN accurately, and whether residuals
 * (expert - centroid) have rapidly-decaying singular values.
 *
 * Usage: tools/expert_cluster <store.qmoe> [--hidden 3072] [--intermediate 1024]
 *        [--layer N] [--probes N] [--k N]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include "expert_store.h"
#include "quant.h"
#include "tensor.h"
#include "gguf.h"

// LAPACK SVD
extern void sgesvd_(const char *jobu, const char *jobvt,
                    const int *m, const int *n, float *a, const int *lda,
                    float *s, float *u, const int *ldu,
                    float *vt, const int *ldvt,
                    float *work, const int *lwork, int *info);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float cosine_sim(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i];
    }
    return (float)(dot / (sqrt(na)*sqrt(nb) + 1e-15));
}

static void rand_vector(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) { v[i] = (float)drand48()*2-1; norm += v[i]*v[i]; }
    float s = sqrtf((float)n) / sqrtf(norm);
    for (int i = 0; i < n; i++) v[i] *= s;
}

static const char *dtype_name(int t) {
    switch (t) {
    case GGML_TYPE_Q4_K: return "Q4_K"; case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K"; case GGML_TYPE_Q8_0: return "Q8_0";
    default: return "???";
    }
}

// --- FFN using quantized expert data ---
static void ffn_quant(const void *gate_data, int gate_type,
                      const void *up_data, int up_type,
                      const void *down_data, int down_type,
                      int hidden, int inter,
                      const float *h, float *output,
                      float *buf_g, float *buf_u) {
    mat_vec_mul(buf_g, gate_data, gate_type, h, inter, hidden);
    silu(buf_g, inter);
    mat_vec_mul(buf_u, up_data, up_type, h, inter, hidden);
    vec_mul(buf_g, buf_g, buf_u, inter);
    mat_vec_mul(output, down_data, down_type, buf_g, hidden, inter);
}

// --- FFN using F32 matrices (for centroid) ---
static void ffn_f32(const float *gate, const float *up, const float *down,
                    int hidden, int inter,
                    const float *h, float *output,
                    float *buf_g, float *buf_u) {
    mat_vec_mul(buf_g, gate, GGML_TYPE_F32, h, inter, hidden);
    silu(buf_g, inter);
    mat_vec_mul(buf_u, up, GGML_TYPE_F32, h, inter, hidden);
    vec_mul(buf_g, buf_g, buf_u, inter);
    mat_vec_mul(output, down, GGML_TYPE_F32, buf_g, hidden, inter);
}

// --- Dequantize weight matrix ---
static float *dequantize_weight(const void *data, int type, int rows, int cols) {
    float *out = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!out) return NULL;
    size_t rb = gguf_tensor_nbytes(type, cols);
    for (int r = 0; r < rows; r++) {
        const uint8_t *rp = (const uint8_t *)data + (size_t)r * rb;
        float *op = out + (size_t)r * cols;
        switch (type) {
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *)rp, op, cols); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *)rp, op, cols); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *)rp, op, cols); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *)rp, op, cols); break;
        default: free(out); return NULL;
        }
    }
    return out;
}

// --- K-means clustering ---
// data: [n][dim] row-major, assignments: [n] output cluster IDs
static void kmeans(const float *data, int n, int dim, int k,
                   int *assignments, int max_iter) {
    float *centroids = (float *)calloc((size_t)k * dim, sizeof(float));

    // Initialize: pick k random distinct points
    int *picked = (int *)calloc(k, sizeof(int));
    for (int i = 0; i < k; i++) {
        int p;
        do {
            p = (int)(drand48() * n);
            int dup = 0;
            for (int j = 0; j < i; j++) if (picked[j] == p) { dup = 1; break; }
            if (!dup) break;
        } while (1);
        picked[i] = p;
        memcpy(centroids + (size_t)i * dim, data + (size_t)p * dim, dim * sizeof(float));
    }
    free(picked);

    int *counts = (int *)calloc(k, sizeof(int));

    for (int iter = 0; iter < max_iter; iter++) {
        // Assign
        int changed = 0;
        for (int i = 0; i < n; i++) {
            const float *xi = data + (size_t)i * dim;
            float best_sim = -2;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                float sim = cosine_sim(xi, centroids + (size_t)c * dim, dim);
                if (sim > best_sim) { best_sim = sim; best_c = c; }
            }
            if (assignments[i] != best_c) { changed++; assignments[i] = best_c; }
        }
        if (changed == 0) break;

        // Recompute centroids
        memset(centroids, 0, (size_t)k * dim * sizeof(float));
        memset(counts, 0, k * sizeof(int));
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            const float *xi = data + (size_t)i * dim;
            float *ci = centroids + (size_t)c * dim;
            for (int j = 0; j < dim; j++) ci[j] += xi[j];
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                float *ci = centroids + (size_t)c * dim;
                for (int j = 0; j < dim; j++) ci[j] *= inv;
            }
        }
    }
    free(centroids);
    free(counts);
}

// --- SVD singular values only ---
static float *svd_spectrum(const float *A, int M, int K, int *out_n) {
    int m = K, n = M, mn = (m < n) ? m : n;
    int lda = m, one = 1;
    float *wa = (float *)malloc((size_t)m * n * sizeof(float));
    float *s = (float *)malloc(mn * sizeof(float));
    memcpy(wa, A, (size_t)m * n * sizeof(float));

    int lwork = -1; float wopt; int info;
    sgesvd_("N","N", &m, &n, wa, &lda, s, NULL, &one, NULL, &one, &wopt, &lwork, &info);
    lwork = (int)wopt;
    float *work = (float *)malloc(lwork * sizeof(float));
    sgesvd_("N","N", &m, &n, wa, &lda, s, NULL, &one, NULL, &one, work, &lwork, &info);
    free(work); free(wa);

    if (info != 0) { free(s); return NULL; }
    *out_n = mn;
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <store.qmoe> [--hidden 3072] [--intermediate 1024]\n"
            "       [--layer N] [--probes 16] [--k 16]\n", argv[0]);
        return 1;
    }

    const char *store_path = argv[1];
    int hidden = 3072, inter = 1024;
    int target_layer = -1; // -1 = middle layer
    int n_probes = 16;
    int target_k = 16;

    for (int i = 2; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--hidden"))       hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--intermediate")) inter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layer"))   target_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probes"))  n_probes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k"))       target_k = atoi(argv[++i]);
    }

    expert_store_t *store = expert_store_open(store_path);
    if (!store) return 1;

    int fd = open(store_path, O_RDONLY);
    if (fd < 0) { perror("open"); expert_store_close(store); return 1; }

    int n_layers = store->header.n_moe_layers;
    int n_experts = store->header.n_experts;
    int gate_type = store->header.quant_type;
    int up_type = gate_type;

    if (target_layer < 0) target_layer = n_layers / 2;
    int down_type = store->layer_index[target_layer].down_type;
    if (down_type == 0) down_type = gate_type;

    uint64_t gate_sz = store->header.expert_gate_size;
    uint64_t up_sz   = store->header.expert_up_size;
    uint64_t stride  = store->layer_index[target_layer].expert_stride;
    if (stride == 0) stride = store->header.expert_stride;

    printf("=== Expert Clustering Analysis ===\n\n");
    printf("Store:  %s (%d layers, %d experts)\n", store_path, n_layers, n_experts);
    printf("Layer:  %d (gate=%s, up=%s, down=%s)\n",
           target_layer, dtype_name(gate_type), dtype_name(up_type), dtype_name(down_type));
    printf("Config: hidden=%d, intermediate=%d, probes=%d\n\n", hidden, inter, n_probes);

    srand48(42);

    // Generate shared probe inputs
    float **probes = (float **)malloc(n_probes * sizeof(float *));
    for (int p = 0; p < n_probes; p++) {
        probes[p] = (float *)malloc(hidden * sizeof(float));
        rand_vector(probes[p], hidden);
    }

    // Scratch buffers
    float *buf_g = (float *)malloc(inter * sizeof(float));
    float *buf_u = (float *)malloc(inter * sizeof(float));
    float *buf_out = (float *)malloc(hidden * sizeof(float));

    // ===== Phase 1: Functional fingerprints =====
    printf("Phase 1: Computing functional fingerprints for %d experts...\n", n_experts);
    int fp_dim = n_probes * hidden;
    float **fingerprints = (float **)malloc(n_experts * sizeof(float *));
    double t0 = now_sec();

    for (int e = 0; e < n_experts; e++) {
        fingerprints[e] = (float *)malloc(fp_dim * sizeof(float));
        uint64_t offset = expert_store_offset(store, target_layer, e);
        void *ebuf = malloc(stride);
        ssize_t nr = pread(fd, ebuf, stride, offset);
        if (nr < (ssize_t)stride) {
            fprintf(stderr, "read failed expert %d\n", e);
            memset(fingerprints[e], 0, fp_dim * sizeof(float));
            free(ebuf); continue;
        }

        uint8_t *ep = (uint8_t *)ebuf;
        for (int p = 0; p < n_probes; p++) {
            ffn_quant(ep, gate_type, ep + gate_sz, up_type,
                      ep + gate_sz + up_sz, down_type,
                      hidden, inter, probes[p],
                      fingerprints[e] + p * hidden, buf_g, buf_u);
        }
        free(ebuf);
        if ((e+1) % 64 == 0) { printf("  %d/%d\n", e+1, n_experts); fflush(stdout); }
    }
    printf("  Done in %.1fs\n\n", now_sec() - t0);

    // ===== Phase 2: Pairwise similarity =====
    printf("Phase 2: Pairwise similarity matrix...\n");
    int n_pairs = n_experts * (n_experts - 1) / 2;
    float *sims = (float *)malloc(n_pairs * sizeof(float));
    int si = 0;
    float sim_max = -2, sim_min = 2;
    double sim_sum = 0;
    int best_i = 0, best_j = 1;
    float best_sim = -2;

    for (int i = 0; i < n_experts; i++) {
        for (int j = i+1; j < n_experts; j++) {
            float s = cosine_sim(fingerprints[i], fingerprints[j], fp_dim);
            sims[si++] = s;
            sim_sum += s;
            if (s > sim_max) sim_max = s;
            if (s < sim_min) sim_min = s;
            if (s > best_sim) { best_sim = s; best_i = i; best_j = j; }
        }
    }
    double sim_mean = sim_sum / n_pairs;
    double sim_var = 0;
    for (int i = 0; i < n_pairs; i++) {
        double d = sims[i] - sim_mean;
        sim_var += d * d;
    }
    sim_var /= n_pairs;

    printf("  Pairs: %d\n", n_pairs);
    printf("  Similarity: min=%.4f  max=%.4f  mean=%.4f  std=%.4f\n",
           sim_min, sim_max, sim_mean, sqrt(sim_var));
    printf("  Most similar: expert %d & %d (cos=%.4f)\n\n", best_i, best_j, best_sim);

    // Histogram
    printf("  Similarity histogram:\n");
    int hist[20] = {0};
    for (int i = 0; i < n_pairs; i++) {
        int bin = (int)((sims[i] + 1.0f) * 10);  // [-1,1] → [0,20)
        if (bin < 0) bin = 0; if (bin >= 20) bin = 19;
        hist[bin]++;
    }
    for (int b = 0; b < 20; b++) {
        float lo = -1.0f + b * 0.1f, hi = lo + 0.1f;
        printf("    [%+.1f,%+.1f): %5d ", lo, hi, hist[b]);
        int bar = (hist[b] * 50) / n_pairs;
        for (int i = 0; i < bar; i++) printf("#");
        printf("\n");
    }
    printf("\n");
    free(sims);

    // ===== Phase 3: K-means clustering =====
    // Flatten fingerprints for kmeans
    float *fp_flat = (float *)malloc((size_t)n_experts * fp_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++)
        memcpy(fp_flat + (size_t)e * fp_dim, fingerprints[e], fp_dim * sizeof(float));

    int ks[] = {4, 8, 16, 32, 64};
    int n_ks = sizeof(ks) / sizeof(ks[0]);
    int *assignments = (int *)calloc(n_experts, sizeof(int));

    printf("Phase 3: K-means clustering + centroid FFN accuracy\n\n");

    for (int ki = 0; ki < n_ks; ki++) {
        int k = ks[ki];
        if (k > n_experts) continue;

        memset(assignments, 0, n_experts * sizeof(int));
        srand48(42 + k);  // reproducible per k
        kmeans(fp_flat, n_experts, fp_dim, k, assignments, 100);

        // Cluster sizes
        int *sizes = (int *)calloc(k, sizeof(int));
        for (int e = 0; e < n_experts; e++) sizes[assignments[e]]++;

        printf("  k=%d: sizes=[", k);
        for (int c = 0; c < k; c++) printf("%d%s", sizes[c], c<k-1?",":"");
        printf("]\n");

        // Within-cluster similarity
        double wc_sim_sum = 0;
        int wc_count = 0;
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < n_experts; i++) {
                if (assignments[i] != c) continue;
                for (int j = i+1; j < n_experts; j++) {
                    if (assignments[j] != c) continue;
                    wc_sim_sum += cosine_sim(fingerprints[i], fingerprints[j], fp_dim);
                    wc_count++;
                }
            }
        }
        double wc_mean = wc_count > 0 ? wc_sim_sum / wc_count : 0;
        printf("    Within-cluster similarity: %.4f (from %d pairs)\n", wc_mean, wc_count);

        // Centroid FFN accuracy (use the target k or all)
        if (k == target_k || ki == n_ks - 1) {
            printf("    Computing weight-space centroid FFN accuracy...\n");
            double cs_total = 0;
            int cs_count = 0;

            for (int c = 0; c < k; c++) {
                if (sizes[c] == 0) continue;
                int csize = sizes[c];

                // Compute centroid by accumulating dequantized weights
                size_t gate_elems = (size_t)inter * hidden;
                size_t down_elems = (size_t)hidden * inter;
                float *cen_gate = (float *)calloc(gate_elems, sizeof(float));
                float *cen_up   = (float *)calloc(gate_elems, sizeof(float));
                float *cen_down = (float *)calloc(down_elems, sizeof(float));
                float inv_n = 1.0f / csize;

                for (int e = 0; e < n_experts; e++) {
                    if (assignments[e] != c) continue;
                    uint64_t offset = expert_store_offset(store, target_layer, e);
                    void *ebuf = malloc(stride);
                    pread(fd, ebuf, stride, offset);
                    uint8_t *ep = (uint8_t *)ebuf;

                    float *g = dequantize_weight(ep, gate_type, inter, hidden);
                    float *u = dequantize_weight(ep + gate_sz, up_type, inter, hidden);
                    float *d = dequantize_weight(ep + gate_sz + up_sz, down_type, hidden, inter);

                    for (size_t i = 0; i < gate_elems; i++) {
                        cen_gate[i] += g[i] * inv_n;
                        cen_up[i]   += u[i] * inv_n;
                    }
                    for (size_t i = 0; i < down_elems; i++)
                        cen_down[i] += d[i] * inv_n;

                    free(g); free(u); free(d); free(ebuf);
                }

                // Test centroid accuracy vs each expert
                float *out_real = (float *)malloc(hidden * sizeof(float));
                float *out_cen  = (float *)malloc(hidden * sizeof(float));
                double cluster_cs = 0;

                for (int e = 0; e < n_experts; e++) {
                    if (assignments[e] != c) continue;
                    uint64_t offset = expert_store_offset(store, target_layer, e);
                    void *ebuf = malloc(stride);
                    pread(fd, ebuf, stride, offset);
                    uint8_t *ep = (uint8_t *)ebuf;

                    double expert_cs = 0;
                    for (int p = 0; p < n_probes; p++) {
                        ffn_quant(ep, gate_type, ep + gate_sz, up_type,
                                  ep + gate_sz + up_sz, down_type,
                                  hidden, inter, probes[p], out_real, buf_g, buf_u);
                        ffn_f32(cen_gate, cen_up, cen_down,
                                hidden, inter, probes[p], out_cen, buf_g, buf_u);
                        expert_cs += cosine_sim(out_real, out_cen, hidden);
                    }
                    expert_cs /= n_probes;
                    cluster_cs += expert_cs;
                    free(ebuf);
                }
                cluster_cs /= csize;
                cs_total += cluster_cs * csize;
                cs_count += csize;

                printf("      Cluster %2d (%3d experts): centroid cos_sim = %.4f\n",
                       c, csize, cluster_cs);

                free(out_real); free(out_cen);
                free(cen_gate); free(cen_up); free(cen_down);
            }
            printf("    Overall centroid cos_sim: %.4f\n\n",
                   cs_count > 0 ? cs_total / cs_count : 0.0);
        }
        free(sizes);
    }

    // ===== Phase 4: Residual SVD for one clustering =====
    printf("Phase 4: Residual SVD analysis (k=%d)\n\n", target_k);
    srand48(42 + target_k);
    memset(assignments, 0, n_experts * sizeof(int));
    kmeans(fp_flat, n_experts, fp_dim, target_k, assignments, 100);

    // Pick 3 clusters (largest ones)
    int *sizes = (int *)calloc(target_k, sizeof(int));
    for (int e = 0; e < n_experts; e++) sizes[assignments[e]]++;

    // Sort clusters by size (descending)
    int *order = (int *)malloc(target_k * sizeof(int));
    for (int c = 0; c < target_k; c++) order[c] = c;
    for (int i = 0; i < target_k - 1; i++)
        for (int j = i + 1; j < target_k; j++)
            if (sizes[order[j]] > sizes[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    int n_clusters_svd = 3;
    if (n_clusters_svd > target_k) n_clusters_svd = target_k;
    int n_experts_per = 3;  // SVD for this many experts per cluster

    int ranks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_ranks = sizeof(ranks)/sizeof(ranks[0]);
    int mn = (inter < hidden) ? inter : hidden;
    while (n_ranks > 0 && ranks[n_ranks-1] > mn) n_ranks--;

    for (int ci = 0; ci < n_clusters_svd; ci++) {
        int c = order[ci];
        int csize = sizes[c];
        printf("  Cluster %d (%d experts):\n", c, csize);

        // Compute centroid
        size_t ge = (size_t)inter * hidden;
        size_t de = (size_t)hidden * inter;
        float *cen_g = (float *)calloc(ge, sizeof(float));
        float *cen_u = (float *)calloc(ge, sizeof(float));
        float *cen_d = (float *)calloc(de, sizeof(float));
        float inv_n = 1.0f / csize;

        for (int e = 0; e < n_experts; e++) {
            if (assignments[e] != c) continue;
            uint64_t off = expert_store_offset(store, target_layer, e);
            void *eb = malloc(stride); pread(fd, eb, stride, off);
            uint8_t *ep = (uint8_t *)eb;
            float *g = dequantize_weight(ep, gate_type, inter, hidden);
            float *u = dequantize_weight(ep + gate_sz, up_type, inter, hidden);
            float *d = dequantize_weight(ep + gate_sz + up_sz, down_type, hidden, inter);
            for (size_t i = 0; i < ge; i++) { cen_g[i] += g[i]*inv_n; cen_u[i] += u[i]*inv_n; }
            for (size_t i = 0; i < de; i++) cen_d[i] += d[i]*inv_n;
            free(g); free(u); free(d); free(eb);
        }

        // SVD of residuals for a few experts
        int done = 0;
        for (int e = 0; e < n_experts && done < n_experts_per; e++) {
            if (assignments[e] != c) continue;
            done++;

            uint64_t off = expert_store_offset(store, target_layer, e);
            void *eb = malloc(stride); pread(fd, eb, stride, off);
            uint8_t *ep = (uint8_t *)eb;
            float *g = dequantize_weight(ep, gate_type, inter, hidden);
            float *u = dequantize_weight(ep + gate_sz, up_type, inter, hidden);
            float *d = dequantize_weight(ep + gate_sz + up_sz, down_type, hidden, inter);
            free(eb);

            // Compute residual = expert - centroid
            for (size_t i = 0; i < ge; i++) { g[i] -= cen_g[i]; u[i] -= cen_u[i]; }
            for (size_t i = 0; i < de; i++) d[i] -= cen_d[i];

            printf("    Expert %d residual SVD:\n", e);
            int sn;
            float *sg = svd_spectrum(g, inter, hidden, &sn);
            float *su = svd_spectrum(u, inter, hidden, &sn);
            float *sd = svd_spectrum(d, hidden, inter, &sn);

            if (sg && su && sd) {
                double tg=0, tu=0, td=0;
                for (int i = 0; i < sn; i++) {
                    tg += (double)sg[i]*sg[i]; tu += (double)su[i]*su[i]; td += (double)sd[i]*sd[i];
                }
                printf("      %6s %8s %8s %8s\n", "Rank", "gate", "up", "down");
                for (int ri = 0; ri < n_ranks; ri++) {
                    int R = ranks[ri];
                    double eg=0, eu=0, ed=0;
                    for (int i = 0; i < R && i < sn; i++) {
                        eg += (double)sg[i]*sg[i]; eu += (double)su[i]*su[i]; ed += (double)sd[i]*sd[i];
                    }
                    printf("      %6d %7.1f%% %7.1f%% %7.1f%%\n",
                           R, 100*eg/tg, 100*eu/tu, 100*ed/td);
                }
            }
            free(sg); free(su); free(sd);
            free(g); free(u); free(d);
        }
        printf("\n");
        free(cen_g); free(cen_u); free(cen_d);
    }

    // Cleanup
    free(order); free(sizes); free(assignments); free(fp_flat);
    for (int e = 0; e < n_experts; e++) free(fingerprints[e]);
    free(fingerprints);
    for (int p = 0; p < n_probes; p++) free(probes[p]);
    free(probes);
    free(buf_g); free(buf_u); free(buf_out);
    close(fd);
    expert_store_close(store);
    return 0;
}
