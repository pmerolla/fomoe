/*
 * tools/svd_experiment.c - SVD analysis of MoE expert weight matrices
 *
 * Measures singular value spectrum and low-rank FFN approximation accuracy.
 * Answers: "Can SVD-approximate expert FFN work for routing prediction?"
 *
 * Usage: tools/svd_experiment <store.qmoe> [--hidden N] [--intermediate N]
 *                             [--experts N] [--layers N] [--inputs N]
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
#include "gguf.h"

// LAPACK SVD (Fortran interface)
extern void sgesvd_(const char *jobu, const char *jobvt,
                    const int *m, const int *n, float *a, const int *lda,
                    float *s, float *u, const int *ldu,
                    float *vt, const int *ldvt,
                    float *work, const int *lwork, int *info);

// ---- Dequantize a weight matrix [rows, cols] to F32 ----
static float *dequantize_weight(const void *data, enum ggml_dtype type,
                                int rows, int cols) {
    float *out = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!out) return NULL;
    size_t row_bytes = gguf_tensor_nbytes(type, cols);

    for (int r = 0; r < rows; r++) {
        const uint8_t *rp = (const uint8_t *)data + (size_t)r * row_bytes;
        float *op = out + (size_t)r * cols;
        switch (type) {
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K((const block_q4_K *)rp, op, cols); break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K((const block_q5_K *)rp, op, cols); break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K((const block_q6_K *)rp, op, cols); break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0((const block_q8_0 *)rp, op, cols); break;
        default:
            fprintf(stderr, "ERROR: unsupported quant type %d\n", type);
            free(out); return NULL;
        }
    }
    return out;
}

// ---- SVD of row-major A[M][K] ----
// Row-major A[M][K] has same memory layout as col-major A^T[K][M].
// LAPACK computes SVD(A^T) = U_l * S * VT_l.
// Since A = VT_l^T * S * U_l^T, and col-major VT_l = row-major U_a,
// col-major U_l = row-major VT_a.
//
// out_U:  [M, min(M,K)] row-major   (caller frees)
// out_S:  [min(M,K)]                (caller allocates)
// out_VT: [min(M,K), K] row-major   (caller frees)
static int compute_svd(const float *A, int M, int K,
                       float **out_U, float *out_S, float **out_VT) {
    int m = K, n = M;
    int mn = (m < n) ? m : n;
    int lda = m, ldu = m, ldvt = mn;

    float *work_a = (float *)malloc((size_t)m * n * sizeof(float));
    float *lap_u  = (float *)malloc((size_t)m * mn * sizeof(float));
    float *lap_vt = (float *)malloc((size_t)mn * n * sizeof(float));
    if (!work_a || !lap_u || !lap_vt) {
        free(work_a); free(lap_u); free(lap_vt); return -1;
    }
    memcpy(work_a, A, (size_t)m * n * sizeof(float));

    int lwork = -1;
    float work_opt;
    int info;
    sgesvd_("S", "S", &m, &n, work_a, &lda, out_S,
            lap_u, &ldu, lap_vt, &ldvt, &work_opt, &lwork, &info);
    lwork = (int)work_opt;
    float *work = (float *)malloc(lwork * sizeof(float));

    sgesvd_("S", "S", &m, &n, work_a, &lda, out_S,
            lap_u, &ldu, lap_vt, &ldvt, work, &lwork, &info);

    free(work);
    free(work_a);
    if (info != 0) {
        fprintf(stderr, "sgesvd failed: info=%d\n", info);
        free(lap_u); free(lap_vt); return info;
    }

    *out_U  = lap_vt;  // col-major VT_l = row-major U_a [M, mn]
    *out_VT = lap_u;   // col-major U_l  = row-major VT_a [mn, K]
    return 0;
}

// ---- Rank-R matvec: (A truncated to rank R) @ x ----
// U[M][mn], S[mn], VT[mn][K], x[K] → out[M]
static void svd_matvec(const float *U, const float *S, const float *VT,
                       int M, int K, int mn, int R,
                       const float *x, float *out) {
    if (R > mn) R = mn;
    float *proj = (float *)calloc(R, sizeof(float));

    // proj[i] = S[i] * (VT[i,:] @ x)
    for (int i = 0; i < R; i++) {
        float sum = 0;
        const float *vt_row = VT + (size_t)i * K;
        for (int j = 0; j < K; j++) sum += vt_row[j] * x[j];
        proj[i] = sum * S[i];
    }
    // out[i] = U[i,:R] @ proj
    for (int i = 0; i < M; i++) {
        float sum = 0;
        const float *u_row = U + (size_t)i * mn;
        for (int j = 0; j < R; j++) sum += u_row[j] * proj[j];
        out[i] = sum;
    }
    free(proj);
}

// ---- Real FFN: output = down @ (SiLU(gate @ h) * (up @ h)) ----
static void compute_ffn(const float *gate, const float *up, const float *down,
                        int hidden, int inter, const float *h, float *output) {
    float *g = (float *)malloc(inter * sizeof(float));
    float *u = (float *)malloc(inter * sizeof(float));

    for (int i = 0; i < inter; i++) {
        float sg = 0, su = 0;
        const float *gr = gate + (size_t)i * hidden;
        const float *ur = up + (size_t)i * hidden;
        for (int j = 0; j < hidden; j++) { sg += gr[j]*h[j]; su += ur[j]*h[j]; }
        g[i] = sg / (1.0f + expf(-sg));  // SiLU
        u[i] = su;
    }
    for (int i = 0; i < inter; i++) g[i] *= u[i];
    for (int i = 0; i < hidden; i++) {
        float sum = 0;
        const float *dr = down + (size_t)i * inter;
        for (int j = 0; j < inter; j++) sum += dr[j] * g[j];
        output[i] = sum;
    }
    free(g); free(u);
}

// ---- SVD-approximate FFN at rank R ----
static void compute_ffn_svd(const float *Ug, const float *Sg, const float *VTg, int mn_g,
                            const float *Uu, const float *Su, const float *VTu, int mn_u,
                            const float *Ud, const float *Sd, const float *VTd, int mn_d,
                            int hidden, int inter, int R,
                            const float *h, float *output) {
    float *g = (float *)malloc(inter * sizeof(float));
    float *u = (float *)malloc(inter * sizeof(float));

    svd_matvec(Ug, Sg, VTg, inter, hidden, mn_g, R, h, g);
    for (int i = 0; i < inter; i++) g[i] = g[i] / (1.0f + expf(-g[i]));
    svd_matvec(Uu, Su, VTu, inter, hidden, mn_u, R, h, u);
    for (int i = 0; i < inter; i++) g[i] *= u[i];
    svd_matvec(Ud, Sd, VTd, hidden, inter, mn_d, R, g, output);

    free(g); free(u);
}

// ---- Utilities ----
static float cosine_sim(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i];
    }
    return (float)(dot / (sqrt(na)*sqrt(nb) + 1e-15));
}

static float rel_error(const float *real, const float *approx, int n) {
    double err = 0, norm = 0;
    for (int i = 0; i < n; i++) {
        double d = real[i] - approx[i]; err += d*d; norm += (double)real[i]*real[i];
    }
    return (float)(sqrt(err) / (sqrt(norm) + 1e-15));
}

static void rand_vector(float *v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) { v[i] = (float)drand48()*2-1; norm += v[i]*v[i]; }
    float s = sqrtf((float)n) / sqrtf(norm);  // norm ≈ sqrt(n)
    for (int i = 0; i < n; i++) v[i] *= s;
}

static const char *dtype_name(int t) {
    switch (t) {
    case GGML_TYPE_Q4_K: return "Q4_K"; case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K"; case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_F32:  return "F32";   default: return "???";
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---- Main ----
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <store.qmoe> [--hidden N] [--intermediate N]\n"
            "       [--experts N] [--layers N] [--inputs N]\n\n"
            "Defaults: hidden=3072, intermediate=1024, experts=5, layers=5, inputs=10\n",
            argv[0]);
        return 1;
    }

    const char *store_path = argv[1];
    int hidden = 3072, inter = 1024;
    int n_samp_exp = 5, n_samp_lay = 5, n_inputs = 10;

    for (int i = 2; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--hidden"))       hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--intermediate")) inter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts"))  n_samp_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers"))   n_samp_lay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inputs"))   n_inputs = atoi(argv[++i]);
    }

    // Open store for header/index
    expert_store_t *store = expert_store_open(store_path);
    if (!store) return 1;

    // Reopen without O_DIRECT for easy unaligned reads
    int fd = open(store_path, O_RDONLY);
    if (fd < 0) { perror("open"); expert_store_close(store); return 1; }

    int n_layers  = store->header.n_moe_layers;
    int n_experts = store->header.n_experts;
    int gate_type = store->header.quant_type;
    int up_type   = gate_type;

    // Verify dimensions match stored sizes
    size_t computed_gate = gguf_tensor_nbytes(gate_type, (int64_t)inter * hidden);
    if (computed_gate != store->header.expert_gate_size) {
        fprintf(stderr, "WARNING: computed gate_size=%zu != stored=%lu\n"
                "  Try --hidden and --intermediate to match your model\n",
                computed_gate, (unsigned long)store->header.expert_gate_size);
        close(fd); expert_store_close(store); return 1;
    }

    printf("=== SVD Experiment: Expert Weight Matrix Analysis ===\n\n");
    printf("Config: hidden=%d, intermediate=%d\n", hidden, inter);
    printf("Store:  %s (%d layers, %d experts, stride=%lu)\n",
           store_path, n_layers, n_experts,
           (unsigned long)store->header.expert_stride);
    printf("Quant:  gate=%s, up=%s\n", dtype_name(gate_type), dtype_name(up_type));
    printf("Sample: %d layers x %d experts x %d test inputs\n\n",
           n_samp_lay, n_samp_exp, n_inputs);

    srand48(42);

    // Ranks to test
    int ranks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_ranks = sizeof(ranks)/sizeof(ranks[0]);
    int mn_max = (hidden < inter) ? hidden : inter;
    while (n_ranks > 0 && ranks[n_ranks-1] > mn_max) n_ranks--;

    // Accumulators
    int total = 0;
    double eg_sum[9]={0}, eu_sum[9]={0}, ed_sum[9]={0};
    double eg_sq[9]={0},  eu_sq[9]={0},  ed_sq[9]={0};
    double cs_sum[9]={0}, cs_sq[9]={0};
    double re_sum[9]={0}, re_sq[9]={0};

    // Test inputs
    float **inputs = (float **)malloc(n_inputs * sizeof(float *));
    for (int t = 0; t < n_inputs; t++) {
        inputs[t] = (float *)malloc(hidden * sizeof(float));
        rand_vector(inputs[t], hidden);
    }
    float *ffn_real  = (float *)malloc(hidden * sizeof(float));
    float *ffn_approx = (float *)malloc(hidden * sizeof(float));

    double t_start = now_sec();

    for (int li = 0; li < n_samp_lay; li++) {
        int layer = (n_samp_lay > 1) ? li * (n_layers-1) / (n_samp_lay-1) : 0;
        if (layer >= n_layers) layer = n_layers - 1;

        int down_type = store->layer_index[layer].down_type;
        if (down_type == 0) down_type = gate_type;

        uint64_t stride = store->layer_index[layer].expert_stride;
        if (stride == 0) stride = store->header.expert_stride;

        printf("--- Layer %d (down=%s) ---\n", layer, dtype_name(down_type));

        for (int ei = 0; ei < n_samp_exp; ei++) {
            int eid = (int)(drand48() * n_experts);
            if (eid >= n_experts) eid = n_experts - 1;

            uint64_t offset = expert_store_offset(store, layer, eid);
            void *ebuf = malloc(stride);
            if (!ebuf) { fprintf(stderr, "OOM\n"); goto done; }

            ssize_t nr = pread(fd, ebuf, stride, offset);
            if (nr < (ssize_t)stride) {
                fprintf(stderr, "  Expert %3d: read failed\n", eid);
                free(ebuf); continue;
            }

            uint8_t *ep = (uint8_t *)ebuf;
            void *gate_data = ep;
            void *up_data   = ep + store->header.expert_gate_size;
            void *down_data = ep + store->header.expert_gate_size
                                 + store->header.expert_up_size;

            double t0 = now_sec();
            float *gf = dequantize_weight(gate_data, gate_type, inter, hidden);
            float *uf = dequantize_weight(up_data,   up_type,   inter, hidden);
            float *df = dequantize_weight(down_data,  down_type, hidden, inter);
            if (!gf || !uf || !df) {
                free(gf); free(uf); free(df); free(ebuf); continue;
            }
            double t_dq = now_sec() - t0;

            int mn_gu = (inter < hidden) ? inter : hidden;
            int mn_d  = (hidden < inter) ? hidden : inter;

            float *Ug=NULL, *Sg=NULL, *VTg=NULL;
            float *Uu=NULL, *Su=NULL, *VTu=NULL;
            float *Ud=NULL, *Sd=NULL, *VTd=NULL;
            Sg = (float *)malloc(mn_gu * sizeof(float));
            Su = (float *)malloc(mn_gu * sizeof(float));
            Sd = (float *)malloc(mn_d  * sizeof(float));

            printf("  Expert %3d: dequant %.1fms, SVD...", eid, t_dq*1000);
            fflush(stdout);
            t0 = now_sec();

            if (compute_svd(gf, inter, hidden, &Ug, Sg, &VTg) ||
                compute_svd(uf, inter, hidden, &Uu, Su, &VTu) ||
                compute_svd(df, hidden, inter, &Ud, Sd, &VTd)) {
                printf(" FAILED\n");
                goto next;
            }
            double t_svd = now_sec() - t0;
            printf(" %.1fs\n", t_svd);

            // Energy spectrum
            double tot_g=0, tot_u=0, tot_d=0;
            for (int i = 0; i < mn_gu; i++) {
                tot_g += (double)Sg[i]*Sg[i]; tot_u += (double)Su[i]*Su[i];
            }
            for (int i = 0; i < mn_d; i++) tot_d += (double)Sd[i]*Sd[i];

            // Print top-5 singular values
            printf("    Top-5 σ: gate=[");
            for (int i = 0; i < 5 && i < mn_gu; i++) printf("%.1f%s", Sg[i], i<4?",":"");
            printf("]  up=[");
            for (int i = 0; i < 5 && i < mn_gu; i++) printf("%.1f%s", Su[i], i<4?",":"");
            printf("]  down=[");
            for (int i = 0; i < 5 && i < mn_d; i++) printf("%.1f%s", Sd[i], i<4?",":"");
            printf("]\n");

            printf("    Energy:  ");
            for (int ri = 0; ri < n_ranks; ri++) {
                int R = ranks[ri];
                double eg=0, eu=0, ed=0;
                for (int i = 0; i < R && i < mn_gu; i++) {
                    eg += (double)Sg[i]*Sg[i]; eu += (double)Su[i]*Su[i];
                }
                for (int i = 0; i < R && i < mn_d; i++) ed += (double)Sd[i]*Sd[i];

                double pg = 100*eg/tot_g, pu = 100*eu/tot_u, pd = 100*ed/tot_d;
                if (ri == 0) printf("\n    %8s %8s %8s %8s\n", "Rank", "gate", "up", "down");
                printf("    %8d %7.1f%% %7.1f%% %7.1f%%\n", R, pg, pu, pd);

                eg_sum[ri] += pg; eg_sq[ri] += pg*pg;
                eu_sum[ri] += pu; eu_sq[ri] += pu*pu;
                ed_sum[ri] += pd; ed_sq[ri] += pd*pd;
            }

            // FFN accuracy
            printf("    FFN accuracy:\n");
            printf("    %8s %10s %10s\n", "Rank", "cos_sim", "rel_err");
            for (int ri = 0; ri < n_ranks; ri++) {
                int R = ranks[ri];
                double cs_s=0, re_s=0;
                for (int t = 0; t < n_inputs; t++) {
                    compute_ffn(gf, uf, df, hidden, inter, inputs[t], ffn_real);
                    compute_ffn_svd(Ug, Sg, VTg, mn_gu,
                                   Uu, Su, VTu, mn_gu,
                                   Ud, Sd, VTd, mn_d,
                                   hidden, inter, R, inputs[t], ffn_approx);
                    cs_s += cosine_sim(ffn_real, ffn_approx, hidden);
                    re_s += rel_error(ffn_real, ffn_approx, hidden);
                }
                double avg_cs = cs_s / n_inputs, avg_re = re_s / n_inputs;
                printf("    %8d %10.6f %10.6f\n", R, avg_cs, avg_re);
                cs_sum[ri] += avg_cs; cs_sq[ri] += avg_cs*avg_cs;
                re_sum[ri] += avg_re; re_sq[ri] += avg_re*avg_re;
            }

            total++;
next:
            free(Ug); free(Sg); free(VTg);
            free(Uu); free(Su); free(VTu);
            free(Ud); free(Sd); free(VTd);
            free(gf); free(uf); free(df);
            free(ebuf);
        }
    }

done:
    printf("\n=== Aggregate (%d experts, %.1fs) ===\n\n", total, now_sec()-t_start);

    if (total > 0) {
        printf("Energy Spectrum (mean +/- std):\n");
        printf("%6s  %14s  %14s  %14s\n", "Rank", "gate", "up", "down");
        for (int ri = 0; ri < n_ranks; ri++) {
            double mg = eg_sum[ri]/total, sg = sqrt(fabs(eg_sq[ri]/total - mg*mg));
            double mu = eu_sum[ri]/total, su = sqrt(fabs(eu_sq[ri]/total - mu*mu));
            double md = ed_sum[ri]/total, sd = sqrt(fabs(ed_sq[ri]/total - md*md));
            printf("%6d  %5.1f%%+/-%-4.1f  %5.1f%%+/-%-4.1f  %5.1f%%+/-%-4.1f\n",
                   ranks[ri], mg, sg, mu, su, md, sd);
        }

        printf("\nFFN Approximation (mean +/- std):\n");
        printf("%6s  %16s  %16s\n", "Rank", "cos_sim", "rel_err");
        for (int ri = 0; ri < n_ranks; ri++) {
            double mc = cs_sum[ri]/total, sc = sqrt(fabs(cs_sq[ri]/total - mc*mc));
            double mr = re_sum[ri]/total, sr = sqrt(fabs(re_sq[ri]/total - mr*mr));
            printf("%6d  %7.4f+/-%-6.4f  %7.4f+/-%-6.4f\n",
                   ranks[ri], mc, sc, mr, sr);
        }

        printf("\nSVD Storage Estimates (all %d layers x %d experts):\n",
               n_layers, n_experts);
        for (int ri = 0; ri < n_ranks; ri++) {
            int R = ranks[ri];
            // Per expert: 3 matrices, each needs U and VT columns/rows for rank R
            // gate/up: U[inter,R] + VT[R,hidden]  down: U[hidden,R] + VT[R,inter]
            // Plus S vectors (negligible)
            size_t per = (size_t)R * ((size_t)inter + hidden) * 2  // gate+up
                       + (size_t)R * ((size_t)hidden + inter)      // down
                       + (size_t)R * 3;
            size_t all = per * sizeof(float) * n_layers * n_experts;
            printf("  Rank %4d: %6.2f MB/expert, %7.1f GB total\n",
                   R, per*sizeof(float)/(1024.0*1024), all/(1024.0*1024*1024));
        }
    }

    for (int t = 0; t < n_inputs; t++) free(inputs[t]);
    free(inputs); free(ffn_real); free(ffn_approx);
    close(fd);
    expert_store_close(store);
    return 0;
}
