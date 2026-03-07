#include "cpu_expert.h"
#include "quant.h"
#include "gguf.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <stdio.h>

#define CPU_EXPERT_MAX_WORK 16

struct cpu_expert_ctx {
    int n_embd;
    int expert_intermediate;

    // Scratch buffers for expert FFN computation
    float *gate_buf;   // [expert_intermediate]
    float *up_buf;     // [expert_intermediate]
    float *down_buf;   // [n_embd]

    // Accumulated partial result (weighted sum of expert outputs)
    float *partial;    // [n_embd]

    // Pre-quantized input (reused across experts in same layer)
    block_q8_K *q8_input;   // [n_embd / QK_K] blocks
    int q8_input_valid;      // whether q8_input has been quantized for current layer

    // Async worker thread
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t cond_work;   // main → worker: new work available
    pthread_cond_t cond_done;   // worker → main: work completed
    const float *work_input;
    cpu_expert_work_t work_items[CPU_EXPERT_MAX_WORK];
    int work_count;
    int busy;       // 1 = worker is computing
    int quit;       // 1 = shutdown requested
    int has_thread; // 1 = worker thread was created

    // Internal timing (updated by worker, read after wait)
    double last_wakeup_us;   // condvar wakeup latency
    double last_compute_us;  // actual FFN compute time
    double last_signal_us;   // condvar signal time
    int timing_samples;
    double total_wakeup_us;
    double total_compute_us;
};

// Internal: fast matvec using pre-quantized Q8_K input
static void matvec_fast_q8(float *out, const void *mat, int mat_type,
                           const block_q8_K *q8, int M, int K) {
    const int nb = K / QK_K;

    if (mat_type == GGML_TYPE_Q4_K) {
        const block_q4_K *mq = (const block_q4_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q4_K_q8_K(mq + i * nb, q8, nb);
    } else if (mat_type == GGML_TYPE_Q6_K) {
        const block_q6_K *mq = (const block_q6_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q6_K_q8_K(mq + i * nb, q8, nb);
    } else if (mat_type == GGML_TYPE_Q5_K) {
        const block_q5_K *mq = (const block_q5_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q5_K_q8_K(mq + i * nb, q8, nb);
    }
}

// Internal: compute one expert FFN and accumulate (called from worker or main thread)
static void compute_one_expert(cpu_expert_ctx_t *ctx,
                               const void *expert_data,
                               const float *input,
                               float score,
                               int gate_type, int up_type, int down_type,
                               uint64_t expert_gate_size, uint64_t expert_up_size) {
    const int inter = ctx->expert_intermediate;
    const int n_embd = ctx->n_embd;

    // Parse expert data layout: gate | up | down (contiguous)
    const uint8_t *edata = (const uint8_t *)expert_data;
    const void *gate_w = edata;
    const void *up_w   = edata + expert_gate_size;
    const void *down_w = edata + expert_gate_size + expert_up_size;

    // Quantize input to Q8_K once per layer (reused across all experts)
    if (!ctx->q8_input_valid) {
        quantize_row_q8_K(input, ctx->q8_input, n_embd);
        ctx->q8_input_valid = 1;
    }

    // Gate projection: [inter x n_embd] @ [n_embd] -> [inter]
    matvec_fast_q8(ctx->gate_buf, gate_w, gate_type, ctx->q8_input, inter, n_embd);

    // Up projection: [inter x n_embd] @ [n_embd] -> [inter]
    matvec_fast_q8(ctx->up_buf, up_w, up_type, ctx->q8_input, inter, n_embd);

    // SiLU(gate) * up
    for (int i = 0; i < inter; i++) {
        float g = ctx->gate_buf[i];
        ctx->gate_buf[i] = (g / (1.0f + expf(-g))) * ctx->up_buf[i];
    }

    // Down projection: [n_embd x inter] @ [inter] -> [n_embd]
    block_q8_K q8_gate[inter / QK_K];  // 4 blocks for inter=1024
    quantize_row_q8_K(ctx->gate_buf, q8_gate, inter);
    matvec_fast_q8(ctx->down_buf, down_w, down_type, q8_gate, n_embd, inter);

    // Accumulate weighted result
    for (int i = 0; i < n_embd; i++)
        ctx->partial[i] += score * ctx->down_buf[i];
}

static inline double elapsed_us(struct timespec *a, struct timespec *b) {
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

// Worker thread: waits for work, computes all experts, signals done
static void *cpu_expert_worker_fn(void *arg) {
    cpu_expert_ctx_t *ctx = (cpu_expert_ctx_t *)arg;
    struct timespec t0, t1, t2;

    pthread_mutex_lock(&ctx->mtx);
    while (!ctx->quit) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        while (!ctx->busy && !ctx->quit)
            pthread_cond_wait(&ctx->cond_work, &ctx->mtx);
        if (ctx->quit) break;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        int n = ctx->work_count;
        const float *input = ctx->work_input;

        pthread_mutex_unlock(&ctx->mtx);

        // Reset state for this batch
        memset(ctx->partial, 0, ctx->n_embd * sizeof(float));
        ctx->q8_input_valid = 0;

        // Compute all experts
        for (int i = 0; i < n; i++) {
            const cpu_expert_work_t *w = &ctx->work_items[i];
            compute_one_expert(ctx, w->expert_data, input, w->score,
                              w->gate_type, w->up_type, w->down_type,
                              w->expert_gate_size, w->expert_up_size);
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);

        ctx->last_wakeup_us = elapsed_us(&t0, &t1);
        ctx->last_compute_us = elapsed_us(&t1, &t2);
        ctx->total_wakeup_us += ctx->last_wakeup_us;
        ctx->total_compute_us += ctx->last_compute_us;
        ctx->timing_samples++;

        pthread_mutex_lock(&ctx->mtx);
        ctx->busy = 0;
        pthread_cond_signal(&ctx->cond_done);
    }
    pthread_mutex_unlock(&ctx->mtx);
    return NULL;
}

cpu_expert_ctx_t *cpu_expert_create(int n_embd, int expert_intermediate) {
    cpu_expert_ctx_t *ctx = calloc(1, sizeof(cpu_expert_ctx_t));
    if (!ctx) return NULL;

    ctx->n_embd = n_embd;
    ctx->expert_intermediate = expert_intermediate;

    ctx->gate_buf = calloc(expert_intermediate, sizeof(float));
    ctx->up_buf   = calloc(expert_intermediate, sizeof(float));
    ctx->down_buf = calloc(n_embd, sizeof(float));
    ctx->partial  = calloc(n_embd, sizeof(float));
    ctx->q8_input = calloc(n_embd / QK_K, sizeof(block_q8_K));

    if (!ctx->gate_buf || !ctx->up_buf || !ctx->down_buf ||
        !ctx->partial || !ctx->q8_input) {
        cpu_expert_free(ctx);
        return NULL;
    }

    // Start worker thread
    pthread_mutex_init(&ctx->mtx, NULL);
    pthread_cond_init(&ctx->cond_work, NULL);
    pthread_cond_init(&ctx->cond_done, NULL);
    ctx->busy = 0;
    ctx->quit = 0;
    ctx->has_thread = 0;

    if (pthread_create(&ctx->thread, NULL, cpu_expert_worker_fn, ctx) == 0) {
        ctx->has_thread = 1;
    }

    return ctx;
}

void cpu_expert_free(cpu_expert_ctx_t *ctx) {
    if (!ctx) return;

    // Shut down worker thread
    if (ctx->has_thread) {
        pthread_mutex_lock(&ctx->mtx);
        ctx->quit = 1;
        pthread_cond_signal(&ctx->cond_work);
        pthread_mutex_unlock(&ctx->mtx);
        pthread_join(ctx->thread, NULL);
    }

    pthread_mutex_destroy(&ctx->mtx);
    pthread_cond_destroy(&ctx->cond_work);
    pthread_cond_destroy(&ctx->cond_done);

    free(ctx->gate_buf);
    free(ctx->up_buf);
    free(ctx->down_buf);
    free(ctx->partial);
    free(ctx->q8_input);
    free(ctx);
}

void cpu_expert_reset(cpu_expert_ctx_t *ctx) {
    memset(ctx->partial, 0, ctx->n_embd * sizeof(float));
    ctx->q8_input_valid = 0;
}

void cpu_expert_ffn(cpu_expert_ctx_t *ctx,
                    const void *expert_data,
                    const float *input,
                    float score,
                    int gate_type, int up_type, int down_type,
                    uint64_t expert_gate_size, uint64_t expert_up_size) {
    compute_one_expert(ctx, expert_data, input, score,
                      gate_type, up_type, down_type,
                      expert_gate_size, expert_up_size);
}

void cpu_expert_submit_async(cpu_expert_ctx_t *ctx, const float *input,
                             const cpu_expert_work_t *work, int n_work) {
    if (n_work <= 0) return;
    if (n_work > CPU_EXPERT_MAX_WORK) n_work = CPU_EXPERT_MAX_WORK;

    // Fallback to synchronous if no worker thread
    if (!ctx->has_thread) {
        cpu_expert_reset(ctx);
        for (int i = 0; i < n_work; i++) {
            const cpu_expert_work_t *w = &work[i];
            compute_one_expert(ctx, w->expert_data, input, w->score,
                              w->gate_type, w->up_type, w->down_type,
                              w->expert_gate_size, w->expert_up_size);
        }
        return;
    }

    pthread_mutex_lock(&ctx->mtx);
    // Wait for any previous work to finish (shouldn't happen in practice)
    while (ctx->busy)
        pthread_cond_wait(&ctx->cond_done, &ctx->mtx);

    ctx->work_input = input;
    memcpy(ctx->work_items, work, n_work * sizeof(cpu_expert_work_t));
    ctx->work_count = n_work;
    ctx->busy = 1;
    pthread_cond_signal(&ctx->cond_work);
    pthread_mutex_unlock(&ctx->mtx);
}

void cpu_expert_wait(cpu_expert_ctx_t *ctx) {
    if (!ctx->has_thread) return;  // sync fallback already finished

    pthread_mutex_lock(&ctx->mtx);
    while (ctx->busy)
        pthread_cond_wait(&ctx->cond_done, &ctx->mtx);
    pthread_mutex_unlock(&ctx->mtx);
}

float *cpu_expert_result(cpu_expert_ctx_t *ctx) {
    return ctx->partial;
}

void cpu_expert_get_timing(cpu_expert_ctx_t *ctx, double *avg_wakeup_us,
                           double *avg_compute_us, int *samples) {
    *samples = ctx->timing_samples;
    if (ctx->timing_samples > 0) {
        *avg_wakeup_us = ctx->total_wakeup_us / ctx->timing_samples;
        *avg_compute_us = ctx->total_compute_us / ctx->timing_samples;
    } else {
        *avg_wakeup_us = 0;
        *avg_compute_us = 0;
    }
}
