#include "tensor.h"
#include "quant.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

tensor_t *tensor_alloc(enum ggml_dtype type, int n_dims, const int64_t *shape) {
    tensor_t *t = calloc(1, sizeof(tensor_t));
    if (!t) return NULL;

    t->type = type;
    t->n_dims = n_dims;
    t->nelements = 1;
    for (int i = 0; i < n_dims; i++) {
        t->shape[i] = shape[i];
        t->nelements *= shape[i];
    }
    for (int i = n_dims; i < MAX_DIMS; i++) {
        t->shape[i] = 1;
    }

    t->nbytes = gguf_tensor_nbytes(type, t->nelements);
    if (t->nbytes == 0) { free(t); return NULL; }

    t->data = calloc(1, t->nbytes);
    if (!t->data) { free(t); return NULL; }
    t->owns_data = true;

    return t;
}

tensor_t *tensor_wrap(enum ggml_dtype type, int n_dims, const int64_t *shape, void *data) {
    tensor_t *t = calloc(1, sizeof(tensor_t));
    if (!t) return NULL;

    t->type = type;
    t->n_dims = n_dims;
    t->nelements = 1;
    for (int i = 0; i < n_dims; i++) {
        t->shape[i] = shape[i];
        t->nelements *= shape[i];
    }
    for (int i = n_dims; i < MAX_DIMS; i++) {
        t->shape[i] = 1;
    }

    t->nbytes = gguf_tensor_nbytes(type, t->nelements);
    t->data = data;
    t->owns_data = false;

    return t;
}

void tensor_free(tensor_t *t) {
    if (!t) return;
    if (t->owns_data) free(t->data);
    free(t);
}

tensor_t *tensor_to_f32(const tensor_t *t) {
    if (t->type == GGML_TYPE_F32) return (tensor_t *)t;  // already f32

    int64_t shape_f32[] = { t->shape[0], t->shape[1], t->shape[2], t->shape[3] };
    tensor_t *out = tensor_alloc(GGML_TYPE_F32, t->n_dims, shape_f32);
    if (!out) return NULL;

    if (t->type == GGML_TYPE_Q4_K) {
        dequantize_row_q4_K((const block_q4_K *)t->data, (float *)out->data, t->nelements);
    } else if (t->type == GGML_TYPE_Q6_K) {
        dequantize_row_q6_K((const block_q6_K *)t->data, (float *)out->data, t->nelements);
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        float *dst = (float *)out->data;
        for (int64_t i = 0; i < t->nelements; i++) {
            dst[i] = fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const uint16_t *src = (const uint16_t *)t->data;
        float *dst = (float *)out->data;
        for (int64_t i = 0; i < t->nelements; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&dst[i], &bits, sizeof(float));
        }
    } else if (t->type == GGML_TYPE_Q5_K) {
        dequantize_row_q5_K((const block_q5_K *)t->data, (float *)out->data, t->nelements);
    } else if (t->type == GGML_TYPE_Q8_0) {
        dequantize_row_q8_0((const block_q8_0 *)t->data, (float *)out->data, t->nelements);
    } else if (t->type == GGML_TYPE_MXFP4) {
        dequantize_row_mxfp4((const block_mxfp4 *)t->data, (float *)out->data, t->nelements);
    } else {
        // Unsupported type for dequant
        tensor_free(out);
        return NULL;
    }

    return out;
}

// ---- Math operations ----

void mat_vec_mul(float *out, const void *mat, enum ggml_dtype mat_type,
                 const float *vec, int64_t M, int64_t K) {
    if (mat_type == GGML_TYPE_F32) {
        const float *mf = (const float *)mat;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < K; j++) {
                sum += mf[i * K + j] * vec[j];
            }
            out[i] = sum;
        }
    } else if (mat_type == GGML_TYPE_Q4_K) {
        const block_q4_K *mq = (const block_q4_K *)mat;
        int64_t blocks_per_row = K / QK_K;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            out[i] = vec_dot_q4_K_f32(mq + i * blocks_per_row, vec, K);
        }
    } else if (mat_type == GGML_TYPE_Q6_K) {
        const block_q6_K *mq = (const block_q6_K *)mat;
        int64_t blocks_per_row = K / QK_K;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            out[i] = vec_dot_q6_K_f32(mq + i * blocks_per_row, vec, K);
        }
    } else if (mat_type == GGML_TYPE_F16) {
        const uint16_t *mh = (const uint16_t *)mat;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < K; j++) {
                sum += fp16_to_fp32(mh[i * K + j]) * vec[j];
            }
            out[i] = sum;
        }
    } else if (mat_type == GGML_TYPE_BF16) {
        const uint16_t *mb = (const uint16_t *)mat;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < K; j++) {
                uint32_t bits = (uint32_t)mb[i * K + j] << 16;
                float val;
                memcpy(&val, &bits, sizeof(float));
                sum += val * vec[j];
            }
            out[i] = sum;
        }
    } else if (mat_type == GGML_TYPE_Q5_K) {
        const block_q5_K *mq = (const block_q5_K *)mat;
        int64_t blocks_per_row = K / QK_K;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            out[i] = vec_dot_q5_K_f32(mq + i * blocks_per_row, vec, K);
        }
    } else if (mat_type == GGML_TYPE_Q8_0) {
        const block_q8_0 *mq = (const block_q8_0 *)mat;
        int64_t blocks_per_row = K / QK8_0;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            out[i] = vec_dot_q8_0_f32(mq + i * blocks_per_row, vec, K);
        }
    } else if (mat_type == GGML_TYPE_MXFP4) {
        const block_mxfp4 *mq = (const block_mxfp4 *)mat;
        int64_t blocks_per_row = K / QK_MXFP4;
        #pragma omp parallel for if(M > 64)
        for (int64_t i = 0; i < M; i++) {
            out[i] = vec_dot_mxfp4_f32(mq + i * blocks_per_row, vec, K);
        }
    }
}

void rms_norm(float *out, const float *x, const float *weight, int64_t n, float eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int64_t i = 0; i < n; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void softmax(float *x, int64_t n) {
    float max_val = x[0];
    for (int64_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void silu(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void vec_mul(float *out, const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void vec_add(float *out, const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scaled_add(float *out, const float *x, float scale, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] += scale * x[i];
    }
}

void top_k(const float *x, int64_t n, int k, int *indices, float *values) {
    // Simple selection sort for small k
    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -1e30f;
    }
    for (int64_t j = 0; j < n; j++) {
        // Find the position in top-k
        int pos = k;
        for (int i = k - 1; i >= 0; i--) {
            if (x[j] > values[i]) {
                pos = i;
            } else {
                break;
            }
        }
        if (pos < k) {
            // Shift down
            for (int i = k - 1; i > pos; i--) {
                indices[i] = indices[i - 1];
                values[i] = values[i - 1];
            }
            indices[pos] = (int)j;
            values[pos] = x[j];
        }
    }
}
