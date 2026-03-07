#ifndef QMOE_TENSOR_H
#define QMOE_TENSOR_H

#include "gguf.h"
#include <stdint.h>
#include <stddef.h>

#define MAX_DIMS 4

// Tensor: holds either quantized or float data
typedef struct {
    enum ggml_dtype type;
    int             n_dims;
    int64_t         shape[MAX_DIMS];  // shape[0] = rows, shape[1] = cols, etc.
    int64_t         nelements;
    size_t          nbytes;
    void           *data;             // owned if allocated by us
    bool            owns_data;
} tensor_t;

// Allocate a tensor (allocates data buffer)
tensor_t *tensor_alloc(enum ggml_dtype type, int n_dims, const int64_t *shape);

// Create a tensor wrapping existing data (does not own)
tensor_t *tensor_wrap(enum ggml_dtype type, int n_dims, const int64_t *shape, void *data);

// Free tensor
void tensor_free(tensor_t *t);

// Dequantize tensor to FP32 (returns new tensor or self if already F32)
tensor_t *tensor_to_f32(const tensor_t *t);

// --- Math operations (all operate on FP32 tensors) ---

// Matrix-vector multiply: out[M] = mat[M,K] @ vec[K]
// mat can be quantized (Q4_K), vec must be F32, out must be F32
void mat_vec_mul(float *out, const void *mat, enum ggml_dtype mat_type,
                 const float *vec, int64_t M, int64_t K);

// RMS normalization: out[n] = x[n] / rms(x) * weight[n]
void rms_norm(float *out, const float *x, const float *weight, int64_t n, float eps);

// Softmax in-place
void softmax(float *x, int64_t n);

// SiLU activation in-place: x = x * sigmoid(x)
void silu(float *x, int64_t n);

// Element-wise multiply: out[i] = a[i] * b[i]
void vec_mul(float *out, const float *a, const float *b, int64_t n);

// Element-wise add: out[i] = a[i] + b[i]
void vec_add(float *out, const float *a, const float *b, int64_t n);

// Scaled add: out[i] += scale * x[i]
void vec_scaled_add(float *out, const float *x, float scale, int64_t n);

// Top-k selection: find top k indices from x[n], write to indices[k] and values[k]
void top_k(const float *x, int64_t n, int k, int *indices, float *values);

#endif // QMOE_TENSOR_H
