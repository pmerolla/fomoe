#ifndef QMOE_QUANT_H
#define QMOE_QUANT_H

#include <stdint.h>
#include <stddef.h>

// Q4_K quantization format constants
#define QK_K 256           // super-block size (values per block)
#define K_SCALE_SIZE 12    // bytes for scales/mins

// Q4_K block: 144 bytes encodes 256 values (~4.5 bits per weight)
typedef struct {
    uint16_t d;                    // super-block scale (FP16)
    uint16_t dmin;                 // super-block min scale (FP16)
    uint8_t  scales[K_SCALE_SIZE]; // sub-block scales and mins, 6-bit packed
    uint8_t  qs[QK_K / 2];        // 4-bit quantized values (nibbles)
} block_q4_K;

_Static_assert(sizeof(block_q4_K) == 2*sizeof(uint16_t) + K_SCALE_SIZE + QK_K/2,
               "wrong q4_K block size");

// FP16 <-> FP32 conversion
float    fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);

// Dequantize a row of Q4_K blocks to FP32
// k must be a multiple of QK_K (256)
void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);

// Q6_K block: 210 bytes encodes 256 values (~6.5 bits per weight)
typedef struct {
    uint8_t ql[QK_K / 2];      // quants, lower 4 bits
    uint8_t qh[QK_K / 4];      // quants, upper 2 bits
    int8_t  scales[QK_K / 16]; // scales, quantized with 8 bits
    uint16_t d;                 // super-block scale (FP16)
} block_q6_K;

_Static_assert(sizeof(block_q6_K) == sizeof(uint16_t) + QK_K/16 + 3*QK_K/4,
               "wrong q6_K block size");

// Quantized dot product: compute dot(x_q4k, y_f32) without full dequant
// n must be a multiple of QK_K
float vec_dot_q4_K_f32(const block_q4_K *x, const float *y, int64_t n);

// Dequantize Q6_K
void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);

// Quantized dot product for Q6_K
float vec_dot_q6_K_f32(const block_q6_K *x, const float *y, int64_t n);

// Q5_K block: 176 bytes encodes 256 values (~5.5 bits per weight)
typedef struct {
    uint16_t d;                    // super-block scale (FP16)
    uint16_t dmin;                 // super-block min scale (FP16)
    uint8_t  scales[K_SCALE_SIZE]; // sub-block scales and mins, 6-bit packed
    uint8_t  qh[QK_K / 8];        // high 5th bits (32 bytes)
    uint8_t  qs[QK_K / 2];        // low 4-bit quantized values (128 bytes)
} block_q5_K;

_Static_assert(sizeof(block_q5_K) == 2*sizeof(uint16_t) + K_SCALE_SIZE + QK_K/8 + QK_K/2,
               "wrong q5_K block size");

void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k);
float vec_dot_q5_K_f32(const block_q5_K *x, const float *y, int64_t n);

// Q8_0 block: 34 bytes encodes 32 values (~8.5 bits per weight)
#define QK8_0 32
typedef struct {
    uint16_t d;         // scale (FP16)
    int8_t   qs[QK8_0]; // quantized values
} block_q8_0;

_Static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + QK8_0,
               "wrong q8_0 block size");

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k);
float vec_dot_q8_0_f32(const block_q8_0 *x, const float *y, int64_t n);

// MXFP4 block: 17 bytes encodes 32 values (MX Floating Point 4-bit, E2M1)
#define QK_MXFP4 32
typedef struct {
    uint8_t s;                   // shared exponent (E8M0)
    uint8_t qs[QK_MXFP4 / 2];   // FP4 E2M1 values (16 bytes, 32 nibbles)
} block_mxfp4;

_Static_assert(sizeof(block_mxfp4) == 1 + QK_MXFP4/2,
               "wrong mxfp4 block size");

void dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k);
float vec_dot_mxfp4_f32(const block_mxfp4 *x, const float *y, int64_t n);

// Q8_K block: intermediate quantization for fast SIMD dot products
// Used to pre-quantize float input vectors before Q4_K/Q6_K dot products
typedef struct {
    float    d;              // block scale
    int8_t   qs[QK_K];      // quantized values [-127, 127]
    int16_t  bsums[QK_K/16]; // sum of qs in groups of 16 (for "mins" shortcut)
} block_q8_K;

// Quantize float vector to Q8_K (k must be multiple of QK_K)
void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);

// Fast dot products: Q4_K/Q6_K weights × Q8_K pre-quantized input
// Uses AVX2 SIMD when available, scalar fallback otherwise
// nb = number of super-blocks (n / QK_K)
float vec_dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, int nb);
float vec_dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, int nb);
float vec_dot_q5_K_q8_K(const block_q5_K *x, const block_q8_K *y, int nb);

// Fast mat-vec: quantizes input once, then uses SIMD dot products for all rows
// out[M] = mat[M × K] @ vec[K], where mat is Q4_K/Q6_K/Q5_K
void mat_vec_mul_fast(float *out, const void *mat, int mat_type,
                      const float *vec, int M, int K);

#endif // QMOE_QUANT_H
