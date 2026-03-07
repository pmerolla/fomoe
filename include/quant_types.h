/*
 * quant_types.h — Quantization block types for GPU kernels
 *
 * Shared between CPU and GPU code. GPU kernels operate directly
 * on quantized blocks without full dequantization.
 */

#ifndef FR_QUANT_TYPES_H
#define FR_QUANT_TYPES_H

#include <stdint.h>

// Q4_K: 144 bytes encodes 256 values (~4.5 bits/weight)
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;                    // super-block scale (FP16)
    uint16_t dmin;                 // super-block min scale (FP16)
    uint8_t  scales[K_SCALE_SIZE]; // sub-block scales and mins, 6-bit packed
    uint8_t  qs[QK_K / 2];        // 4-bit quantized values (nibbles)
} block_q4_K;

// Q5_K: 176 bytes encodes 256 values (~5.5 bits/weight)
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];        // high 5th bits
    uint8_t  qs[QK_K / 2];        // low 4-bit values
} block_q5_K;

// Q6_K: 210 bytes encodes 256 values (~6.5 bits/weight)
typedef struct {
    uint8_t  ql[QK_K / 2];        // quants, lower 4 bits
    uint8_t  qh[QK_K / 4];        // quants, upper 2 bits
    int8_t   scales[QK_K / 16];   // scales, 8-bit
    uint16_t d;                    // super-block scale (FP16)
} block_q6_K;

// Q8_0: 34 bytes encodes 32 values (~8.5 bits/weight)
#define QK8_0 32
typedef struct {
    uint16_t d;          // scale (FP16)
    int8_t   qs[QK8_0];  // quantized values
} block_q8_0;

// MXFP4: 17 bytes encodes 32 values (MX Floating Point 4-bit, E2M1)
#define QK_MXFP4 32
typedef struct {
    uint8_t s;                   // shared exponent (E8M0)
    uint8_t qs[QK_MXFP4 / 2];   // FP4 E2M1 values
} block_mxfp4;

// GGML dtype IDs (matching gguf format)
enum ggml_dtype {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_MXFP4 = 39,
};

// Block sizes in bytes for each quant type
static inline uint64_t ggml_type_block_size(enum ggml_dtype t) {
    switch (t) {
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_MXFP4: return sizeof(block_mxfp4);
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_F32: return 4;
        default: return 0;
    }
}

// Elements per block
static inline int ggml_type_block_elements(enum ggml_dtype t) {
    switch (t) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return QK_K;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_MXFP4: return QK_MXFP4;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_F32: return 1;
        default: return 0;
    }
}

// Bytes per row of K elements
static inline uint64_t ggml_row_bytes(enum ggml_dtype t, int K) {
    int elts = ggml_type_block_elements(t);
    if (elts == 0) return 0;
    return (uint64_t)(K / elts) * ggml_type_block_size(t);
}

#endif // FR_QUANT_TYPES_H
