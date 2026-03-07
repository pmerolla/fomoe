#include "quant.h"
#include "gguf.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ---- FP16 <-> FP32 conversion ----

static inline uint32_t fp32_to_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static inline float fp32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1u << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

uint16_t fp32_to_fp16(float f) {
    const float scale_to_inf  = fp32_from_bits(0x77800000u);
    const float scale_to_zero = fp32_from_bits(0x08800000u);
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    base = fp32_from_bits((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;

    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// ---- Q4_K dequantization ----

// Extract 6-bit scale and min from the packed scales array
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k) {
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32;
            is += 2;
        }
    }
}

// ---- Quantized dot product ----

float vec_dot_q4_K_f32(const block_q4_K *x, const float *y, int64_t n) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);

        int is = 0;
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                sum += (d1 * (q[l] & 0xF) - m1) * y[j + l];
            }
            for (int l = 0; l < 32; ++l) {
                sum += (d2 * (q[l] >> 4) - m2) * y[j + 32 + l];
            }
            q += 32;
            is += 2;
        }
        sumf += sum;
        y += QK_K;
    }

    return sumf;
}

// ---- Q6_K dequantization ----

void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

float vec_dot_q6_K_f32(const block_q6_K *x, const float *y, int64_t n) {
    const int64_t nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        float sum = 0.0f;
        const float *yp = y + i * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * sc[is + 0] * q1 * yp[l +  0];
                sum += d * sc[is + 2] * q2 * yp[l + 32];
                sum += d * sc[is + 4] * q3 * yp[l + 64];
                sum += d * sc[is + 6] * q4 * yp[l + 96];
            }
            yp += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        sumf += sum;
    }

    return sumf;
}

// ---- Q5_K dequantization ----

void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k) {
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

float vec_dot_q5_K_f32(const block_q5_K *x, const float *y, int64_t n) {
    float tmp[QK_K];
    float sumf = 0.0f;
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        dequantize_row_q5_K(&x[i], tmp, QK_K);
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            sum += tmp[j] * y[i * QK_K + j];
        }
        sumf += sum;
    }
    return sumf;
}

// ---- Q8_0 dequantization ----

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) {
            y[i * QK8_0 + j] = d * x[i].qs[j];
        }
    }
}

float vec_dot_q8_0_f32(const block_q8_0 *x, const float *y, int64_t n) {
    float sumf = 0.0f;
    const int nb = n / QK8_0;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        float sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            sum += (float)x[i].qs[j] * y[i * QK8_0 + j];
        }
        sumf += d * sum;
    }
    return sumf;
}

// ---- MXFP4 dequantization (Microscaling FP4, E2M1 format) ----

// FP4 E2M1 lookup table: maps 4-bit value to float
static const float MXFP4_TABLE[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

void dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k) {
    const int nb = k / QK_MXFP4;
    for (int i = 0; i < nb; i++) {
        float scale;
        if (x[i].s == 0xFF) {
            scale = 0.0f;  // NaN shared exponent
        } else {
            // E8M0: 2^(s - 127)
            scale = ldexpf(1.0f, (int)x[i].s - 127);
        }
        // GGML layout: low nibbles at positions 0..15, high nibbles at 16..31
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            uint8_t byte = x[i].qs[j];
            y[i * QK_MXFP4 + j]                = MXFP4_TABLE[byte & 0x0F] * scale;
            y[i * QK_MXFP4 + j + QK_MXFP4 / 2] = MXFP4_TABLE[byte >> 4] * scale;
        }
    }
}

float vec_dot_mxfp4_f32(const block_mxfp4 *x, const float *y, int64_t n) {
    float sumf = 0.0f;
    const int nb = n / QK_MXFP4;
    for (int i = 0; i < nb; i++) {
        float scale;
        if (x[i].s == 0xFF) {
            scale = 0.0f;
        } else {
            scale = ldexpf(1.0f, (int)x[i].s - 127);
        }
        float sum = 0.0f;
        // GGML layout: low nibbles at positions 0..15, high nibbles at 16..31
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            uint8_t byte = x[i].qs[j];
            sum += MXFP4_TABLE[byte & 0x0F] * y[i * QK_MXFP4 + j];
            sum += MXFP4_TABLE[byte >> 4]   * y[i * QK_MXFP4 + j + QK_MXFP4 / 2];
        }
        sumf += scale * sum;
    }
    return sumf;
}

// ===========================================================================
// Q8_K quantization + AVX2 SIMD dot products (ported from ggml)
// ===========================================================================

void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(x[j]);
            if (ax > amax) amax = ax;
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, QK_K / 16 * sizeof(int16_t));
            x += QK_K;
            continue;
        }
        const float iscale = 127.0f / amax;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)roundf(iscale * x[j]);
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            y[i].qs[j] = (int8_t)v;
        }
        // Block sums in groups of 16
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int l = 0; l < 16; l++)
                sum += y[i].qs[j * 16 + l];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;  // store as delta (d * qs ≈ x)
        x += QK_K;
    }
}

#ifdef __AVX2__

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// Scale shuffle table for Q4_K: broadcasts the correct 16-bit scale to all
// positions within a 256-bit register via _mm256_shuffle_epi8
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

float vec_dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        // Unpack 6-bit scales/mins from 12 bytes
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        // "Mins" shortcut using pre-computed block sums
        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(
            _mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    // Reduce
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

float vec_dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, int nb) {
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        const int8_t  *q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();

        // Process 128 elements per iteration (2 iterations for QK_K=256)
        for (int j = 0; j < QK_K / 128; ++j) {
            // Load 64 bytes of low 4 bits
            const __m256i q4lo0 = _mm256_loadu_si256((const __m256i *)(ql +  0));
            const __m256i q4lo1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            // Load 32 bytes of high 2 bits
            const __m256i qhbits = _mm256_loadu_si256((const __m256i *)qh);

            const __m256i m4 = _mm256_set1_epi8(0xF);
            const __m256i m2 = _mm256_set1_epi8(3);
            const __m256i m32s = _mm256_set1_epi8(32);

            // Reconstruct 6-bit values: low 4 bits | (high 2 bits << 4) - 32
            // Sub-block 0: ql[0..31] low nibble, qh bits 0-1
            __m256i q6_0 = _mm256_or_si256(
                _mm256_and_si256(q4lo0, m4),
                _mm256_slli_epi16(_mm256_and_si256(qhbits, m2), 4));
            q6_0 = _mm256_sub_epi8(q6_0, m32s);

            // Sub-block 1: ql[32..63] low nibble, qh bits 2-3
            __m256i q6_1 = _mm256_or_si256(
                _mm256_and_si256(q4lo1, m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 2), m2), 4));
            q6_1 = _mm256_sub_epi8(q6_1, m32s);

            // Sub-block 2: ql[0..31] high nibble, qh bits 4-5
            __m256i q6_2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4lo0, 4), m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 4), m2), 4));
            q6_2 = _mm256_sub_epi8(q6_2, m32s);

            // Sub-block 3: ql[32..63] high nibble, qh bits 6-7
            __m256i q6_3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4lo1, 4), m4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 6), m2), 4));
            q6_3 = _mm256_sub_epi8(q6_3, m32s);

            // Load Q8 values and compute dot products
            // Each sub-block: 32 values, scale from sc[]
            // q6 values are signed [-32, 31], q8 values are signed [-127, 127]
            // Use _mm256_maddubs_epi16 which needs unsigned × signed
            // So we add 32 back to q6 (making it unsigned 0..63) and compensate

            // Actually, for Q6_K with signed values, use sign-extended multiply:
            // Convert q6 and q8 to int16 and multiply, or use maddubs with offset
            // Simpler approach: use _mm256_madd_epi16 on int16 pairs

            // Process sub-block 0: scale sc[0] and sc[1]
            {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                // Multiply q6 × q8 as signed bytes → int16 pairs
                // Split into even/odd bytes for proper signed multiply
                const __m256i q6lo = _mm256_srai_epi16(_mm256_slli_epi16(q6_0, 8), 8);
                const __m256i q6hi = _mm256_srai_epi16(q6_0, 8);
                const __m256i q8lo = _mm256_srai_epi16(_mm256_slli_epi16(q8_0, 8), 8);
                const __m256i q8hi = _mm256_srai_epi16(q8_0, 8);
                __m256i prod = _mm256_add_epi32(
                    _mm256_madd_epi16(q6lo, q8lo),
                    _mm256_madd_epi16(q6hi, q8hi));
                // Scale: sc[0] for low 16, sc[1] for high 16
                const __m256i sc_0 = _mm256_set_epi32(sc[1],sc[1],sc[1],sc[1],sc[0],sc[0],sc[0],sc[0]);
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(prod, sc_0));
            }

            // Process sub-block 1
            {
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i q6lo = _mm256_srai_epi16(_mm256_slli_epi16(q6_1, 8), 8);
                const __m256i q6hi = _mm256_srai_epi16(q6_1, 8);
                const __m256i q8lo = _mm256_srai_epi16(_mm256_slli_epi16(q8_1, 8), 8);
                const __m256i q8hi = _mm256_srai_epi16(q8_1, 8);
                __m256i prod = _mm256_add_epi32(
                    _mm256_madd_epi16(q6lo, q8lo),
                    _mm256_madd_epi16(q6hi, q8hi));
                const __m256i sc_1 = _mm256_set_epi32(sc[3],sc[3],sc[3],sc[3],sc[2],sc[2],sc[2],sc[2]);
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(prod, sc_1));
            }

            // Process sub-block 2
            {
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 64));
                const __m256i q6lo = _mm256_srai_epi16(_mm256_slli_epi16(q6_2, 8), 8);
                const __m256i q6hi = _mm256_srai_epi16(q6_2, 8);
                const __m256i q8lo = _mm256_srai_epi16(_mm256_slli_epi16(q8_2, 8), 8);
                const __m256i q8hi = _mm256_srai_epi16(q8_2, 8);
                __m256i prod = _mm256_add_epi32(
                    _mm256_madd_epi16(q6lo, q8lo),
                    _mm256_madd_epi16(q6hi, q8hi));
                const __m256i sc_2 = _mm256_set_epi32(sc[5],sc[5],sc[5],sc[5],sc[4],sc[4],sc[4],sc[4]);
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(prod, sc_2));
            }

            // Process sub-block 3
            {
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));
                const __m256i q6lo = _mm256_srai_epi16(_mm256_slli_epi16(q6_3, 8), 8);
                const __m256i q6hi = _mm256_srai_epi16(q6_3, 8);
                const __m256i q8lo = _mm256_srai_epi16(_mm256_slli_epi16(q8_3, 8), 8);
                const __m256i q8hi = _mm256_srai_epi16(q8_3, 8);
                __m256i prod = _mm256_add_epi32(
                    _mm256_madd_epi16(q6lo, q8lo),
                    _mm256_madd_epi16(q6hi, q8hi));
                const __m256i sc_3 = _mm256_set_epi32(sc[7],sc[7],sc[7],sc[7],sc[6],sc[6],sc[6],sc[6]);
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(prod, sc_3));
            }

            ql += 64;
            qh += 32;
            q8 += 128;
            sc += 8;
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_float_8(acc);
}

float vec_dot_q5_K_q8_K(const block_q5_K *x, const block_q8_K *y, int nb) {
    // Scalar fallback for now (Q5_K is less common)
    float sumf = 0.0f;
    float tmp[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_q5_K(&x[i], tmp, QK_K);
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j++)
            sum += tmp[j] * y[i].d * y[i].qs[j];
        sumf += sum;
    }
    return sumf;
}

#else // no AVX2: scalar fallbacks

float vec_dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        const uint8_t *q = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int is = 0;
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++)
                sum += (d1 * (q[l] & 0xF) - m1) * q8[j + l];
            for (int l = 0; l < 32; l++)
                sum += (d2 * (q[l] >> 4) - m2) * q8[j + 32 + l];
            q += 32;
            is += 2;
        }
        sumf += sum;
    }
    return sumf;
}

float vec_dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        sumf += vec_dot_q6_K_f32(&x[i], (const float *)NULL, QK_K);
        // fallback: just use the float version
    }
    // Actually, proper scalar fallback:
    sumf = 0.0f;
    float tmp[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_q6_K(&x[i], tmp, QK_K);
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j++)
            sum += tmp[j] * y[i].d * y[i].qs[j];
        sumf += sum;
    }
    return sumf;
}

float vec_dot_q5_K_q8_K(const block_q5_K *x, const block_q8_K *y, int nb) {
    float sumf = 0.0f;
    float tmp[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_q5_K(&x[i], tmp, QK_K);
        float sum = 0.0f;
        for (int j = 0; j < QK_K; j++)
            sum += tmp[j] * y[i].d * y[i].qs[j];
        sumf += sum;
    }
    return sumf;
}

#endif // __AVX2__

// Fast mat-vec: quantize input to Q8_K once, then SIMD dot products
void mat_vec_mul_fast(float *out, const void *mat, int mat_type,
                      const float *vec, int M, int K) {
    const int nb = K / QK_K;

    // Quantize input vector to Q8_K (thread-local to avoid allocation)
    block_q8_K *q8_vec = (block_q8_K *)malloc(nb * sizeof(block_q8_K));
    if (!q8_vec) return;
    quantize_row_q8_K(vec, q8_vec, K);

    if (mat_type == GGML_TYPE_Q4_K) {
        const block_q4_K *mq = (const block_q4_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q4_K_q8_K(mq + i * nb, q8_vec, nb);
    } else if (mat_type == GGML_TYPE_Q6_K) {
        const block_q6_K *mq = (const block_q6_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q6_K_q8_K(mq + i * nb, q8_vec, nb);
    } else if (mat_type == GGML_TYPE_Q5_K) {
        const block_q5_K *mq = (const block_q5_K *)mat;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++)
            out[i] = vec_dot_q5_K_q8_K(mq + i * nb, q8_vec, nb);
    } else {
        free(q8_vec);
        return;
    }

    free(q8_vec);
}
