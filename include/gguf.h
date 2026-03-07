#ifndef QMOE_GGUF_H
#define QMOE_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// GGUF format constants
#define GGUF_MAGIC   0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3

// Maximum number of split files
#define GGUF_MAX_SPLITS 16

// GGUF value types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ggml tensor types (subset we care about)
enum ggml_dtype {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_MXFP4   = 39,
};

#define GGUF_MAX_DIMS 4
#define GGUF_DEFAULT_ALIGNMENT 32

// Key-value pair
typedef struct {
    char    *key;
    enum gguf_type type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        bool      b;
        char     *str;
        struct {
            enum gguf_type type;
            uint64_t       n;
            void          *data;    // raw array data
            char         **strings; // only for string arrays
        } arr;
    } value;
} gguf_kv_t;

// Tensor info (metadata only, not the data itself)
typedef struct {
    char          *name;
    uint32_t       n_dims;
    int64_t        dims[GGUF_MAX_DIMS];
    enum ggml_dtype type;
    uint64_t       offset;     // offset within data section of its split file
    uint64_t       size;       // computed size in bytes
    int            split_idx;  // which split file this tensor is in (0 for non-split)
} gguf_tensor_info_t;

// Per-split file info
typedef struct {
    int         fd;
    uint64_t    data_offset;   // absolute file offset where tensor data begins
} gguf_split_t;

// Main GGUF context (supports single or split files)
typedef struct {
    uint32_t version;
    uint64_t alignment;
    uint64_t data_offset;  // data_offset of the first/primary file

    int64_t         n_kv;
    gguf_kv_t      *kv;

    int64_t              n_tensors;
    gguf_tensor_info_t  *tensors;

    // File handle(s) for reading tensor data
    const char *filename;
    int         fd;            // primary file fd (for single-file compat)

    // Split file support
    int          n_splits;
    gguf_split_t splits[GGUF_MAX_SPLITS];
} gguf_ctx_t;

// Open and parse a GGUF file (reads all metadata, not tensor data)
// For split files, pass the first file (xxx-00001-of-NNNNN.gguf) and
// the rest will be auto-detected.
gguf_ctx_t *gguf_open(const char *filename);

// Close and free
void gguf_close(gguf_ctx_t *ctx);

// Look up a KV pair by key, returns index or -1
int64_t gguf_find_key(const gguf_ctx_t *ctx, const char *key);

// Get typed values (caller must check type)
uint32_t    gguf_get_u32(const gguf_ctx_t *ctx, const char *key);
int32_t     gguf_get_i32(const gguf_ctx_t *ctx, const char *key);
float       gguf_get_f32(const gguf_ctx_t *ctx, const char *key);
uint64_t    gguf_get_u64(const gguf_ctx_t *ctx, const char *key);
const char *gguf_get_str(const gguf_ctx_t *ctx, const char *key);

// Get array KV
uint64_t    gguf_get_arr_n(const gguf_ctx_t *ctx, const char *key);
const void *gguf_get_arr_data(const gguf_ctx_t *ctx, const char *key);
const char *gguf_get_arr_str(const gguf_ctx_t *ctx, const char *key, uint64_t i);

// Tensor lookup
int64_t gguf_find_tensor(const gguf_ctx_t *ctx, const char *name);

// Read tensor data into pre-allocated buffer
// Returns bytes read, or 0 on error
size_t gguf_read_tensor(const gguf_ctx_t *ctx, int64_t tensor_id, void *buf, size_t buf_size);

// Compute total byte size for a tensor given its type and element count
size_t gguf_tensor_nbytes(enum ggml_dtype type, int64_t nelements);

// Get type info
size_t      ggml_type_block_size(enum ggml_dtype type);
size_t      ggml_type_size(enum ggml_dtype type);
const char *ggml_type_name(enum ggml_dtype type);

#endif // QMOE_GGUF_H
