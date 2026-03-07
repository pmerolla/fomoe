#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// ---- Type info tables ----

typedef struct {
    const char *name;
    size_t      type_size;   // bytes per block
    size_t      block_size;  // elements per block
} type_info_t;

static const type_info_t TYPE_INFO[] = {
    [GGML_TYPE_F32]  = { "f32",  4, 1 },
    [GGML_TYPE_F16]  = { "f16",  2, 1 },
    [GGML_TYPE_Q4_0] = { "q4_0", 18, 32 },
    [GGML_TYPE_Q4_1] = { "q4_1", 20, 32 },
    [GGML_TYPE_Q5_0] = { "q5_0", 22, 32 },
    [GGML_TYPE_Q5_1] = { "q5_1", 24, 32 },
    [GGML_TYPE_Q8_0] = { "q8_0", 34, 32 },
    [GGML_TYPE_Q8_1] = { "q8_1", 36, 32 },
    [GGML_TYPE_Q2_K] = { "q2_K", 2*2 + 256/16 + 256/4, 256 },  // 84
    [GGML_TYPE_Q3_K] = { "q3_K", 2 + 256/4 + 256/8 + 12, 256 },  // 110
    [GGML_TYPE_Q4_K] = { "q4_K", 2*2 + 12 + 256/2, 256 },  // 144
    [GGML_TYPE_Q5_K] = { "q5_K", 2*2 + 12 + 256/8 + 256/2, 256 },  // 176
    [GGML_TYPE_Q6_K] = { "q6_K", 2 + 256/2 + 256/4 + 256/16, 256 },  // 210
    [GGML_TYPE_Q8_K] = { "q8_K", 4 + 256 + 256/16*2, 256 },  // 292
    [GGML_TYPE_BF16]  = { "bf16", 2, 1 },
    [GGML_TYPE_MXFP4] = { "mxfp4", 17, 32 },
};
#define N_TYPE_INFO (sizeof(TYPE_INFO) / sizeof(TYPE_INFO[0]))

size_t ggml_type_block_size(enum ggml_dtype type) {
    if ((int)type < 0 || (size_t)type >= N_TYPE_INFO) return 0;
    return TYPE_INFO[type].block_size;
}

size_t ggml_type_size(enum ggml_dtype type) {
    if ((int)type < 0 || (size_t)type >= N_TYPE_INFO) return 0;
    return TYPE_INFO[type].type_size;
}

const char *ggml_type_name(enum ggml_dtype type) {
    if ((int)type < 0 || (size_t)type >= N_TYPE_INFO) return "unknown";
    return TYPE_INFO[type].name ? TYPE_INFO[type].name : "unknown";
}

size_t gguf_tensor_nbytes(enum ggml_dtype type, int64_t nelements) {
    size_t bs = ggml_type_block_size(type);
    size_t ts = ggml_type_size(type);
    if (bs == 0 || ts == 0) return 0;
    return (nelements / bs) * ts;
}

// ---- File reading helpers ----

typedef struct {
    FILE  *f;
    size_t pos;
} reader_t;

static bool read_bytes(reader_t *r, void *buf, size_t n) {
    if (fread(buf, 1, n, r->f) != n) return false;
    r->pos += n;
    return true;
}

static bool read_u8(reader_t *r, uint8_t *v)   { return read_bytes(r, v, 1); }
static bool read_u16(reader_t *r, uint16_t *v)  { return read_bytes(r, v, 2); }
static bool read_u32(reader_t *r, uint32_t *v)  { return read_bytes(r, v, 4); }
static bool read_i32(reader_t *r, int32_t *v)   { return read_bytes(r, v, 4); }
static bool read_u64(reader_t *r, uint64_t *v)  { return read_bytes(r, v, 8); }
static bool read_i64(reader_t *r, int64_t *v)   { return read_bytes(r, v, 8); }
static bool read_f32(reader_t *r, float *v)     { return read_bytes(r, v, 4); }
static bool read_f64(reader_t *r, double *v)    { return read_bytes(r, v, 8); }

static char *read_string(reader_t *r) {
    uint64_t len;
    if (!read_u64(r, &len)) return NULL;
    if (len > 1024 * 1024) return NULL;  // sanity check: 1MB max string

    char *s = malloc(len + 1);
    if (!s) return NULL;
    if (len > 0 && !read_bytes(r, s, len)) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

// Size in bytes of a single GGUF scalar value
static size_t gguf_type_size(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

// Read a KV value into the kv struct
static bool read_kv_value(reader_t *r, gguf_kv_t *kv) {
    int32_t type_i;
    if (!read_i32(r, &type_i)) return false;
    kv->type = (enum gguf_type)type_i;

    switch (kv->type) {
        case GGUF_TYPE_UINT8:   return read_u8(r, &kv->value.u8);
        case GGUF_TYPE_INT8:    return read_bytes(r, &kv->value.i8, 1);
        case GGUF_TYPE_UINT16:  return read_u16(r, &kv->value.u16);
        case GGUF_TYPE_INT16:   return read_bytes(r, &kv->value.i16, 2);
        case GGUF_TYPE_UINT32:  return read_u32(r, &kv->value.u32);
        case GGUF_TYPE_INT32:   return read_i32(r, &kv->value.i32);
        case GGUF_TYPE_FLOAT32: return read_f32(r, &kv->value.f32);
        case GGUF_TYPE_BOOL:    { int8_t b; if (!read_bytes(r, &b, 1)) return false; kv->value.b = b; return true; }
        case GGUF_TYPE_STRING:  kv->value.str = read_string(r); return kv->value.str != NULL;
        case GGUF_TYPE_UINT64:  return read_u64(r, &kv->value.u64);
        case GGUF_TYPE_INT64:   return read_i64(r, &kv->value.i64);
        case GGUF_TYPE_FLOAT64: return read_f64(r, &kv->value.f64);
        case GGUF_TYPE_ARRAY: {
            int32_t arr_type_i;
            if (!read_i32(r, &arr_type_i)) return false;
            kv->value.arr.type = (enum gguf_type)arr_type_i;
            if (!read_u64(r, &kv->value.arr.n)) return false;

            uint64_t n = kv->value.arr.n;

            if (kv->value.arr.type == GGUF_TYPE_STRING) {
                kv->value.arr.strings = calloc(n, sizeof(char *));
                kv->value.arr.data = NULL;
                if (!kv->value.arr.strings) return false;
                for (uint64_t i = 0; i < n; i++) {
                    kv->value.arr.strings[i] = read_string(r);
                    if (!kv->value.arr.strings[i]) return false;
                }
            } else {
                size_t elem_size = gguf_type_size(kv->value.arr.type);
                if (elem_size == 0) return false;
                kv->value.arr.data = malloc(n * elem_size);
                kv->value.arr.strings = NULL;
                if (!kv->value.arr.data && n > 0) return false;
                if (!read_bytes(r, kv->value.arr.data, n * elem_size)) return false;
            }
            return true;
        }
        default: return false;
    }
}

// ---- Split GGUF helpers ----

// Parse a single GGUF file header+KV+tensor info.
// Returns a temporary ctx, or NULL on error. Caller must close the FILE.
static gguf_ctx_t *gguf_parse_one(const char *filename, FILE *f) {
    reader_t r = { .f = f, .pos = 0 };

    uint32_t magic;
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic in %s (got 0x%08x)\n", filename, magic);
        return NULL;
    }

    uint32_t version;
    if (!read_u32(&r, &version)) return NULL;
    if (version < 2 || version > GGUF_VERSION) {
        fprintf(stderr, "gguf: unsupported version %u in %s\n", version, filename);
        return NULL;
    }

    int64_t n_tensors, n_kv;
    if (!read_i64(&r, &n_tensors)) return NULL;
    if (!read_i64(&r, &n_kv))     return NULL;

    gguf_ctx_t *ctx = calloc(1, sizeof(gguf_ctx_t));
    if (!ctx) return NULL;

    ctx->version   = version;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->n_kv      = n_kv;
    ctx->n_tensors = n_tensors;
    ctx->fd        = -1;

    // Read KV pairs
    ctx->kv = calloc(n_kv, sizeof(gguf_kv_t));
    if (!ctx->kv && n_kv > 0) { gguf_close(ctx); return NULL; }

    for (int64_t i = 0; i < n_kv; i++) {
        ctx->kv[i].key = read_string(&r);
        if (!ctx->kv[i].key) { gguf_close(ctx); return NULL; }
        if (!read_kv_value(&r, &ctx->kv[i])) { gguf_close(ctx); return NULL; }

        if (strcmp(ctx->kv[i].key, "general.alignment") == 0 &&
            ctx->kv[i].type == GGUF_TYPE_UINT32) {
            ctx->alignment = ctx->kv[i].value.u32;
        }
    }

    // Read tensor info
    ctx->tensors = calloc(n_tensors, sizeof(gguf_tensor_info_t));
    if (!ctx->tensors && n_tensors > 0) { gguf_close(ctx); return NULL; }

    for (int64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[i];

        ti->name = read_string(&r);
        if (!ti->name) { gguf_close(ctx); return NULL; }

        if (!read_u32(&r, &ti->n_dims)) { gguf_close(ctx); return NULL; }
        if (ti->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "gguf: tensor '%s' has %u dims (max %d)\n",
                    ti->name, ti->n_dims, GGUF_MAX_DIMS);
            gguf_close(ctx); return NULL;
        }

        int64_t nelements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            if (!read_i64(&r, &ti->dims[d])) { gguf_close(ctx); return NULL; }
            nelements *= ti->dims[d];
        }
        for (uint32_t d = ti->n_dims; d < GGUF_MAX_DIMS; d++) {
            ti->dims[d] = 1;
        }

        int32_t type_i;
        if (!read_i32(&r, &type_i)) { gguf_close(ctx); return NULL; }
        ti->type = (enum ggml_dtype)type_i;

        if (!read_u64(&r, &ti->offset)) { gguf_close(ctx); return NULL; }

        ti->size = gguf_tensor_nbytes(ti->type, nelements);
        ti->split_idx = 0;
    }

    size_t header_end = r.pos;
    ctx->data_offset = (header_end + ctx->alignment - 1) & ~(ctx->alignment - 1);

    return ctx;
}

// Detect split count from filename pattern: xxx-00001-of-NNNNN.gguf
static int detect_split_count(const char *filename) {
    const char *p = strstr(filename, "-of-");
    if (!p) return 1;
    p += 4;
    int count = atoi(p);
    return (count > 1 && count <= GGUF_MAX_SPLITS) ? count : 1;
}

// Generate split filename by replacing the NNNNN part before "-of-"
static char *make_split_filename(const char *filename, int split_no, int total) {
    const char *dash_of = strstr(filename, "-of-");
    if (!dash_of) return strdup(filename);

    // Walk back to find start of split number
    const char *num_start = dash_of;
    while (num_start > filename && num_start[-1] >= '0' && num_start[-1] <= '9') {
        num_start--;
    }

    size_t prefix_len = num_start - filename;
    char *result = malloc(strlen(filename) + 16);
    if (!result) return NULL;

    memcpy(result, filename, prefix_len);
    int n = sprintf(result + prefix_len, "%05d-of-%05d", split_no, total);
    // Skip past the original "NNNNN-of-NNNNN" in the source
    const char *after_of = dash_of + 4;
    while (*after_of >= '0' && *after_of <= '9') after_of++;
    strcpy(result + prefix_len + n, after_of);

    return result;
}

// ---- Public API ----

gguf_ctx_t *gguf_open(const char *filename) {
    int n_splits = detect_split_count(filename);

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "gguf: cannot open %s: %s\n", filename, strerror(errno));
        return NULL;
    }

    fprintf(stderr, "gguf: opening %s", filename);
    if (n_splits > 1) fprintf(stderr, " (%d splits)", n_splits);
    fprintf(stderr, "\n");

    gguf_ctx_t *ctx = gguf_parse_one(filename, f);
    fclose(f);
    if (!ctx) return NULL;

    fprintf(stderr, "gguf: version=%u, n_tensors=%ld, n_kv=%ld\n",
            ctx->version, (long)ctx->n_tensors, (long)ctx->n_kv);

    ctx->filename = filename;

    // Open primary fd
    ctx->fd = open(filename, O_RDONLY);
    if (ctx->fd < 0) {
        fprintf(stderr, "gguf: cannot open fd for %s\n", filename);
        gguf_close(ctx);
        return NULL;
    }

    ctx->n_splits = 1;
    ctx->splits[0].fd = ctx->fd;
    ctx->splits[0].data_offset = ctx->data_offset;

    // All tensors from the first file belong to split 0
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        ctx->tensors[i].split_idx = 0;
    }

    // Handle split files
    if (n_splits > 1) {
        for (int s = 2; s <= n_splits; s++) {
            char *split_path = make_split_filename(filename, s, n_splits);
            if (!split_path) continue;

            FILE *sf = fopen(split_path, "rb");
            if (!sf) {
                fprintf(stderr, "gguf: cannot open split %s: %s\n", split_path, strerror(errno));
                free(split_path);
                continue;
            }

            gguf_ctx_t *sc = gguf_parse_one(split_path, sf);
            fclose(sf);

            if (!sc) {
                fprintf(stderr, "gguf: failed parsing split %s\n", split_path);
                free(split_path);
                continue;
            }

            int split_fd = open(split_path, O_RDONLY);
            if (split_fd < 0) {
                fprintf(stderr, "gguf: cannot open fd for %s\n", split_path);
                gguf_close(sc);
                free(split_path);
                continue;
            }

            int si = ctx->n_splits;
            ctx->splits[si].fd = split_fd;
            ctx->splits[si].data_offset = sc->data_offset;
            ctx->n_splits++;

            fprintf(stderr, "gguf: split %d/%d: %ld tensors from %s\n",
                    s, n_splits, (long)sc->n_tensors, split_path);

            // Merge tensors
            if (sc->n_tensors > 0) {
                int64_t old_n = ctx->n_tensors;
                int64_t new_n = old_n + sc->n_tensors;
                gguf_tensor_info_t *nt = realloc(ctx->tensors,
                    new_n * sizeof(gguf_tensor_info_t));
                if (nt) {
                    ctx->tensors = nt;
                    for (int64_t i = 0; i < sc->n_tensors; i++) {
                        ctx->tensors[old_n + i] = sc->tensors[i];
                        ctx->tensors[old_n + i].split_idx = si;
                        sc->tensors[i].name = NULL;  // ownership transferred
                    }
                    ctx->n_tensors = new_n;
                }
            }

            // Merge new KV pairs (skip duplicates)
            for (int64_t i = 0; i < sc->n_kv; i++) {
                if (sc->kv[i].key && gguf_find_key(ctx, sc->kv[i].key) < 0) {
                    gguf_kv_t *nkv = realloc(ctx->kv,
                        (ctx->n_kv + 1) * sizeof(gguf_kv_t));
                    if (nkv) {
                        ctx->kv = nkv;
                        ctx->kv[ctx->n_kv] = sc->kv[i];
                        sc->kv[i].key = NULL;
                        if (sc->kv[i].type == GGUF_TYPE_STRING)
                            sc->kv[i].value.str = NULL;
                        ctx->n_kv++;
                    }
                }
            }

            gguf_close(sc);
            free(split_path);
        }

        fprintf(stderr, "gguf: merged %d splits, total %ld tensors, %ld kv\n",
                ctx->n_splits, (long)ctx->n_tensors, (long)ctx->n_kv);
    }

    return ctx;
}

void gguf_close(gguf_ctx_t *ctx) {
    if (!ctx) return;

    // Close split fds
    for (int s = 0; s < ctx->n_splits; s++) {
        if (ctx->splits[s].fd >= 0) close(ctx->splits[s].fd);
    }
    // If no splits were set up, close the primary fd
    if (ctx->n_splits == 0 && ctx->fd >= 0) close(ctx->fd);

    for (int64_t i = 0; i < ctx->n_kv; i++) {
        free(ctx->kv[i].key);
        if (ctx->kv[i].type == GGUF_TYPE_STRING) {
            free(ctx->kv[i].value.str);
        } else if (ctx->kv[i].type == GGUF_TYPE_ARRAY) {
            if (ctx->kv[i].value.arr.type == GGUF_TYPE_STRING && ctx->kv[i].value.arr.strings) {
                for (uint64_t j = 0; j < ctx->kv[i].value.arr.n; j++) {
                    free(ctx->kv[i].value.arr.strings[j]);
                }
                free(ctx->kv[i].value.arr.strings);
            }
            free(ctx->kv[i].value.arr.data);
        }
    }
    free(ctx->kv);

    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        free(ctx->tensors[i].name);
    }
    free(ctx->tensors);

    free(ctx);
}

int64_t gguf_find_key(const gguf_ctx_t *ctx, const char *key) {
    for (int64_t i = 0; i < ctx->n_kv; i++) {
        if (ctx->kv[i].key && strcmp(ctx->kv[i].key, key) == 0) return i;
    }
    return -1;
}

uint32_t gguf_get_u32(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) { fprintf(stderr, "gguf: key not found: %s\n", key); return 0; }
    return ctx->kv[idx].value.u32;
}

int32_t gguf_get_i32(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) { fprintf(stderr, "gguf: key not found: %s\n", key); return 0; }
    return ctx->kv[idx].value.i32;
}

float gguf_get_f32(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) { fprintf(stderr, "gguf: key not found: %s\n", key); return 0; }
    return ctx->kv[idx].value.f32;
}

uint64_t gguf_get_u64(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return 0;
    return ctx->kv[idx].value.u64;
}

const char *gguf_get_str(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    return ctx->kv[idx].value.str;
}

uint64_t gguf_get_arr_n(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return 0;
    return ctx->kv[idx].value.arr.n;
}

const void *gguf_get_arr_data(const gguf_ctx_t *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    return ctx->kv[idx].value.arr.data;
}

const char *gguf_get_arr_str(const gguf_ctx_t *ctx, const char *key, uint64_t i) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    if (i >= ctx->kv[idx].value.arr.n) return NULL;
    return ctx->kv[idx].value.arr.strings[i];
}

int64_t gguf_find_tensor(const gguf_ctx_t *ctx, const char *name) {
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].name && strcmp(ctx->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

size_t gguf_read_tensor(const gguf_ctx_t *ctx, int64_t tensor_id, void *buf, size_t buf_size) {
    if (tensor_id < 0 || tensor_id >= ctx->n_tensors) return 0;

    const gguf_tensor_info_t *ti = &ctx->tensors[tensor_id];
    if (ti->size > buf_size) {
        fprintf(stderr, "gguf: buffer too small for tensor '%s' (need %zu, have %zu)\n",
                ti->name, ti->size, buf_size);
        return 0;
    }

    // Select the correct file and data offset for this tensor's split
    int split = ti->split_idx;
    int fd;
    uint64_t data_off;

    if (split >= 0 && split < ctx->n_splits) {
        fd = ctx->splits[split].fd;
        data_off = ctx->splits[split].data_offset;
    } else {
        fd = ctx->fd;
        data_off = ctx->data_offset;
    }

    off_t file_offset = data_off + ti->offset;

    // Handle large reads (>2GB) that may need multiple pread calls
    size_t total_read = 0;
    while (total_read < ti->size) {
        size_t to_read = ti->size - total_read;
        if (to_read > (size_t)(1 << 30)) to_read = (size_t)(1 << 30);

        ssize_t n = pread(fd, (uint8_t *)buf + total_read, to_read,
                          file_offset + (off_t)total_read);
        if (n <= 0) {
            fprintf(stderr, "gguf: failed reading tensor '%s' at offset %ld: %s\n",
                    ti->name, (long)(file_offset + total_read), strerror(errno));
            return 0;
        }
        total_read += n;
    }

    return ti->size;
}
