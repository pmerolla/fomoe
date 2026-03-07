#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- GPT-2 byte-level encoding ----
// GPT-2 BPE maps each byte (0-255) to a Unicode codepoint.
// "Printable" bytes map to themselves, others shift to U+0100+.
// This means space (0x20) -> U+0120 (Ġ), newline (0x0A) -> U+010A, etc.

static void build_gpt2_tables(uint32_t byte_to_cp[256], uint8_t cp_to_byte[324]) {
    memset(cp_to_byte, 0, 324);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            byte_to_cp[b] = (uint32_t)b;
        } else {
            byte_to_cp[b] = 0x100 + n;
            n++;
        }
        cp_to_byte[byte_to_cp[b]] = (uint8_t)b;
    }
}

// Encode a Unicode codepoint (< 0x800) to UTF-8, return number of bytes written
static int cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
}

// Parse one UTF-8 codepoint from string, return bytes consumed
static int utf8_to_cp(const char *s, uint32_t *cp) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    } else if ((c & 0xE0) == 0xC0) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    } else if ((c & 0xF0) == 0xE0) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    } else {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
              ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
}

// Decode a GPT-2 token string to raw bytes
static char *decode_gpt2_token(const char *token, const uint8_t cp_to_byte[324]) {
    size_t len = strlen(token);
    char *out = malloc(len + 1);
    if (!out) return NULL;
    int oi = 0;
    const char *p = token;

    while (*p) {
        uint32_t cp;
        int n = utf8_to_cp(p, &cp);
        if (cp < 324) {
            out[oi++] = (char)cp_to_byte[cp];
        } else {
            // Codepoint outside GPT-2 range - copy raw UTF-8 bytes
            for (int j = 0; j < n; j++) out[oi++] = p[j];
        }
        p += n;
    }

    out[oi] = '\0';
    return out;
}

// ---- Hash table for O(1) lookups ----

typedef struct {
    char    *key;       // owned string
    int      value;     // merge rank or token ID
} ht_entry_t;

typedef struct {
    ht_entry_t *entries;
    int          cap;       // power of 2
    int          mask;      // cap - 1
} hash_table_t;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void ht_init(hash_table_t *ht, int min_cap) {
    int cap = 1;
    while (cap < min_cap * 2) cap <<= 1;  // load factor < 0.5
    ht->cap = cap;
    ht->mask = cap - 1;
    ht->entries = calloc(cap, sizeof(ht_entry_t));
}

static void ht_free(hash_table_t *ht) {
    if (!ht->entries) return;
    for (int i = 0; i < ht->cap; i++)
        free(ht->entries[i].key);
    free(ht->entries);
    ht->entries = NULL;
}

static void ht_insert(hash_table_t *ht, const char *key, int value) {
    uint32_t idx = fnv1a(key) & ht->mask;
    while (ht->entries[idx].key) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            ht->entries[idx].value = value;  // update
            return;
        }
        idx = (idx + 1) & ht->mask;
    }
    ht->entries[idx].key = strdup(key);
    ht->entries[idx].value = value;
}

static int ht_lookup(const hash_table_t *ht, const char *key) {
    if (!ht->entries) return -1;
    uint32_t idx = fnv1a(key) & ht->mask;
    while (ht->entries[idx].key) {
        if (strcmp(ht->entries[idx].key, key) == 0)
            return ht->entries[idx].value;
        idx = (idx + 1) & ht->mask;
    }
    return -1;
}

// ---- Global hash tables (built once at load time) ----

static hash_table_t g_merge_ht;   // "a\x00b" -> merge rank
static hash_table_t g_token_ht;   // token string -> token ID

// Build merge key: "a\x01b" (using \x01 as separator since it doesn't appear in tokens)
static char *make_merge_key(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *key = malloc(la + 1 + lb + 1);
    memcpy(key, a, la);
    key[la] = '\x01';
    memcpy(key + la + 1, b, lb);
    key[la + 1 + lb] = '\0';
    return key;
}

tokenizer_t *tokenizer_load(const gguf_ctx_t *ctx) {
    tokenizer_t *tok = calloc(1, sizeof(tokenizer_t));
    if (!tok) return NULL;

    // Load vocab
    uint64_t n_tokens = gguf_get_arr_n(ctx, "tokenizer.ggml.tokens");
    if (n_tokens == 0) {
        fprintf(stderr, "tokenizer: no tokens found\n");
        free(tok);
        return NULL;
    }

    tok->vocab_size = (int)n_tokens;
    tok->tokens = calloc(n_tokens, sizeof(char *));
    if (!tok->tokens) { free(tok); return NULL; }

    for (uint64_t i = 0; i < n_tokens; i++) {
        const char *s = gguf_get_arr_str(ctx, "tokenizer.ggml.tokens", i);
        tok->tokens[i] = s ? strdup(s) : strdup("");
    }

    // Load token types
    int64_t type_idx = gguf_find_key(ctx, "tokenizer.ggml.token_type");
    if (type_idx >= 0) {
        tok->token_types = calloc(n_tokens, sizeof(int));
        const void *type_data = gguf_get_arr_data(ctx, "tokenizer.ggml.token_type");
        if (type_data && tok->token_types) {
            memcpy(tok->token_types, type_data, n_tokens * sizeof(int32_t));
        }
    }

    // Load BPE merges
    uint64_t n_merges = gguf_get_arr_n(ctx, "tokenizer.ggml.merges");
    if (n_merges > 0) {
        tok->n_merges = (int)n_merges;
        tok->merge_a = calloc(n_merges, sizeof(char *));
        tok->merge_b = calloc(n_merges, sizeof(char *));

        for (uint64_t i = 0; i < n_merges; i++) {
            const char *merge = gguf_get_arr_str(ctx, "tokenizer.ggml.merges", i);
            if (!merge) continue;

            // Each merge is "token_a token_b" separated by space
            const char *space = strchr(merge, ' ');
            if (space) {
                size_t len_a = space - merge;
                tok->merge_a[i] = strndup(merge, len_a);
                tok->merge_b[i] = strdup(space + 1);
            }
        }
    }

    // Special tokens
    tok->eos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.eos_token_id");
    tok->bos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.bos_token_id");
    tok->pad_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.padding_token_id");

    // Build GPT-2 byte encoding tables
    build_gpt2_tables(tok->byte_to_cp, tok->cp_to_byte);

    // Pre-decode all tokens from GPT-2 encoding to raw bytes
    tok->decoded = calloc(n_tokens, sizeof(char *));
    if (tok->decoded) {
        for (uint64_t i = 0; i < n_tokens; i++) {
            tok->decoded[i] = decode_gpt2_token(tok->tokens[i], tok->cp_to_byte);
        }
    }

    // Build hash tables for O(1) lookups
    // Merge hash: "a\x01b" -> rank
    ht_init(&g_merge_ht, tok->n_merges + 1);
    for (int i = 0; i < tok->n_merges; i++) {
        if (tok->merge_a[i] && tok->merge_b[i]) {
            char *key = make_merge_key(tok->merge_a[i], tok->merge_b[i]);
            ht_insert(&g_merge_ht, key, i);
            free(key);
        }
    }

    // Token hash: token string -> ID
    ht_init(&g_token_ht, tok->vocab_size + 1);
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i][0])  // skip empty strings
            ht_insert(&g_token_ht, tok->tokens[i], i);
    }

    fprintf(stderr, "tokenizer: loaded %d tokens, %d merges\n",
            tok->vocab_size, tok->n_merges);
    fprintf(stderr, "  bos=%d, eos=%d, pad=%d\n", tok->bos_id, tok->eos_id, tok->pad_id);

    // Verify GPT-2 mapping: space (0x20) should map to codepoint 0x120
    fprintf(stderr, "  GPT-2 byte map: space(0x20)->U+%04X, 'A'(0x41)->U+%04X\n",
            tok->byte_to_cp[0x20], tok->byte_to_cp[0x41]);

    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;

    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->tokens[i]);
        if (tok->decoded) free(tok->decoded[i]);
    }
    free(tok->tokens);
    free(tok->decoded);
    free(tok->token_types);

    for (int i = 0; i < tok->n_merges; i++) {
        free(tok->merge_a[i]);
        free(tok->merge_b[i]);
    }
    free(tok->merge_a);
    free(tok->merge_b);

    ht_free(&g_merge_ht);
    ht_free(&g_token_ht);

    free(tok);
}

// ---- BPE Encoding ----

typedef struct {
    char **pieces;   // current sequence of string pieces
    int    n_pieces;
    int    cap;
} bpe_state_t;

static void bpe_init(bpe_state_t *s, int cap) {
    s->pieces = calloc(cap, sizeof(char *));
    s->n_pieces = 0;
    s->cap = cap;
}

static void bpe_free(bpe_state_t *s) {
    for (int i = 0; i < s->n_pieces; i++) free(s->pieces[i]);
    free(s->pieces);
}

// Find the merge rank for a pair (a, b), returns -1 if not found — O(1) via hash
static int find_merge(const char *a, const char *b) {
    char *key = make_merge_key(a, b);
    int rank = ht_lookup(&g_merge_ht, key);
    free(key);
    return rank;
}

// Find token ID for a string — O(1) via hash
static int find_token(const char *s) {
    return ht_lookup(&g_token_ht, s);
}

// Apply BPE merges to a list of pieces
static void apply_bpe(bpe_state_t *s) {
    while (s->n_pieces > 1) {
        // Find the highest-priority merge (lowest merge rank)
        int best_rank = 0x7fffffff;
        int best_pos = -1;

        for (int i = 0; i < s->n_pieces - 1; i++) {
            int rank = find_merge(s->pieces[i], s->pieces[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_pos < 0) break;  // no more merges

        // Merge pieces at best_pos and best_pos+1
        size_t len = strlen(s->pieces[best_pos]) + strlen(s->pieces[best_pos + 1]);
        char *merged = malloc(len + 1);
        strcpy(merged, s->pieces[best_pos]);
        strcat(merged, s->pieces[best_pos + 1]);

        free(s->pieces[best_pos]);
        free(s->pieces[best_pos + 1]);
        s->pieces[best_pos] = merged;

        // Shift remaining pieces left
        for (int i = best_pos + 1; i < s->n_pieces - 1; i++) {
            s->pieces[i] = s->pieces[i + 1];
        }
        s->n_pieces--;
    }
}

int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int *out_ids, int max_tokens) {
    if (!text || !text[0]) return 0;

    int n_out = 0;
    size_t text_len = strlen(text);

    // Initialize BPE state: each byte becomes a GPT-2 encoded piece
    bpe_state_t s;
    bpe_init(&s, text_len * 2 + 1);  // each byte may become 2-byte UTF-8

    // Convert each input byte to its GPT-2 Unicode character (UTF-8 encoded)
    for (size_t i = 0; i < text_len && s.n_pieces < s.cap; i++) {
        uint8_t byte = (uint8_t)text[i];
        uint32_t cp = tok->byte_to_cp[byte];
        char buf[4];
        int n = cp_to_utf8(cp, buf);
        buf[n] = '\0';
        s.pieces[s.n_pieces++] = strdup(buf);
    }

    // Apply BPE merges
    apply_bpe(&s);

    // Convert pieces to token IDs
    for (int i = 0; i < s.n_pieces && n_out < max_tokens; i++) {
        int id = find_token(s.pieces[i]);
        if (id >= 0) {
            out_ids[n_out++] = id;
        } else {
            // Unknown merged piece - split back into individual GPT-2 chars
            const char *p = s.pieces[i];
            while (*p && n_out < max_tokens) {
                uint32_t cp;
                int n = utf8_to_cp(p, &cp);
                char buf[4];
                memcpy(buf, p, n);
                buf[n] = '\0';
                id = find_token(buf);
                if (id >= 0) {
                    out_ids[n_out++] = id;
                }
                p += n;
            }
        }
    }

    bpe_free(&s);
    return n_out;
}

const char *tokenizer_decode(const tokenizer_t *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    // Return the pre-decoded raw bytes version (GPT-2 chars -> actual bytes)
    if (tok->decoded && tok->decoded[token_id])
        return tok->decoded[token_id];
    return tok->tokens[token_id];
}
