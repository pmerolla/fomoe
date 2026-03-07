#ifndef QMOE_TOKENIZER_H
#define QMOE_TOKENIZER_H

#include "gguf.h"
#include <stdint.h>

#define MAX_TOKEN_LEN 256
#define MAX_MERGE_PARTS 2

typedef struct {
    char    **tokens;        // token_id -> string (GPT-2 encoded)
    char    **decoded;       // token_id -> string (raw bytes, for output)
    int      *token_types;   // token types
    int       vocab_size;

    // BPE merges: pairs of token strings that merge
    char    **merge_a;       // left side of merge
    char    **merge_b;       // right side of merge
    int       n_merges;

    // GPT-2 byte-level encoding tables
    uint32_t  byte_to_cp[256];    // byte value -> Unicode codepoint
    uint8_t   cp_to_byte[324];    // codepoint (0..323) -> byte value

    // Special tokens
    int       bos_id;
    int       eos_id;
    int       pad_id;
} tokenizer_t;

// Load tokenizer from GGUF metadata
tokenizer_t *tokenizer_load(const gguf_ctx_t *ctx);

// Free tokenizer
void tokenizer_free(tokenizer_t *tok);

// Encode text to token IDs
// Returns number of tokens written to out_ids
// out_ids must be pre-allocated (max_tokens capacity)
int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int *out_ids, int max_tokens);

// Decode a single token ID to string
const char *tokenizer_decode(const tokenizer_t *tok, int token_id);

#endif // QMOE_TOKENIZER_H
