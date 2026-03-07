#ifndef QMOE_FREQ_PROFILE_H
#define QMOE_FREQ_PROFILE_H

#include <stdint.h>

#define FREQ_MAGIC   0x46524551  // "FREQ" little-endian
#define FREQ_VERSION 1

// On-disk header (16 bytes)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_entries;
} freq_header_t;

// Single entry: which expert in which layer, how often it was used
typedef struct {
    uint16_t layer;
    uint16_t expert_id;
    float    frequency;     // normalized 0-1
} freq_entry_t;

// Runtime frequency profile
typedef struct freq_profile {
    int           n_layers;
    int           n_entries;
    freq_entry_t *entries;      // sorted by frequency descending
} freq_profile_t;

// Load from binary file. Returns NULL on error.
freq_profile_t *freq_profile_load(const char *path);

// Save to binary file. Returns 0 on success.
int freq_profile_save(const char *path, const freq_profile_t *fp);

// Free profile
void freq_profile_free(freq_profile_t *fp);

// Create profile from observed expert counts during inference.
// counts: [n_layers * n_experts], raw hit counts per expert.
// n_tokens: total tokens observed (for normalization).
freq_profile_t *freq_profile_from_counts(const uint32_t *counts,
                                          int n_layers, int n_experts,
                                          int n_tokens);

// Get top-N entries for a specific layer.
// Writes up to max_out entries to out[], returns actual count.
int freq_profile_top_for_layer(const freq_profile_t *fp, int layer,
                               freq_entry_t *out, int max_out);

#endif // QMOE_FREQ_PROFILE_H
