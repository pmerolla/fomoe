#include "freq_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

freq_profile_t *freq_profile_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "freq_profile: cannot open %s\n", path);
        return NULL;
    }

    freq_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "freq_profile: failed to read header from %s\n", path);
        fclose(f);
        return NULL;
    }

    if (hdr.magic != FREQ_MAGIC) {
        fprintf(stderr, "freq_profile: bad magic 0x%08x (expected 0x%08x)\n",
                hdr.magic, FREQ_MAGIC);
        fclose(f);
        return NULL;
    }
    if (hdr.version != FREQ_VERSION) {
        fprintf(stderr, "freq_profile: unsupported version %u\n", hdr.version);
        fclose(f);
        return NULL;
    }

    freq_profile_t *fp = calloc(1, sizeof(*fp));
    if (!fp) { fclose(f); return NULL; }

    fp->n_layers = hdr.n_layers;
    fp->n_entries = hdr.n_entries;
    fp->entries = malloc(fp->n_entries * sizeof(freq_entry_t));
    if (!fp->entries) {
        free(fp);
        fclose(f);
        return NULL;
    }

    if (fread(fp->entries, sizeof(freq_entry_t), fp->n_entries, f) != (size_t)fp->n_entries) {
        fprintf(stderr, "freq_profile: truncated data in %s\n", path);
        free(fp->entries);
        free(fp);
        fclose(f);
        return NULL;
    }

    fclose(f);
    fprintf(stderr, "freq_profile: loaded %d entries (%d layers) from %s\n",
            fp->n_entries, fp->n_layers, path);
    return fp;
}

int freq_profile_save(const char *path, const freq_profile_t *fp) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "freq_profile: cannot create %s\n", path);
        return -1;
    }

    freq_header_t hdr = {
        .magic = FREQ_MAGIC,
        .version = FREQ_VERSION,
        .n_layers = fp->n_layers,
        .n_entries = fp->n_entries,
    };

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
        fwrite(fp->entries, sizeof(freq_entry_t), fp->n_entries, f) != (size_t)fp->n_entries) {
        fprintf(stderr, "freq_profile: write error to %s\n", path);
        fclose(f);
        return -1;
    }

    fclose(f);
    fprintf(stderr, "freq_profile: saved %d entries to %s\n", fp->n_entries, path);
    return 0;
}

void freq_profile_free(freq_profile_t *fp) {
    if (!fp) return;
    free(fp->entries);
    free(fp);
}

static int cmp_freq_desc(const void *a, const void *b) {
    float fa = ((const freq_entry_t *)a)->frequency;
    float fb = ((const freq_entry_t *)b)->frequency;
    return (fb > fa) - (fb < fa);
}

freq_profile_t *freq_profile_from_counts(const uint32_t *counts,
                                          int n_layers, int n_experts,
                                          int n_tokens) {
    if (n_tokens <= 0) return NULL;

    // Count non-zero entries
    int n_entries = 0;
    for (int l = 0; l < n_layers; l++)
        for (int e = 0; e < n_experts; e++)
            if (counts[l * n_experts + e] > 0)
                n_entries++;

    freq_profile_t *fp = calloc(1, sizeof(*fp));
    if (!fp) return NULL;

    fp->n_layers = n_layers;
    fp->n_entries = n_entries;
    fp->entries = malloc(n_entries * sizeof(freq_entry_t));
    if (!fp->entries) { free(fp); return NULL; }

    int idx = 0;
    float inv_tokens = 1.0f / n_tokens;
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            uint32_t c = counts[l * n_experts + e];
            if (c > 0) {
                fp->entries[idx++] = (freq_entry_t){
                    .layer = (uint16_t)l,
                    .expert_id = (uint16_t)e,
                    .frequency = c * inv_tokens,
                };
            }
        }
    }

    qsort(fp->entries, n_entries, sizeof(freq_entry_t), cmp_freq_desc);
    return fp;
}

int freq_profile_top_for_layer(const freq_profile_t *fp, int layer,
                               freq_entry_t *out, int max_out) {
    int count = 0;
    for (int i = 0; i < fp->n_entries && count < max_out; i++) {
        if (fp->entries[i].layer == layer)
            out[count++] = fp->entries[i];
    }
    return count;
}
