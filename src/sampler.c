#include "sampler.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *recent_tokens, int n_recent,
                              float penalty) {
    for (int i = 0; i < n_recent; i++) {
        int tid = recent_tokens[i];
        if (tid < 0 || tid >= vocab_size) continue;
        if (logits[tid] > 0.0f) {
            logits[tid] /= penalty;
        } else {
            logits[tid] *= penalty;
        }
    }
}

int sample_argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int sample_token(const float *logits, int vocab_size,
                 float temperature, float top_p) {
    if (temperature <= 0.0f) {
        return sample_argmax(logits, vocab_size);
    }

    // Apply temperature and softmax in-place using thread-local static buffer
    static __thread float *probs = NULL;
    static __thread int probs_size = 0;
    if (probs_size < vocab_size) {
        free(probs);
        probs = (float *)malloc(vocab_size * sizeof(float));
        probs_size = vocab_size;
    }

    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] *= inv_sum;
    }

    // Top-p (nucleus) sampling — use partial selection instead of full qsort.
    // Build a candidate list of tokens with prob above a threshold, then sort
    // only those. For typical distributions, this is 10-100 tokens, not 152K.
    if (top_p < 1.0f) {
        // Find a reasonable threshold: we need tokens that sum to top_p.
        // Start by collecting all tokens above a threshold, then sort them.
        // Threshold = max_prob / 1000 filters out most of the 152K vocab.
        float max_prob = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > max_prob) max_prob = probs[i];
        }
        float threshold = max_prob * 1e-4f; // keep tokens with >0.01% of max

        // Collect candidates
        static __thread int *candidates = NULL;
        static __thread int candidates_cap = 0;
        if (candidates_cap < vocab_size) {
            free(candidates);
            candidates = (int *)malloc(vocab_size * sizeof(int));
            candidates_cap = vocab_size;
        }

        int n_cand = 0;
        float cand_sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] >= threshold) {
                candidates[n_cand++] = i;
                cand_sum += probs[i];
            }
        }

        // Sort candidates by probability (descending) — typically 10-1000 elements
        // Simple insertion sort is fast for small N
        for (int i = 1; i < n_cand; i++) {
            int key = candidates[i];
            float key_prob = probs[key];
            int j = i - 1;
            while (j >= 0 && probs[candidates[j]] < key_prob) {
                candidates[j + 1] = candidates[j];
                j--;
            }
            candidates[j + 1] = key;
        }

        // Find top-p cutoff
        float cumsum = 0.0f;
        int cutoff = n_cand;
        for (int i = 0; i < n_cand; i++) {
            cumsum += probs[candidates[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out everything beyond cutoff
        for (int i = cutoff; i < n_cand; i++) {
            probs[candidates[i]] = 0.0f;
        }

        // Renormalize
        sum = 0.0f;
        for (int i = 0; i < cutoff; i++) sum += probs[candidates[i]];
        inv_sum = 1.0f / sum;
        // Only renormalize the candidates we're keeping
        for (int i = 0; i < cutoff; i++) probs[candidates[i]] *= inv_sum;
    }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return i;
        }
    }

    return vocab_size - 1;
}
