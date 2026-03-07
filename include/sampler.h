#ifndef QMOE_SAMPLER_H
#define QMOE_SAMPLER_H

#include <stdint.h>

// Apply repetition penalty to logits for recently generated tokens
void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *recent_tokens, int n_recent,
                              float penalty);

// Sample a token from logits using temperature + top-p
int sample_token(const float *logits, int vocab_size,
                 float temperature, float top_p);

// Greedy (argmax) sampling
int sample_argmax(const float *logits, int vocab_size);

#endif // QMOE_SAMPLER_H
