#ifndef QMOE_INFERENCE_H
#define QMOE_INFERENCE_H

#include "model.h"
#include <stdint.h>

// Run one forward pass for a single token
// Returns pointer to logits (model->buf_logits, size vocab_size)
float *forward(model_t *model, int token_id, int pos);

#endif // QMOE_INFERENCE_H
