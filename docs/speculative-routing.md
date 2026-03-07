# Speculative Expert Routing for MoE Inference

**Date:** 2026-03-08
**Status:** Implemented and validated in GPU pipeline (v2, 82-88% accuracy in production)

## Discovery

During NVMe-offloaded MoE inference, the GPU is idle while waiting for expert
weights to transfer from NVMe/RAM to VRAM. We discovered that this idle time
can be used to predict the *next* layer's expert routing with 86% accuracy
by applying a trivially cheap operation: skip the next layer's attention/SSM
entirely and run only `rms_norm + router_matvec + softmax + top_k` on the
current layer's post-FFN hidden state.

This enables **speculative expert prefetching** — start loading predicted
experts from NVMe before the current layer finishes, so they're already
available when needed.

## Key Result

| Predictor              | Accuracy | Notes                          |
|------------------------|----------|--------------------------------|
| Skip-attention routing | **85.8%** | ~7/8 experts correct per layer |
| Same-expert baseline   | 3.4%     | Near random (8/256 = 3.1%)     |
| Random baseline        | 3.1%     | 8 of 256 experts               |

Measured on Qwen 3.5 122B-A10B (Q4_K_M), 48 layers, 256 experts per layer,
8 active per token. Consistent across multiple tokens and layers. Later
layers (30+) frequently achieve 8/8 perfect prediction.

## Why It Works

In a hybrid transformer-MoE architecture, the hidden state residual stream
carries most of the information the router needs. The attention/SSM transform
at layer L+1 *refines* the hidden state but doesn't *replace* it — the
residual connection preserves the post-FFN signal from layer L. Since the
router is a simple linear projection (F32 matvec) of the normalized hidden
state, it's largely determined by the residual stream, not by what attention
adds on top.

This also explains why the "same-expert" baseline fails (3.4%): expert
selection depends on the *content* of the hidden state (which changes every
layer via the FFN), not just its recent history.

## What We Ruled Out First

Before discovering skip-attention prediction, we tested two weight-space
approximation approaches. Both failed:

### 1. Per-Expert SVD Approximation
**Idea:** Store low-rank SVD of each expert's weight matrices in VRAM, use
them to compute approximate FFN output for routing prediction.

**Result:** Expert weight matrices are effectively full-rank with flat
singular value spectra. At rank 64 (3 MB/expert), only 31% of energy is
captured and FFN cosine similarity is 0.26. Even rank 512 (24 MB/expert —
larger than the quantized expert itself at 6 MB) only achieves 0.82 cosine
similarity.

### 2. Expert Clustering + Shared-Basis SVD
**Idea:** Cluster similar experts, compute cluster centroids, use centroid
FFN as approximation. Residuals (expert - centroid) might be low-rank even
if individuals aren't.

**Result:** Experts are functionally orthogonal. Maximum pairwise cosine
similarity across all 256 experts: 0.20. Mean: 0.009. No natural clusters
exist. Even at k=64 clusters, centroid FFN cosine similarity is only 0.05.
Residual SVD spectra are identical to individual SVD — the centroid subtracts
essentially nothing.

**Conclusion:** The 256 experts per layer are designed to be maximally
diverse. No weight-space compression is viable. The skip-attention approach
works precisely because it doesn't try to approximate the expert FFN — it
uses the *real* FFN output (or CAR-approximated output) and just skips the
cheap attention step for prediction.

## Proposed Implementation

### Architecture

```
Layer L (current):
  1. Attention/SSM
  2. FFN norm + Router -> select experts
  3. Classify: VRAM cached / RAM cached / NVMe needed
  4. Phase 1: FFN for VRAM-cached experts (immediate)
  5. SPECULATIVE (on separate GPU stream, during NVMe wait):
     a. Weighted-sum partial FFN from cached experts only
     b. Add to residual -> approximate x for L+1
     c. rms_norm(x, L+1.ffn_norm) -> speculative h
     d. router_matvec(h, L+1.router) -> predicted expert IDs
     e. Submit NVMe prefetch for predicted L+1 experts
  6. Phase 2: FFN for NVMe/RAM experts as they arrive
  7. Weighted sum + residual

Layer L+1:
  - If predicted experts match actual: already loaded, no NVMe wait
  - If mismatch: CAR substitutes cached experts (no stall)
```

### Compute Cost
- RMS norm: ~0.01 ms
- F32 router matvec (256 x 3072): ~0.02 ms (vectorized kernel)
- Softmax + top-k: ~0.01 ms
- **Total: ~0.04 ms** (vs ~1.5 ms idle during NVMe wait)

### Expected Impact
With 86% prediction accuracy:
- ~6.9 of 8 experts per layer are prefetched correctly
- Remaining ~1.1 experts handled by CAR (cached substitutes, no stall)
- Effective NVMe wait time drops by up to 86%
- Combined with existing RAM/VRAM caching, most layers may have
  zero NVMe stalls

### Key Properties
- **Zero-penalty fallback:** Wrong predictions just trigger CAR, which is
  the existing fallback behavior. Speculation can only help, never hurt.
- **Negligible compute cost:** 0.04 ms prediction vs 1.5 ms idle time.
- **No additional memory:** Uses existing VRAM weight buffers and scratch
  space. Only needs a second GPU stream for concurrent execution.
- **No model changes:** Pure inference optimization, works with any
  pretrained MoE model.

## Applicability Beyond This Project

This technique should generalize to any MoE architecture where:
1. Expert weights are too large to fit in fast memory (VRAM/RAM)
2. Expert loading from slow storage (NVMe, network) is the bottleneck
3. The model uses residual connections around the MoE FFN block

The 86% accuracy on Qwen 3.5 suggests the residual stream dominates router
decisions in modern MoE transformers. This may hold for other MoE models
(Mixtral, DeepSeek-V3, etc.) but should be validated per-architecture.

The technique is complementary to other MoE optimizations:
- **Expert caching (VRAM/RAM LRU):** Reduces how many experts need loading
- **Cache-Aware Routing (CAR):** Provides zero-stall fallback for mispredictions
- **Speculative routing (this work):** Prefetches experts before they're needed

## Files

| File | Purpose |
|------|---------|
| `tools/svd_experiment.c` | Per-expert SVD spectrum analysis (negative result) |
| `tools/expert_cluster.c` | Expert clustering analysis (negative result) |
| `src/inference.c` | Routing prediction measurement (`QMOE_PREDICT_ROUTING=1`) |

## Reproducing

```bash
# Build (CPU path, no GPU needed for measurement)
make

# Run with prediction measurement
QMOE_PREDICT_ROUTING=1 ./qwen-moe generate --ram-cache 12000 \
  model.gguf store1.qmoe store2.qmoe store3.qmoe \
  -- "The capital of France is"

# Output shows per-layer prediction accuracy and running totals
# PREDICT L01: skip-attn=6/8  same-expert=0/8
# ...
# PREDICT: skip-attn=328/376 (87.2%)  same-expert=15/376 (4.0%)
```
