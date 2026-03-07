# v2 RAM Cache Size Benchmarks

**Date:** 2026-03-09
**Config:** v2 optimal — `CAR_THRESHOLD=0.3`, `PREFETCH_BUDGET=0`, freq profile seeding
**GPU:** AMD Radeon RX 9060 XT (16 GB VRAM, gfx1200, RDNA 4)
**CPU:** AMD (X870E Taichi Lite), 12 cores
**RAM:** 64 GB DDR5
**NVMe:** 3 drives (2× ~10 GB/s + 1× ~6.6 GB/s)
**Generation:** 60 tokens, `--no-eos`
**Prompt:** `"The key insight behind mixture-of-experts models is"`

---

## 397B Model (Qwen3.5-397B-A17B, Q4_K_M)

60 layers, 512 experts/layer, 10 active, stride=7.25 MB, 30,720 total experts.
VRAM cache: 720 slots (12/layer, 5.2 GB), 2.3% of total.

| RAM Cache | Slots | Coverage | Gen Speed | Prompt Speed | VRAM$ Hit | RAM$ Hit |
|-----------|-------|----------|-----------|--------------|-----------|----------|
| **18 GB** | 2,460 (41/layer) | 8.0% | **7.59 tok/s** | 6.5 tok/s | ~40% | ~3% |
| **32 GB** | 4,380 (73/layer) | 14.3% | **7.56 tok/s** | 6.9 tok/s | ~48% | ~7% |
| **auto (33 GB)** | 4,620 (77/layer) | 15.0% | **7.35 tok/s** | 6.6 tok/s | ~48% | ~7% |

### Per-Token Progression (18 GB)

| Token | Wall (ms) | tok/s | Notes |
|-------|-----------|-------|-------|
| 1 | 303 | 3.30 | Cold-start (freq seeded) |
| 10 | 135 | 7.42 | Warming up |
| 30 | 133 | 7.50 | |
| 50 | 138 | 7.25 | |
| 80 | 129 | 7.74 | Steady state |

### Per-Token Progression (32 GB)

| Token | Wall (ms) | tok/s | Notes |
|-------|-----------|-------|-------|
| 1 | 266 | 3.76 | Cold-start (freq seeded) |
| 10 | 129 | 7.75 | Warming up |
| 30 | 135 | 7.43 | |
| 60 | 122 | 8.22 | |
| 80 | 110 | 9.11 | Best single-token |

### Analysis

For 397B with v2, **RAM cache size has minimal impact** on steady-state speed:
- 18 GB → 7.59 tok/s
- 32 GB → 7.56 tok/s
- Auto (33 GB) → 7.35 tok/s

This is because CAR=0.3 aggressively substitutes NVMe misses with VRAM-cached experts,
making RAM cache hits less critical than in v1. The VRAM cache (720 slots, 2.3%) is the
main performance driver. A larger RAM cache does help with cold-start (3.76 vs 3.30 tok/s
at tok=1) and peak single-token performance (9.11 vs 7.74).

**Recommendation:** 18 GB is sufficient for 397B. Use the extra RAM for other workloads.

---

## 122B Model (Qwen3.5-122B-A10B, Q4_K_M)

48 layers, 256 experts/layer, 8 active, stride=5.84 MB, 12,288 total experts.
VRAM cache: 1,680 slots (35/layer, 9.8 GB), 13.7% of total.

| RAM Cache | Slots | Coverage | Gen Speed | Prompt Speed | Notes |
|-----------|-------|----------|-----------|--------------|-------|
| **18 GB** | 3,072 (64/layer) | 25.0% | **19.39 tok/s** | 14.6 tok/s | |
| **32 GB** | 5,472 (114/layer) | 44.5% | **19.52 tok/s** | 14.8 tok/s | |
| **auto (37 GB)** | 6,528 (136/layer) | 53.1% | **19.66 tok/s** | — | |

### Per-Token Progression (18 GB)

| Token | Wall (ms) | tok/s | Notes |
|-------|-----------|-------|-------|
| 1 | 74 | 13.53 | Cold-start (freq seeded) |
| 10 | 67 | 15.05 | Warming up |
| 30 | 51 | 19.57 | Near peak |
| 40 | 51 | 19.57 | |
| 70 | 44 | 22.68 | Best single-token |

### Per-Token Progression (32 GB)

| Token | Wall (ms) | tok/s | Notes |
|-------|-----------|-------|-------|
| 1 | 75 | 13.33 | Cold-start (freq seeded) |
| 10 | 67 | 14.91 | Warming up |
| 30 | 50 | 20.04 | Near peak |
| 70 | 48 | 20.98 | Best single-token |

### Analysis

For 122B, **RAM cache size also has minimal impact** — the model is nearly GPU compute-bound:
- 18 GB → 19.39 tok/s
- 32 GB → 19.52 tok/s
- Auto (37 GB) → 19.66 tok/s

The 122B model's VRAM cache (1,680 slots, 13.7%) provides much better coverage than
397B's (720 slots, 2.3%), so most experts are served from VRAM directly. Peak
single-token reaches 22.7 tok/s.

**Recommendation:** 18 GB is more than sufficient for 122B.

---

## Summary

| Model | 18 GB | 32 GB | Auto | Best Single Token |
|-------|-------|-------|------|-------------------|
| **397B** | 7.59 tok/s | 7.56 tok/s | 7.35 tok/s | 9.11 tok/s |
| **122B** | 19.39 tok/s | 19.52 tok/s | 19.66 tok/s | 22.68 tok/s |

With v2's CAR + freq seeding, RAM cache size beyond 18 GB provides diminishing returns.
The VRAM cache + CAR substitution dominate performance. 18 GB is the sweet spot for
both models on this hardware.
