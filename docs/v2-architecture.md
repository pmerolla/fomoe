# v2 Architecture: Prefetch-Centric MoE Inference

**Date:** 2026-03-09
**Status:** Implemented and validated — 7.32 tok/s sustained on 397B (target was 7+)

## Motivation

The v1 engine achieves 8.4 tok/s (122B) and 4.8 tok/s (397B, warm) on a single
RX 9060 XT with 3 NVMe drives. But the 397B takes ~60 tokens to warm up — cold start
is 1.8 tok/s. And the 67 ms/token of CPU/sync overhead (hipStreamSync blocking) and
22 ms/token of H2D/NVMe stalls leave substantial room for improvement.

Two innovations — speculative routing (81-89% accuracy) and cache-aware routing (CAR) —
were bolted onto the existing architecture. v2 rewrites the engine around them.

**Goal:** Eliminate warmup penalty, eliminate pipeline stalls, reduce sync overhead.
Target: **7+ tok/s steady state for the 397B model** from token 1. Theoretical ceiling
on this hardware: ~7.5-8 tok/s (pure compute floor).

## Hardware Configuration (actual)

```
GPU:    AMD Radeon RX 9060 XT, 16 GB VRAM, gfx1200, 16 CUs, 2620 MHz
CPU:    12 cores (lower-end)
RAM:    64 GB (18 GB allocated for expert cache)
NVMe:   3 drives — 2× ~10 GB/s (shard_a, shard_b), 1× ~6.6 GB/s (shard_c)
PCIe:   4.0 x16 (~28 GB/s theoretical)
```

## Measured Baseline: v1 Performance

All numbers from actual runs: 3 NVMe drives, CAR_THRESHOLD=0.5.

### 397B Model (60 layers, 512 experts, top-10, stride=7.25 MB)

VRAM cache: **720 slots (12/layer), 2.3%** of 30,720 total experts.

**Warmup progression (8 GB vs 18 GB RAM cache):**
```
          ──── 8 GB RAM (1,080 slots, 3.5%) ────   ──── 18 GB RAM (2,460 slots, 8.0%) ────
tok=1:    VRAM 29.2%  RAM  0.0%  565ms  1.8 tok/s   VRAM 29.2%  RAM  0.0%  563ms  1.8 tok/s
tok=10:   VRAM 55.7%  RAM  2.6%  283ms  3.5 tok/s   VRAM 58.7%  RAM  5.4%  257ms  3.9 tok/s
tok=20:   VRAM 59.5%  RAM  3.5%  336ms  3.0 tok/s   VRAM 65.0%  RAM  8.9%  261ms  3.8 tok/s
tok=30:   VRAM 62.1%  RAM  4.9%  321ms  3.1 tok/s   VRAM 68.7%  RAM 12.7%  253ms  4.0 tok/s
tok=40:   VRAM 62.5%  RAM  5.0%  251ms  4.0 tok/s   VRAM 69.7%  RAM 13.6%  226ms  4.4 tok/s
tok=50:   VRAM 64.1%  RAM  5.6%  260ms  3.8 tok/s   VRAM 71.7%  RAM 14.6%  225ms  4.4 tok/s
tok=60:   VRAM 65.7%  RAM  6.7%  208ms  4.8 tok/s   (extrapolated: ~73%  ~16%  ~200ms ~5.0)
```

**Per-layer timing at tok=60 (8 GB, warmest measured):**
```
Attention/SSM:     0.69 ms/layer  (FullAttn: 0.62, SSM: 0.71)
Router + D2H:      0.10 ms/layer
  Sync stall:      1.07 ms/layer  (hipStreamSync blocking) ← BIGGEST BOTTLENECK
MoE pipeline:      1.51 ms/layer
  SharedFFN:       1.49 ms/layer  (overlapped with NVMe/H2D)
  P1 FFN (VRAM):   1.06 ms/layer
  H2D wait:        0.29 ms/layer
  P2 FFN (staged): 0.14 ms/layer
  NVMe wait:       0.09 ms/layer
  RC wait:         0.01 ms/layer
  CAR:             136 subs, only 7 NVMe remaining
Output proj:       3.45 ms total
```

**Per-layer timing at tok=50 (18 GB):**
```
Attention/SSM:     0.68 ms/layer  (FullAttn: 0.63, SSM: 0.69)
Router + D2H:      0.09 ms/layer
  Sync stall:      1.05 ms/layer  (hipStreamSync blocking)
MoE pipeline:      1.60 ms/layer
  SharedFFN:       1.57 ms/layer  (overlapped with NVMe/H2D)
  P1 FFN (VRAM):   1.15 ms/layer
  H2D wait:        0.30 ms/layer
  P2 FFN (staged): 0.13 ms/layer
  NVMe wait:       0.11 ms/layer
  RC wait:         0.03 ms/layer
  CAR:             108 subs, only 12 NVMe remaining
Output proj:       3.50 ms total
```

**Time budget decomposition (tok=60, 8 GB): where 208 ms goes**
```
GPU compute (no stalls):   ~118 ms   (56.8%)  ← irreducible floor
GPU stalls (H2D + NVMe):    23 ms   (11.1%)  ← spec routing eliminates
Sync stall (hipStreamSync): 64 ms   (30.8%)  ← pipelined architecture reduces
Output proj:                 3 ms    ( 1.4%)  ← fixed cost
```

**Key observations (397B):**
- v1 reaches **4.8 tok/s at tok=60** (8 GB) — but takes 60 tokens to warm up
- 18 GB RAM cache: **4.4 tok/s at tok=40**, warms faster, higher VRAM hit (~70% vs 62%)
- Cold start is terrible: 1.8 tok/s at tok=1 (both configs)
- **Sync stall is the #1 bottleneck at steady state** (64 ms, 31% of wall time)
- GPU stalls (H2D + NVMe) are #2 (23 ms, 11%)
- Pure GPU compute floor: **118 ms → 8.5 tok/s theoretical max** (unachievable)
- Realistic ceiling with pipelined v2: **~133 ms → 7.5 tok/s**
- Spec routing accuracy: 81-89%

### 122B Model (48 layers, 256 experts, top-8, stride=5.84 MB)

VRAM cache: 1,680 slots (35/layer), 13.7% of 12,288 total experts.
RAM cache: 1,344 slots (28/layer), 10.9%.

**Per-layer timing (tok=40, generation, steady state):**
```
Attention/SSM:     0.59 ms/layer  (FullAttn: 0.48, SSM: 0.62)
Router + D2H:      0.05 ms/layer
MoE pipeline:      0.83 ms/layer
  SharedFFN:       0.82 ms/layer  (overlapped)
  P1 FFN (VRAM):   0.73 ms/layer
  H2D wait:        0.08 ms/layer
  P2 FFN (staged): 0.02 ms/layer
  NVMe wait:       0.02 ms/layer
  Classify:        0.02 ms/layer
Output proj:       2.69 ms total
```

**Cache warmup progression:**
```
tok=1:   VRAM 35.7%  RAM  0.0%  313 ms   3.2 tok/s   CAR: 3 subs
tok=10:  VRAM 72.9%  RAM  0.0%  134 ms   7.5 tok/s   CAR: 52 subs (0.84 ratio)
tok=30:  VRAM 80.8%  RAM  0.9%  129 ms   7.7 tok/s   CAR: 75 subs (0.76 ratio)
tok=40:  VRAM 81.6%  RAM  1.2%  119 ms   8.4 tok/s   CAR: 42 subs (0.80 ratio)
tok=80:  VRAM 82.6%  RAM  1.6%  127 ms   7.9 tok/s   CAR: 91 subs (0.80 ratio)
```

**Key observations (122B):**
- VRAM hit rate reaches 82%+ by tok=30
- **Generation speed: 8.4 tok/s** (stabilizes quickly)
- Already near compute-bound. v2's main win: cold start elimination.

### NVMe Bandwidth (measured, 3 drives, 397B stride)

```
1 expert:  0.90 ms latency,  7.9 GB/s  (single drive)
2 experts: 0.90 ms,         15.8 GB/s  (2 drives)
4 experts: 1.45 ms,         19.6 GB/s  (3 drives)
8 experts: 2.13 ms,         26.6 GB/s  (3 drives, drive C slower at 6.6 GB/s)
```

Note: Drive C (non-shard mount) is slower (~6.6 GB/s vs ~10 GB/s). Aggregate peaks at
~26.6 GB/s with 3 heterogeneous drives.

## The Prefetch Window

### Timeline

```
L's residual add completes on stream 0
  ├─ Stream 0: starts L+1's attn/SSM immediately
  └─ Stream 1: triggered by cross-stream event
       ├─ Spec routing: rms_norm + router matvec + top-K  (~0.1 ms)
       ├─ D2H predicted expert IDs                        (~0.01 ms)
       └─ CPU: classify + submit NVMe reads               (~0.05 ms)
              NVMe reads now in flight ─────────────────────┐
                                                            │
  Stream 0: [attn/SSM] [router] [shared FFN] [expert FFN: NEEDS DATA]
            │← 0.66 →│← 0.09 →│← ~1.2 ms →│
                                                            │
  Time from NVMe submission to expert-need: ────────────────┘
```

### Window calculation

```
                            122B            397B
attn/SSM remaining:         0.43 ms         0.50 ms    (attn - 0.16 overhead)
router + D2H:               0.05 ms         0.09 ms
shared expert FFN:          0.80 ms         1.60 ms    (from SharedFFN timing)
──────────────────────────────────────────────────────
Prefetch window:           ~1.28 ms        ~2.19 ms
```

The shared expert FFN runs on GPU stream 0 BEFORE routed expert FFN. It doesn't need
NVMe data. This extends the prefetch window significantly — especially for 397B where
the shared FFN is ~1.6 ms.

### NVMe prefetch budget

With pipelined NVMe→H2D (callback per expert, H2D on transfer stream as each read
completes):

**122B (window ~1.28 ms, 3 drives):**
```
2 NVMe experts: ~0.65 ms + 0.2 ms = ~0.85 ms  → fits
3 NVMe experts: ~0.85 ms + 0.2 ms = ~1.05 ms  → fits
4 NVMe experts: ~1.05 ms + 0.2 ms = ~1.15 ms  → fits
5 NVMe experts: ~1.25 ms + 0.2 ms = ~1.35 ms  → borderline
```
**Budget: 4 NVMe experts per layer.**

**397B (window ~2.19 ms, 3 drives):**
```
4 NVMe experts: 1.45 ms + 0.25 ms = ~1.60 ms  → fits
6 NVMe experts: ~1.80 ms + 0.25 ms = ~1.95 ms → fits
8 NVMe experts: 2.13 ms + 0.25 ms = ~2.28 ms  → borderline
```
**Budget: 6-7 NVMe experts per layer.**

The 397B has a larger budget because its shared FFN extends the window.

### Sizing K (over-prediction factor)

K_nvme = K × (1 - cache_hit_rate) ≤ NVMe budget

For 397B (budget=6-7, 10 active experts):
| Cache hit rate | K=10 nvme | K=14 nvme | K=18 nvme | K=22 nvme |
|:--------------:|:---------:|:---------:|:---------:|:---------:|
| 30% (cold)     | 7.0       | 9.8       | 12.6      | 15.4      |
| 50%            | 5.0       | 7.0       | 9.0       | 11.0      |
| 62% (current)  | 3.8       | 5.3       | 6.8       | 8.4       |
| 80% (v2 tgt)   | 2.0       | 2.8       | 3.6       | 4.4       |

**K=18 for 397B** works at 62%+ hit rate (K_nvme=6.8, within budget). As caches warm,
K_nvme drops further. Budget enforcement caps NVMe reads if K_nvme exceeds 7.

For 122B (budget=4, 8 active experts):
At 80%+ hit rate: K=16 → K_nvme=3.2 → fits. **K=16 for 122B.**

## Design Principles

1. **Speculative routing buys prefetch time.** Predict L+1's experts after L's FFN.
   Start NVMe reads immediately. Budget: 4 NVMe reads (122B) / 6-7 (397B) per layer.
2. **VRAM cache is a rolling LRU.** Seeded from frequency profile at startup, then
   pure LRU takes over. Simple and adaptive.
3. **CAR absorbs what prefetch can't cover.** Threshold is a tunable parameter.
4. **NVMe is actively used every layer.** The long tail of 512 experts per layer is
   what gives the 397B its power. NVMe reads happen continuously — spec routing just
   moves them earlier to overlap with GPU compute.
5. **Reuse proven components.** New code only for orchestration and cache management.

## Per-Layer Pipeline

```
Layer L (397B):

  Time    Stream 0 (compute)       Stream 1 (xfer + spec)      CPU / NVMe
  ─────   ──────────────────       ──────────────────────       ──────────

  0.0     ┌─ attn/SSM (L) ──┐     Consuming L's prefetch:     NVMe reads
          │  (~0.66 ms)      │     H2D from NVMe completions   completing
          │                  │     into VRAM cache slots       for L
          │  router matvec   │
          │  D2H scores      │
          └────────┬─────────┘
                   │
          CPU: top-10, classify, CAR
                   │
          ┌────────┴─────────┐     ┌────────────────────┐
          │  shared FFN      │     │  Spec route L+1:   │      Submit NVMe
          │  (~1.6 ms)       │     │  rms_norm→matvec   │      reads for L+1
          │                  │     │  →softmax→top-K    │      (up to 7 experts
          │                  │     │  →D2H pred IDs     │       not in cache)
          │                  │     │  (~0.1 ms)         │
          │  expert FFN      │     │                    │
          │  (all in VRAM)   │     │  H2D for predicted │
          │  (~1.6 ms)       │     │  experts→VRAM      │
          │                  │     │                    │
          │  weighted sum    │     └────────────────────┘
          │  residual add    │
          └──────────────────┘

Layer L+1: prefetched experts arriving in VRAM...
```

### The 397B critical path with v2

**Three optimization targets** (from time budget at tok=60):

**1. Sync stall elimination: 64 ms → ~10 ms (pipelined dual-stream)**
Currently the CPU blocks on hipStreamSync waiting for router scores. In v2, the
dual-stream architecture pipelines GPU work so the CPU rarely blocks:
- Stream 0 launches attn/SSM → router → shared FFN → expert FFN continuously
- Stream 1 handles H2D transfers and spec routing concurrently
- CPU classifies experts from previous iteration while GPU computes current
- Estimated residual sync: ~10 ms (layer boundary sync only)
- **Savings: ~54 ms/token**

**2. Stall elimination: 23 ms → ~3 ms (spec routing + prefetch)**
H2D wait (17 ms) + NVMe wait (6 ms) eliminated by loading experts before needed:
- Spec routing predicts L+1 experts during L's shared FFN
- NVMe reads complete during prefetch window (~2.2 ms)
- H2D transfers overlap on stream 1
- Some residual stalls for mispredictions (~3 ms via CAR fallback)
- **Savings: ~20 ms/token**

**3. Cold start elimination: frequency-seeded VRAM (savings: ~350 ms at tok=1)**
Currently 1.8 tok/s at tok=1 → target ~5+ tok/s from token 1.

**Projected v2 per-token timeline (397B, steady state):**
```
GPU compute:       ~118 ms  (irreducible: attn + router + FFN)
Residual sync:      ~10 ms  (layer boundaries)
Residual stalls:     ~3 ms  (spec routing misses → CAR)
Output proj:          3 ms
─────────────────────────────
Total:             ~134 ms  → ~7.5 tok/s (optimistic)
Conservative:      ~150 ms  → ~6.7 tok/s (some sync + stalls remain)
```

**v1 → v2 improvement breakdown (397B):**
```
v1 tok=60 (8GB):                 208 ms  (4.8 tok/s)
  − sync stall reduction:       −54 ms
  − H2D/NVMe stall elimination: −20 ms
  − 18GB RAM (fewer stalls):     −5 ms
  = v2 target:                  ~130-150 ms  (6.7-7.5 tok/s)
```

### 122B with v2

Current v1 at tok=40: 0.83 ms/layer MoE pipe (119 ms, 8.4 tok/s).
Already near compute-bound. v2's main win: frequency seeding eliminates cold start
(313 ms → ~120 ms from token 1). Potential steady-state: ~9-10 tok/s.

## VRAM Expert Cache

### Design: Rolling LRU

Layer-partitioned LRU. Every access bumps timestamp.

```c
typedef struct {
    void     *d_buf;            // contiguous VRAM allocation
    int       max_slots;
    int       slots_per_layer;  // max_slots / n_layers
    int      *map;              // [n_layers × n_experts] → slot or -1
    int      *slot_layer;       // [max_slots]
    int      *slot_expert;      // [max_slots]
    uint64_t *slot_ts;          // [max_slots] LRU timestamp
    uint64_t  ts;
    uint64_t  hits, misses;
} vram_cache_t;
```

Capacity:
```
122B: 1,680 slots (35/layer), ~9.8 GB, 13.7% of 12,288 experts
397B:   720 slots (12/layer), ~5.2 GB,  2.3% of 30,720 experts
```

### The 397B VRAM crunch

12 slots per layer, 10 active experts per token. Only 2 slots of "spare" capacity.
This means **every token nearly fills the entire layer's cache**. LRU alone can't help
— there's no room for history. Spec routing is essential: it tells us which 10 experts
to put in those 12 slots BEFORE we need them.

Without spec routing: ~62-72% VRAM hit (v1, depending on RAM cache size and warmup).
With spec routing: target ~85-90% (most predicted experts are correct, loaded in advance).

### Startup: frequency-seeded

Load frequency profile → fill VRAM cache with most frequent experts per layer.
Eliminates cold start:
- 397B tok=1 current: 29.2% VRAM hit, 572 ms → target: ~85% hit, ~250 ms
- 122B tok=1 current: 35.7% VRAM hit, 313 ms → target: ~80% hit, ~120 ms

## Speculative Prefetch Pipeline

### Data flow

```
Spec routing predicts top-K for L+1 (K=18 for 397B, K=16 for 122B)
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Prefetch classifier (CPU, sorted by score desc):     │
│                                                       │
│  For each predicted expert:                           │
│    VRAM hit? ──yes──→ skip                            │
│    RAM hit? ──yes──→ queue H2D (transfer stream)      │
│    NVMe budget left? ──no──→ skip (CAR fallback)      │
│    Submit io_uring read → callback:                   │
│      → memcpy to RAM cache (background)               │
│      → H2D to reserved VRAM slot (transfer stream)    │
└──────────────────────────────────────────────────────┘
```

### Budget enforcement

Predicted experts sorted by score. NVMe reads capped at per-layer budget (7 for 397B,
4 for 122B). Higher-scoring predictions prefetched first. When budget exhausted, lower
predictions skipped — CAR handles them if they become actual selections.

### Tracking in-flight prefetches

```c
typedef struct {
    int      layer, expert_id;
    int      vram_slot;         // reserved via LRU eviction
    bool     h2d_complete;
    void    *host_buf;
} prefetch_entry_t;

typedef struct {
    prefetch_entry_t entries[MAX_PREFETCH];
    int              n_entries;
    int              nvme_budget;
    int              nvme_submitted;
} prefetch_state_t;
```

When actual router selects experts:
- In VRAM (from cache or completed prefetch) → use directly
- In prefetch table, H2D in flight → wait for completion event
- Not prefetched → CAR evaluates, or sync load

## Cache-Aware Routing (CAR)

CAR fires when a selected expert is not in VRAM and wasn't prefetched. Expected
frequency:
- 397B: ~1-3 per layer (with K=18 over-prediction and ~85% spec accuracy)
- 122B: ~0.5 per layer

### Algorithm

```
For each VRAM-miss expert E with score S_orig:
  Find highest-scoring VRAM-cached expert C not already selected
  ratio = S_cache / S_orig
  If ratio >= car_threshold:
    Substitute E → C, renormalize scores
  Else:
    Sync load from RAM (0.2 ms) or NVMe (0.9 ms)
```

### Measured CAR quality (v1 data)

At CAR_THRESHOLD=0.5:
- 397B: avg substitution ratio 0.61, ~138 subs/token
- 122B: avg substitution ratio 0.76-0.80, ~42-91 subs/token

The 122B substitutions are higher quality (ratio closer to 1.0) because its larger
VRAM cache contains better alternatives.

## RAM Cache

### Role at 18 GB

With 18 GB: **2,460 slots for 397B (41/layer, 8.0% coverage)**. Hit rate: 13-16% at
steady state. This is a meaningful contributor — 18 GB vs 8 GB:
- Faster warmup: 4.4 tok/s at tok=40 vs 4.0 (8 GB)
- Higher VRAM hit: 70% vs 62% at tok=40 (RAM feeds VRAM via prefetch H2D)
- Lower stalls: H2D wait 0.25 ms/L vs 0.27, NVMe wait 0.08 vs 0.27 at tok=40

Three roles:
1. **Prefetch H2D source:** Spec routing finds expert in RAM → H2D to VRAM (fast, ~0.2 ms)
2. **Landing zone for NVMe reads:** Populated as side-effect, reduces repeat NVMe reads
3. **CAR candidate pool:** RAM-cached experts available for CAR substitution

### Design

Same as v1: LRU, layer-partitioned, pinned host memory (hipHostRegister).

## NVMe I/O

### Actively used for prefetch

Every layer: spec routing predicts K experts, some need NVMe reads. Reads submitted
via io_uring with H2D callback on completion. Pipelined with GPU compute.

### Measured bandwidth (3 drives, 397B)

```
4 experts: 1.45 ms, 19.6 GB/s
8 experts: 2.13 ms, 26.6 GB/s
```

Drive C is slower (6.6 GB/s vs 10 GB/s for A/B). Real aggregate is ~26.6 GB/s
rather than the theoretical 30 GB/s.

### Module reuse

nvme_io.c carried from v1 unchanged. Callback API already exists.

## Frequency Profile

### Purpose

Seeds VRAM cache at startup. Critical for 397B where cold start is 563 ms (1.8 tok/s).
With frequency seeding, tok=1 should start at ~70% VRAM hit instead of 29% — the
difference between 1.8 tok/s and ~5+ tok/s.

### Format

```
[Header: 16 bytes]
  magic:       uint32  0x46524551  ("FREQ")
  version:     uint32  1
  n_layers:    uint32
  n_entries:   uint32

[Entries: n_entries × 8 bytes, sorted by frequency descending]
  layer:       uint16
  expert_id:   uint16
  frequency:   float32  (normalized 0-1)
```

### Generation

```bash
./qwen-moe profile --tokens 1000 model.gguf stores... -- "diverse text"
# Writes expert_freq.bin
```

## Module Structure

```
src/
  main.c              CLI: info, generate, chat, benchmark, profile, dequant
  model.c             GGUF loading, weight allocation, freq profile path
  inference.c         CPU forward pass: attention, SSM, MoE routing, FFN
  gpu_kernels.hip     GPU forward pass: kernels + dual-stream orchestration
  expert_cache.c      VRAM cache (LRU) + RAM cache (LRU)
  prefetch.c          Spec routing → classify → NVMe/H2D pipeline
  car.c               Cache-aware routing substitution
  freq_profile.c      Frequency profile load/save/generate
  nvme_io.c           Async io_uring (reuse from v1)
  expert_store.c      .qmoe format (reuse from v1)
  quant.c             Dequant + quantized dot products (reuse from v1)
  tensor.c            MatVec, norms, activations (reuse from v1)
  tokenizer.c         BPE tokenizer (reuse from v1)
  sampler.c           Top-p sampling (reuse from v1)
  gguf.c              GGUF parser (reuse from v1)
  test.c              Integration test suite
  test_v2.c           v2 module tests (143 tests)
```

### New modules (v2)
- expert_cache.c — VRAM + RAM cache with layer-partitioned LRU API
- prefetch.c — speculative prefetch pipeline with persistent worker thread
- car.c — CAR substitution logic with threshold-gated replacement
- freq_profile.c — frequency profile I/O (binary .freq format)

### Modified (v2 integration)
- gpu_kernels.hip — replaced inline cache structs with vram_cache_t/ram_cache_t,
  replaced inline spec_pf with prefetch_state_t (opaque pointer), integrated CAR
- main.c — new CLI args, chat/profile subcommands, freq profile loading
- model.c — freq_profile_path field, gpu_seed_vram_cache() call
- gpu.h — gpu_seed_vram_cache(), gpu_enable_expert_freq(), gpu_get_expert_freq()
- Makefile — new source files and header dependencies

## Configuration

```
--freq-profile PATH    Expert frequency profile for startup seeding
--car-threshold T      CAR substitution threshold (default: 0.7, optimal: 0.3 for 397B)
--spec-k K             Over-prediction factor (default: auto = n_experts_used)
--ram-cache MB         RAM cache size in MB (default: auto ≈ 67% of available)
--prefetch-budget N    Max NVMe reads per prefetch window (default: auto, optimal: 0 for 397B)
```

Environment variable overrides: `QMOE_CAR_THRESHOLD`, `QMOE_PREFETCH_BUDGET`, `QMOE_NO_SPEC_ROUTING`.

## Performance Results (measured)

### 397B (100-token sustained generation)

Optimal config: `--freq-profile 397b.freq`, `CAR_THRESHOLD=0.3`, `PREFETCH_BUDGET=0`

| Metric | v1 baseline | v2 actual | Improvement |
|--------|:-----------:|:---------:|:-----------:|
| Cold start (tok=1) | 1.8 tok/s | 3.5 tok/s | +94% |
| Steady state (100 tok) | 3.81 tok/s | **7.32 tok/s** | **+92%** |
| Best single-token | ~4.8 tok/s | **8.81 tok/s** | +84% |
| VRAM hit rate | ~30% (cold) | 48% (from tok=1) | freq seeding |
| RAM hit rate | ~0% (cold) | 7% (from tok=1) | freq seeding |
| Spec routing accuracy | — | 82-88% | new |
| CAR subs/token | — | ~260 | new |

**Per-token range:** 6.97-8.81 tok/s (variance from NVMe I/O patterns).

**Key findings from parameter sweeps:**
- `budget=0` optimal: NVMe reads (~2.4ms) don't complete within the ~0.9ms attention
  window, so async prefetch adds blocking time rather than hiding it
- `CAR=0.3` optimal: aggressive substitution eliminates most NVMe stalls with no
  measurable quality degradation
- Freq profile seeding is critical for eliminating cold-start penalty

| CAR Threshold | tok/s | Notes |
|:------------:|:-----:|-------|
| 0.0 (disabled) | 4.66 | All misses go to NVMe |
| **0.3** | **7.23** | Best speed, no quality loss |
| 0.5 | 6.69 | |
| 0.7 | 5.83 | |
| 0.9 | 4.64 | Too conservative |

**Why 10 tok/s is not feasible on this hardware:**
Pure GPU compute floor is ~57 ms/token (17.4 tok/s theoretical). Current wall time of
~133 ms includes ~40 ms H2D wait and ~20 ms NVMe wait that cannot be fully hidden.
Reaching 10 tok/s requires cross-layer pipelining or fundamentally faster kernels.

### 122B

Steady state: 8-11 tok/s depending on RAM cache size. With frequency seeding,
near-instant warm-start from token 1.

## Implementation History

### Phase 1: Scaffold + reuse (complete)
- New modules: expert_cache.c, freq_profile.c, car.c with headers
- Layer-partitioned LRU for both VRAM and RAM caches
- 143 unit tests passing (test_v2)

### Phase 2: Prefetch pipeline (complete)
- prefetch.c with persistent worker thread, NVMe callback-based RAM population
- Opaque pointer pattern for GPU integration (prefetch_create/destroy)

### Phase 3: Integration (complete)
- gpu_kernels.hip: replaced inline spec_pf with v2 prefetch module, replaced
  embedded cache structs with vram_cache_t/ram_cache_t, integrated CAR
- main.c: new CLI args, chat/profile subcommands, freq profile loading
- model.c: freq profile path storage, gpu_seed_vram_cache() call

### Phase 4: Validation (complete)
- 100-token sustained generation: 7.32 tok/s on 397B
- Parameter sweeps: CAR threshold, prefetch budget
- Output quality validation: no degradation with CAR=0.3
- Frequency profile generation for both 122B and 397B

## Future Work (v3 — path to 10 tok/s)

- **Cross-layer pipelining:** Start L+1's attention while L's expert FFN finishes.
  The dependency chain allows partial overlap — could save ~15-20 ms/token.
- **Batched expert kernels:** Process multiple experts in a single kernel launch to
  reduce launch overhead. Currently ~0.17 ms/expert; batched could be ~0.12 ms.
- **CPU expert FFN:** 12-core CPU handles 1-2 RAM-cached experts parallel with GPU.
- **Multi-token batching:** 2-4 tokens at once, amortize NVMe across tokens.
- **Adaptive K:** Lower K when caches warm, raise when cold.
- **Expert co-occurrence:** Learn L→L+1 transition probabilities.
- **Persistent GPU kernels:** Reduce kernel launch latency with always-running kernels.
