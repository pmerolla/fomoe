#include "expert_cache.h"
#include "freq_profile.h"
#include "car.h"
#include "prefetch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// ---- VRAM cache tests ----

static void test_vram_cache_basic(void) {
    vram_cache_t c;
    // 4 layers, 16 experts, 8 slots (2 per layer)
    CHECK(vram_cache_init(&c, 4, 16, 8, 1024) == 0, "vram_cache_init");
    CHECK(c.slots_per_layer == 2, "slots_per_layer == 2");
    CHECK(c.max_slots == 8, "max_slots == 8");

    // Empty cache: all lookups miss
    CHECK(vram_cache_lookup(&c, 0, 5) == -1, "empty cache miss");
    CHECK(c.misses == 1, "misses == 1");

    // Alloc slot in layer 0 for expert 5
    int slot = vram_cache_alloc_slot(&c, 0, 5);
    CHECK(slot >= 0 && slot < 2, "alloc slot in layer 0 range");

    // Now lookup should hit
    int found = vram_cache_lookup(&c, 0, 5);
    CHECK(found == slot, "lookup hits after alloc");
    CHECK(c.hits == 1, "hits == 1");

    // Alloc second slot in layer 0
    int slot2 = vram_cache_alloc_slot(&c, 0, 10);
    CHECK(slot2 >= 0 && slot2 < 2, "second slot in layer 0");
    CHECK(slot2 != slot, "different slot");

    // Both should be findable
    CHECK(vram_cache_lookup(&c, 0, 5) == slot, "expert 5 still cached");
    CHECK(vram_cache_lookup(&c, 0, 10) == slot2, "expert 10 cached");

    // Alloc third slot in layer 0 — must evict LRU
    // Expert 5 was looked up more recently than expert 10,
    // Wait — both were just looked up. Let's touch expert 5 to make 10 the LRU.
    vram_cache_touch(&c, slot); // bump expert 5
    int slot3 = vram_cache_alloc_slot(&c, 0, 15);
    CHECK(slot3 == slot2, "evicts LRU (expert 10)");
    CHECK(vram_cache_lookup(&c, 0, 10) == -1, "expert 10 evicted");
    CHECK(vram_cache_lookup(&c, 0, 15) == slot3, "expert 15 cached");

    // Layer isolation: layer 1 has separate slots
    int slot_l1 = vram_cache_alloc_slot(&c, 1, 3);
    CHECK(slot_l1 >= 2 && slot_l1 < 4, "layer 1 slot in range [2,4)");
    CHECK(vram_cache_lookup(&c, 1, 3) == slot_l1, "layer 1 lookup");
    CHECK(vram_cache_lookup(&c, 0, 3) == -1, "layer 0 doesn't see layer 1");

    vram_cache_free(&c);
    fprintf(stderr, "  vram_cache_basic: OK\n");
}

static void test_vram_cache_lru_eviction(void) {
    vram_cache_t c;
    // 1 layer, 256 experts, 4 slots
    vram_cache_init(&c, 1, 256, 4, 1024);

    // Fill all 4 slots
    for (int i = 0; i < 4; i++)
        vram_cache_alloc_slot(&c, 0, i);

    // Access experts 1, 2, 3 (not 0) — expert 0 becomes LRU
    vram_cache_lookup(&c, 0, 1);
    vram_cache_lookup(&c, 0, 2);
    vram_cache_lookup(&c, 0, 3);

    // Alloc new expert should evict expert 0
    int slot = vram_cache_alloc_slot(&c, 0, 100);
    CHECK(vram_cache_lookup(&c, 0, 0) == -1, "expert 0 evicted (LRU)");
    CHECK(vram_cache_lookup(&c, 0, 100) == slot, "expert 100 cached");

    // Experts 1, 2, 3 still present
    CHECK(vram_cache_lookup(&c, 0, 1) >= 0, "expert 1 survives");
    CHECK(vram_cache_lookup(&c, 0, 2) >= 0, "expert 2 survives");
    CHECK(vram_cache_lookup(&c, 0, 3) >= 0, "expert 3 survives");

    vram_cache_free(&c);
    fprintf(stderr, "  vram_cache_lru_eviction: OK\n");
}

// ---- RAM cache tests ----

static void test_ram_cache_basic(void) {
    ram_cache_t c;
    CHECK(ram_cache_init(&c, 2, 32, 6, 512) == 0, "ram_cache_init");
    CHECK(c.slots_per_layer == 3, "ram slots_per_layer == 3");

    int slot = ram_cache_alloc_slot(&c, 0, 7);
    CHECK(slot >= 0 && slot < 3, "ram alloc in range");
    CHECK(ram_cache_lookup(&c, 0, 7) == slot, "ram lookup hit");
    CHECK(ram_cache_lookup(&c, 1, 7) == -1, "ram layer isolation");

    ram_cache_free(&c);
    fprintf(stderr, "  ram_cache_basic: OK\n");
}

// ---- Frequency profile tests ----

static void test_freq_profile_roundtrip(void) {
    const char *path = "/tmp/test_freq_profile.bin";

    // Create profile from counts
    // 2 layers, 8 experts each
    uint32_t counts[16] = {
        // layer 0: experts 0-7
        10, 50, 30, 0, 5, 0, 0, 20,
        // layer 1: experts 0-7
        0, 0, 40, 60, 0, 10, 0, 0,
    };

    freq_profile_t *fp = freq_profile_from_counts(counts, 2, 8, 100);
    CHECK(fp != NULL, "freq_profile_from_counts");
    CHECK(fp->n_layers == 2, "fp n_layers == 2");
    CHECK(fp->n_entries == 8, "fp n_entries == 8 (8 non-zero)");

    // Entries should be sorted by frequency descending
    CHECK(fp->entries[0].frequency >= fp->entries[1].frequency, "sorted desc");

    // Save and reload
    CHECK(freq_profile_save(path, fp) == 0, "freq_profile_save");
    freq_profile_t *fp2 = freq_profile_load(path);
    CHECK(fp2 != NULL, "freq_profile_load");
    CHECK(fp2->n_entries == fp->n_entries, "roundtrip n_entries");
    CHECK(fp2->n_layers == fp->n_layers, "roundtrip n_layers");

    // Verify entries match
    for (int i = 0; i < fp->n_entries; i++) {
        CHECK(fp2->entries[i].layer == fp->entries[i].layer, "roundtrip layer");
        CHECK(fp2->entries[i].expert_id == fp->entries[i].expert_id, "roundtrip expert_id");
        CHECK(fabsf(fp2->entries[i].frequency - fp->entries[i].frequency) < 1e-6f,
              "roundtrip frequency");
    }

    // Test top_for_layer
    freq_entry_t top[4];
    int n = freq_profile_top_for_layer(fp, 0, top, 4);
    CHECK(n == 4, "layer 0 has 4 non-zero experts");
    CHECK(top[0].frequency >= top[1].frequency, "layer 0 sorted");
    CHECK(top[0].expert_id == 1, "layer 0 top expert is 1 (count=50)");

    n = freq_profile_top_for_layer(fp, 1, top, 4);
    CHECK(n == 3, "layer 1 has 3 non-zero experts");
    CHECK(top[0].expert_id == 3, "layer 1 top expert is 3 (count=60)");

    freq_profile_free(fp);
    freq_profile_free(fp2);
    remove(path);
    fprintf(stderr, "  freq_profile_roundtrip: OK\n");
}

static void test_freq_profile_seed(void) {
    // Create a profile and seed a VRAM cache
    uint32_t counts[32] = {}; // 2 layers, 16 experts
    counts[0 * 16 + 3] = 100;  // layer 0, expert 3
    counts[0 * 16 + 7] = 80;   // layer 0, expert 7
    counts[0 * 16 + 1] = 60;   // layer 0, expert 1
    counts[1 * 16 + 5] = 90;   // layer 1, expert 5
    counts[1 * 16 + 9] = 70;   // layer 1, expert 9

    freq_profile_t *fp = freq_profile_from_counts(counts, 2, 16, 100);
    CHECK(fp != NULL, "seed: create profile");

    vram_cache_t c;
    vram_cache_init(&c, 2, 16, 4, 1024); // 2 slots per layer

    int seeded = vram_cache_seed(&c, fp);
    CHECK(seeded == 4, "seeded 4 slots (2 per layer)");

    // Layer 0: top 2 are expert 3 (freq=1.0) and expert 7 (freq=0.8)
    CHECK(vram_cache_lookup(&c, 0, 3) >= 0, "seed: layer 0 expert 3 cached");
    CHECK(vram_cache_lookup(&c, 0, 7) >= 0, "seed: layer 0 expert 7 cached");
    CHECK(vram_cache_lookup(&c, 0, 1) == -1, "seed: layer 0 expert 1 not cached (3rd)");

    // Layer 1: top 2 are expert 5 and expert 9
    CHECK(vram_cache_lookup(&c, 1, 5) >= 0, "seed: layer 1 expert 5 cached");
    CHECK(vram_cache_lookup(&c, 1, 9) >= 0, "seed: layer 1 expert 9 cached");

    freq_profile_free(fp);
    vram_cache_free(&c);
    fprintf(stderr, "  freq_profile_seed: OK\n");
}

// ---- CAR tests ----

static void test_car_basic(void) {
    car_state_t car;
    car_init(&car, 0.5f);

    // Set up a small cache scenario
    // 1 layer, 8 experts, 4 VRAM slots
    vram_cache_t vc;
    vram_cache_init(&vc, 1, 8, 4, 1024);
    // Cache experts 0, 1, 2, 3
    for (int i = 0; i < 4; i++)
        vram_cache_alloc_slot(&vc, 0, i);

    ram_cache_t rc;
    ram_cache_init(&rc, 1, 8, 2, 1024);
    // RAM cache expert 4
    ram_cache_alloc_slot(&rc, 0, 4);

    // Router selected experts 0, 5, 6 with scores
    int expert_ids[3] = {0, 5, 6};
    float expert_scores[3] = {0.5f, 0.3f, 0.2f};

    // Full softmax scores for all 8 experts
    float all_scores[8] = {0.5f, 0.15f, 0.12f, 0.08f, 0.05f, 0.3f, 0.2f, 0.01f};

    // Expert 0 is cached. Experts 5 and 6 are uncached.
    int uncached[2] = {1, 2}; // indices into expert_ids (positions of 5 and 6)
    int n_uncached = 2;

    int vram_sub_idx[3], vram_sub_slot[3], n_vsubs;
    int ram_sub_idx[3], ram_sub_slot[3], n_rsubs;

    int subs = car_evaluate(&car, 0,
        expert_ids, expert_scores, 3,
        all_scores, 8,
        &vc, NULL, &rc,
        uncached, &n_uncached,
        vram_sub_idx, vram_sub_slot, &n_vsubs,
        NULL, NULL, NULL,
        ram_sub_idx, ram_sub_slot, &n_rsubs);

    // Expert 5 (score 0.3): best cached not selected = expert 1 (score 0.15).
    //   ratio = 0.15/0.3 = 0.5 >= threshold 0.5 → substitute
    // Expert 6 (score 0.2): best remaining = expert 2 (score 0.12).
    //   ratio = 0.12/0.2 = 0.6 >= 0.5 → substitute
    CHECK(subs >= 1, "car: at least 1 substitution");
    CHECK(n_uncached == 0, "car: no uncached remaining");
    CHECK(car.substitutions == (uint64_t)subs, "car: stats match");

    // Scores should be renormalized
    float sum = 0;
    for (int i = 0; i < 3; i++) sum += expert_scores[i];
    CHECK(fabsf(sum - 1.0f) < 0.01f, "car: scores renormalized");

    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  car_basic: OK\n");
}

static void test_car_threshold_rejects(void) {
    car_state_t car;
    car_init(&car, 0.9f); // high threshold

    vram_cache_t vc;
    vram_cache_init(&vc, 1, 8, 4, 1024);
    vram_cache_alloc_slot(&vc, 0, 0);

    // Expert 5 selected (score 0.8), best cached = expert 0 (score 0.1)
    // ratio = 0.1/0.8 = 0.125 < 0.9 → reject
    int expert_ids[1] = {5};
    float expert_scores[1] = {0.8f};
    float all_scores[8] = {0.1f, 0, 0, 0, 0, 0.8f, 0, 0};
    int uncached[1] = {0};
    int n_uncached = 1;
    int vs_idx[1], vs_slot[1], n_vs;
    int rs_idx[1], rs_slot[1], n_rs;

    int subs = car_evaluate(&car, 0,
        expert_ids, expert_scores, 1,
        all_scores, 8, &vc, NULL, NULL,
        uncached, &n_uncached,
        vs_idx, vs_slot, &n_vs,
        NULL, NULL, NULL,
        rs_idx, rs_slot, &n_rs);

    CHECK(subs == 0, "car: high threshold rejects");
    CHECK(n_uncached == 1, "car: expert remains uncached");
    CHECK(car.nvme_remaining == 1, "car: nvme_remaining tracked");

    vram_cache_free(&vc);
    fprintf(stderr, "  car_threshold_rejects: OK\n");
}

// ---- Prefetch classification tests ----

static void test_prefetch_classify_all_vram(void) {
    // All predicted experts already in VRAM → no NVMe, no H2D
    vram_cache_t vc;
    vram_cache_init(&vc, 2, 16, 8, 1024);  // 4 slots/layer
    ram_cache_t rc;
    ram_cache_init(&rc, 2, 16, 4, 1024);

    // Fill VRAM cache for layer 1 with experts 0,1,2,3
    for (int i = 0; i < 4; i++)
        vram_cache_alloc_slot(&vc, 1, i);

    prefetch_state_t ps;
    prefetch_init(&ps, 4, 8, NULL, &rc, 16);  // no nvme_io (NULL)

    int predicted[] = {0, 1, 2, 3};
    float scores[] = {0.5f, 0.3f, 0.15f, 0.05f};

    int nvme = prefetch_classify_and_submit(&ps, 1, predicted, scores, 4, &vc, NULL, &rc);

    CHECK(nvme == 0, "pf: all VRAM, no NVMe");
    CHECK(ps.n_entries == 4, "pf: 4 entries");
    CHECK(ps.n_vram == 4, "pf: 4 VRAM hits");
    CHECK(ps.n_ram == 0, "pf: 0 RAM hits");
    CHECK(ps.n_nvme == 0, "pf: 0 NVMe");

    // All entries should have h2d_complete=true (already in VRAM)
    for (int i = 0; i < 4; i++) {
        const prefetch_entry_t *e = prefetch_find(&ps, 1, i);
        CHECK(e != NULL, "pf: entry found");
        CHECK(e->src == PREFETCH_SRC_VRAM, "pf: src is VRAM");
        CHECK(e->h2d_complete, "pf: h2d complete");
    }

    // No pending H2D
    const prefetch_entry_t *pending[8];
    int n_pending = prefetch_get_h2d_pending(&ps, pending, 8);
    CHECK(n_pending == 0, "pf: no H2D pending");

    prefetch_free(&ps);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  prefetch_classify_all_vram: OK\n");
}

static void test_prefetch_classify_ram_hits(void) {
    // Some predicted experts in RAM → need H2D, no NVMe
    vram_cache_t vc;
    vram_cache_init(&vc, 2, 16, 8, 1024);
    ram_cache_t rc;
    ram_cache_init(&rc, 2, 16, 4, 1024);

    // VRAM: layer 1 has experts 0, 1
    vram_cache_alloc_slot(&vc, 1, 0);
    vram_cache_alloc_slot(&vc, 1, 1);

    // RAM: layer 1 has experts 2, 3
    ram_cache_alloc_slot(&rc, 1, 2);
    ram_cache_alloc_slot(&rc, 1, 3);

    prefetch_state_t ps;
    prefetch_init(&ps, 4, 8, NULL, &rc, 16);

    int predicted[] = {0, 1, 2, 3};
    float scores[] = {0.4f, 0.3f, 0.2f, 0.1f};

    int nvme = prefetch_classify_and_submit(&ps, 1, predicted, scores, 4, &vc, NULL, &rc);

    CHECK(nvme == 0, "pf: no NVMe for RAM hits");
    CHECK(ps.n_vram == 2, "pf: 2 VRAM");
    CHECK(ps.n_ram == 2, "pf: 2 RAM");
    CHECK(ps.n_nvme == 0, "pf: 0 NVMe");

    // RAM entries need H2D and have VRAM slots reserved
    const prefetch_entry_t *e2 = prefetch_find(&ps, 1, 2);
    CHECK(e2 != NULL, "pf: expert 2 found");
    CHECK(e2->src == PREFETCH_SRC_RAM, "pf: expert 2 from RAM");
    CHECK(e2->vram_slot >= 0, "pf: expert 2 has VRAM slot");
    CHECK(e2->ram_slot >= 0, "pf: expert 2 has RAM slot");
    CHECK(e2->nvme_complete, "pf: RAM entry nvme_complete=true");
    CHECK(!e2->h2d_complete, "pf: RAM entry h2d not yet done");

    // 2 H2D pending (experts 2 and 3)
    const prefetch_entry_t *pending[8];
    int n_pending = prefetch_get_h2d_pending(&ps, pending, 8);
    CHECK(n_pending == 2, "pf: 2 H2D pending");

    // Mark one done
    prefetch_mark_h2d_done(&ps, 1, 2);
    n_pending = prefetch_get_h2d_pending(&ps, pending, 8);
    CHECK(n_pending == 1, "pf: 1 H2D pending after mark");

    prefetch_free(&ps);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  prefetch_classify_ram_hits: OK\n");
}

static void test_prefetch_classify_nvme_budget(void) {
    // Predicted experts not in any cache → need NVMe, budget enforced
    vram_cache_t vc;
    vram_cache_init(&vc, 2, 32, 8, 1024);
    ram_cache_t rc;
    ram_cache_init(&rc, 2, 32, 4, 1024);

    prefetch_state_t ps;
    // Budget: 3 NVMe reads max, no actual nvme_io (classification only)
    prefetch_init(&ps, 3, 8, NULL, &rc, 16);

    // Predict 6 experts, none cached
    int predicted[] = {10, 11, 12, 13, 14, 15};
    float scores[] = {0.3f, 0.25f, 0.2f, 0.1f, 0.1f, 0.05f};

    int nvme = prefetch_classify_and_submit(&ps, 1, predicted, scores, 6, &vc, NULL, &rc);

    // Only 3 should be submitted (budget), rest skipped
    CHECK(nvme == 3, "pf: 3 NVMe submitted (budget)");
    CHECK(ps.n_nvme == 3, "pf: n_nvme == 3");
    CHECK(ps.n_entries == 3, "pf: only 3 entries (rest skipped)");
    CHECK(ps.stat_budget_skips == 3, "pf: 3 budget skips");

    // First 3 (highest score) should be the ones submitted
    CHECK(prefetch_find(&ps, 1, 10) != NULL, "pf: expert 10 prefetched");
    CHECK(prefetch_find(&ps, 1, 11) != NULL, "pf: expert 11 prefetched");
    CHECK(prefetch_find(&ps, 1, 12) != NULL, "pf: expert 12 prefetched");
    CHECK(prefetch_find(&ps, 1, 13) == NULL, "pf: expert 13 skipped (budget)");

    // NVMe entries have reserved VRAM and RAM slots
    const prefetch_entry_t *e = prefetch_find(&ps, 1, 10);
    CHECK(e->src == PREFETCH_SRC_NVME, "pf: expert 10 from NVMe");
    CHECK(e->vram_slot >= 0, "pf: NVMe entry has VRAM slot");
    CHECK(e->ram_slot >= 0, "pf: NVMe entry has RAM slot");
    CHECK(!e->nvme_complete, "pf: NVMe not yet complete");

    prefetch_free(&ps);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  prefetch_classify_nvme_budget: OK\n");
}

static void test_prefetch_classify_mixed(void) {
    // Realistic scenario: mix of VRAM, RAM, and NVMe
    vram_cache_t vc;
    vram_cache_init(&vc, 1, 32, 8, 1024);  // 8 slots for 1 layer
    ram_cache_t rc;
    ram_cache_init(&rc, 1, 32, 6, 1024);

    // VRAM: experts 0, 1, 2
    for (int i = 0; i < 3; i++) vram_cache_alloc_slot(&vc, 0, i);
    // RAM: experts 3, 4
    ram_cache_alloc_slot(&rc, 0, 3);
    ram_cache_alloc_slot(&rc, 0, 4);

    prefetch_state_t ps;
    prefetch_init(&ps, 4, 10, NULL, &rc, 16);

    // Predict 10 experts: 3 VRAM, 2 RAM, 5 NVMe (budget=4)
    int predicted[] = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
    float scores[] = {0.2f, 0.15f, 0.12f, 0.1f, 0.08f, 0.08f, 0.07f, 0.07f, 0.07f, 0.06f};

    int nvme = prefetch_classify_and_submit(&ps, 0, predicted, scores, 10, &vc, NULL, &rc);

    CHECK(ps.n_vram == 3, "pf: 3 VRAM");
    CHECK(ps.n_ram == 2, "pf: 2 RAM");
    CHECK(nvme == 4, "pf: 4 NVMe (budget)");
    CHECK(ps.stat_budget_skips == 1, "pf: 1 budget skip (5th NVMe)");
    CHECK(ps.n_entries == 9, "pf: 9 entries (3+2+4, 1 skipped)");

    prefetch_free(&ps);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  prefetch_classify_mixed: OK\n");
}

static void test_prefetch_worker_lifecycle(void) {
    // Test that worker thread starts and shuts down cleanly without nvme_io
    prefetch_state_t ps;
    prefetch_init(&ps, 4, 8, NULL, NULL, 16);

    CHECK(!ps.worker.started, "pf: worker not started without nvme_io");

    prefetch_free(&ps);
    fprintf(stderr, "  prefetch_worker_lifecycle: OK\n");
}

static void test_prefetch_reset(void) {
    vram_cache_t vc;
    vram_cache_init(&vc, 1, 16, 4, 1024);
    ram_cache_t rc;
    ram_cache_init(&rc, 1, 16, 4, 1024);
    vram_cache_alloc_slot(&vc, 0, 0);

    prefetch_state_t ps;
    prefetch_init(&ps, 4, 8, NULL, &rc, 16);

    int predicted[] = {0, 5};
    prefetch_classify_and_submit(&ps, 0, predicted, NULL, 2, &vc, NULL, &rc);
    CHECK(ps.n_entries == 2, "pf: 2 entries before reset");

    prefetch_reset(&ps);
    CHECK(ps.n_entries == 0, "pf: 0 entries after reset");
    CHECK(ps.nvme_submitted == 0, "pf: 0 nvme after reset");

    // Stats should persist across reset
    CHECK(ps.stat_vram_hits == 1, "pf: stats persist");

    prefetch_free(&ps);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    fprintf(stderr, "  prefetch_reset: OK\n");
}

// ---- NVMe integration test (requires real drives) ----

static void test_prefetch_nvme_integration(void) {
    // Try to open expert stores — skip if not available
    const char *stores[] = {
        "/mnt/nvme_shard_a/experts-qwen3.5-397b-a17b.qmoe",
        "/mnt/nvme_shard_b/experts-qwen3.5-397b-a17b.qmoe",
        "/home/paul/nvme_shard_c/experts-qwen3.5-397b-a17b.qmoe",
    };

    nvme_io_t *nvme = nvme_io_init(stores, 3);
    if (!nvme) {
        fprintf(stderr, "  prefetch_nvme_integration: SKIPPED (no drives)\n");
        return;
    }

    uint64_t stride = nvme->buffer_size;

    // Set up caches sized for 397B (60 layers, 512 experts)
    vram_cache_t vc;
    vram_cache_init(&vc, 60, 512, 120, stride);  // 2 slots/layer (tiny for test)

    ram_cache_t rc;
    ram_cache_init(&rc, 60, 512, 120, stride);    // 2 slots/layer
    // Allocate actual RAM cache buffer
    rc.buf = calloc(rc.max_slots, stride);
    CHECK(rc.buf != NULL, "nvme: RAM buffer allocated");

    // Pre-populate VRAM cache with experts 0,1 at layer 5
    vram_cache_alloc_slot(&vc, 5, 0);
    vram_cache_alloc_slot(&vc, 5, 1);

    // Pre-populate RAM with expert 2 at layer 5
    ram_cache_alloc_slot(&rc, 5, 2);

    // Init prefetch with real NVMe, budget=2, K=6
    prefetch_state_t ps;
    int rc2 = prefetch_init(&ps, 2, 6, nvme, &rc, 16);
    CHECK(rc2 == 0, "nvme: prefetch_init OK");
    CHECK(ps.worker.started, "nvme: worker thread started");

    // Predict 6 experts for layer 5:
    //   0,1 = VRAM hit, 2 = RAM hit, 3,4 = NVMe (budget=2), 5 = budget skip
    int predicted[] = {0, 1, 2, 3, 4, 5};
    float scores[] = {0.3f, 0.25f, 0.2f, 0.12f, 0.08f, 0.05f};

    int nvme_count = prefetch_classify_and_submit(&ps, 5, predicted, scores, 6,
                                                   &vc, NULL, &rc);
    CHECK(ps.n_vram == 2, "nvme: 2 VRAM hits");
    CHECK(ps.n_ram == 1, "nvme: 1 RAM hit");
    CHECK(nvme_count == 2, "nvme: 2 NVMe reads submitted");
    CHECK(ps.stat_budget_skips == 1, "nvme: 1 budget skip");

    // Wait for NVMe reads to complete
    prefetch_wait_nvme(&ps);

    // NVMe entries should now have data in RAM cache slots
    const prefetch_entry_t *e3 = prefetch_find(&ps, 5, 3);
    CHECK(e3 != NULL, "nvme: expert 3 found");
    CHECK(e3->nvme_complete, "nvme: expert 3 NVMe complete");
    CHECK(e3->ram_slot >= 0, "nvme: expert 3 has RAM slot");

    // Verify data was actually copied to RAM cache (non-zero)
    void *data = ram_cache_slot_ptr(&rc, e3->ram_slot);
    uint8_t *bytes = data;
    int nonzero = 0;
    for (int i = 0; i < 64; i++)
        if (bytes[i] != 0) nonzero++;
    CHECK(nonzero > 0, "nvme: RAM slot has data (non-zero bytes)");

    // H2D pending should include RAM hit + 2 NVMe entries
    const prefetch_entry_t *pending[8];
    int n_pending = prefetch_get_h2d_pending(&ps, pending, 8);
    CHECK(n_pending == 3, "nvme: 3 H2D pending (1 RAM + 2 NVMe)");

    prefetch_free(&ps);
    free(rc.buf);
    vram_cache_free(&vc);
    ram_cache_free(&rc);
    nvme_io_free(nvme);
    fprintf(stderr, "  prefetch_nvme_integration: OK\n");
}

// ---- Main ----

int main(void) {
    fprintf(stderr, "=== v2 module tests ===\n\n");

    fprintf(stderr, "Expert cache:\n");
    test_vram_cache_basic();
    test_vram_cache_lru_eviction();
    test_ram_cache_basic();

    fprintf(stderr, "\nFrequency profile:\n");
    test_freq_profile_roundtrip();
    test_freq_profile_seed();

    fprintf(stderr, "\nCache-aware routing:\n");
    test_car_basic();
    test_car_threshold_rejects();

    fprintf(stderr, "\nPrefetch pipeline:\n");
    test_prefetch_classify_all_vram();
    test_prefetch_classify_ram_hits();
    test_prefetch_classify_nvme_budget();
    test_prefetch_classify_mixed();
    test_prefetch_worker_lifecycle();
    test_prefetch_reset();

    fprintf(stderr, "\nNVMe integration:\n");
    test_prefetch_nvme_integration();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
