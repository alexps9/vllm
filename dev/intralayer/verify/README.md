# dev/intralayer/verify — numbered verification scenarios

Each subfolder `N_<slug>/` represents one self-contained verification
scenario. To add a new scenario, append the next integer.

## Convention

```
verify/
├── README.md                          # this file
├── 1_<slug>/
│   ├── README.md                      # what's being verified, expected outcome, status
│   ├── runs/                          # raw .jsonl / .out from runs (gitignored if large)
│   └── (driver.py, plot.py, ...)      # scenario-specific scripts only if they differ
│                                      #   from the parent dev/intralayer/ drivers
├── 2_<slug>/
│   └── ...
```

Each `N_<slug>/README.md` should answer:

1. **What are we verifying?** (one paragraph)
2. **What's the expected outcome?** (concrete numbers / hypotheses)
3. **How to repro** (exact commands; assume reader is starting cold)
4. **Status**: pending / in-progress / done
5. **Result** (filled in when done): table + verdict

## Why numbered subfolders?

Scenarios accumulate over time. A flat `dev/intralayer/` directory with
ad-hoc `compare_*.jsonl` / `e2e_*.jsonl` from past investigations becomes
unreadable once we have more than a handful. Numbered subfolders give:

- chronological history at a glance
- self-contained scenario provenance (driver + config + raw output + summary in one place)
- zero-coordination expansion: next person just picks the next number

## Active scenarios

| # | slug | what's verified | status (fresh n=3, post-cleanup) | data |
|---|---|---|---|---|
| 1 | [l1_isolation_existing_tests](1_l1_isolation_existing_tests/README.md) | `compare_lru_lpb` PathA under L1-only at multiple `--util` | **L1 wins at both util=0.9 (−8.8 % PhaseH) and util=0.35 (−5 % PhaseH)** under fresh same-env measurement. Earlier "+20.1 % regression" was a stale-baseline artefact (see verify/6 journal/07). | runs/ |
| 2 | [songyang_w1_regression_repro](2_songyang_w1_regression_repro/README.md) | Repro W1 hit-rate collapse on Qwen3-8B; 7-config × turns={32,64} matrix, win=3600 pinned | **turns=32 done (n=1)** — W1 collapse does NOT reproduce (96.8 % hit across all HiMA configs vs W1's 44 %). Workload doesn't pressure KV at util=0.55 on single-group. **turns=64 blocked** by Qwen3-8B 40 K max_position_embeddings. **Surprise**: partial-cache regresses on single-group too (+45 % TTFT). See [results.md](2_songyang_w1_regression_repro/results.md). | runs/ |
| 3 | [l2_isolation_existing_tests](3_l2_isolation_existing_tests/README.md) | Mirror of (1) under L2-only, post-fix factory gate | **PathA n=3 done but stale** (00:18 epoch, before verify/6 fresh-baseline discovery). +26.6 % TTFT figure pending fresh re-measure on verify/6 GPUs. Headline-affecting numbers explicitly flagged in [README](3_l2_isolation_existing_tests/README.md#about-the-baseline). | runs/ |
| 4 | [lpb_scoring_variants](4_lpb_scoring_variants/README.md) | Discriminate two suspected LPB bugs (lazy refresh vs depth-as-integer) | **scaffold only** — original premise (L1 has scoring bugs causing W1 regression) is weakened now that L1 alone beats LRU at util=0.9 and W1 didn't repro on Qwen3-8B. Defer until hybrid-model repro confirms scoring needs investigation. | empty |
| 5 | [window_sensitivity](5_window_sensitivity/README.md) | `VLLM_HIMA_HPB_WINDOW_S` sweep on `e2e_l1_pressure_curve` | **5/5 cells done** — cliff K=15→20 at win ≤60s; K=20→25 at win ≥600s. Songyang's "decay" directionally right but quantitatively small | runs/ |
| 6 | [lpb_heap_perf](6_lpb_heap_perf/README.md) | **Engineering campaign**: drive LPB queue to within 3× LRU per-op, eliminate util=0.9 regression | **all 4 targets met (T1/T2/T3/T4)** — LPB rotate microbench 5065 → 913 ns/op (16.2× → 2.8× LRU). Fresh n=3 e2e: L1-only Phase H **−8.8 % vs LRU**. See [journal/08](6_lpb_heap_perf/journal/08_fresh_n3_final.md). | journal/01-08, patches/, runs/ |

## Key findings rolled up across scenarios (2026-05-26)

1. **`maybe_get_free_queue_factory` bug**: not gated on `hima_l1_enabled`,
   so L2-only mode was silently using LPB queue. **Fixed in
   `vllm/v1/core/hima/integration.py:364`** (the factory now also checks
   `_RUNTIME.config.hima_l1_enabled`). All "L2-only" data collected
   before this fix is tainted; verify/3 was re-run cleanly under
   the fix.
2. **L1 wins or loses depending on KV pressure regime**, not as a
   universal claim. At util=0.9 PathA the pool is slack enough that
   LRU also keeps the anchor → L1's heap overhead surfaces without
   benefit. At util=0.35 PathA the pool is tight, LRU drops the
   anchor → L1's protection saves cache hits.
3. **L1+L2 interaction at util=0.9 PathA is what produces the prior
   −12 % headline.** Neither layer individually beats LRU there. The
   mechanism of the interaction is hypothesized to involve admitter
   DEFER decisions reducing LPB heap churn — needs instrumentation
   (decisions counter) to confirm.
4. **LPBPriorityQueue is ~77× slower per op than the hand-tuned LRU
   `FreeKVCacheBlockQueue`** (CPU microbenchmark at N=8000 blocks).
   This is the constant cost L1 pays regardless of regime.
5. **Window decay matters only between 60 s and 600 s**; long windows
   (600 s / 3600 s / 86400 s) behave identically on
   e2e_l1_pressure_curve. Songyang's `VLLM_HIMA_HPB_WINDOW_S=3600` is
   safely on the "long" side.
