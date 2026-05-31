# 4. LPB scoring variant attribution

> **Scaffold only — no code, no result data yet.** This README
> documents the *design* of the scenario. The `VLLM_HIMA_LPB_SCORING`
> env knob must be added to `vllm/v1/core/hima/lpb_free_queue.py`
> first (Phase 10a, task #53), and the W1 workload
> must produce a baseline turns=64 cell before this scenario can fire.
> Until both land, this folder contains only this README + an empty
> `runs/`.

## What we're verifying

Two suspected bugs in `vllm/v1/core/hima/lpb_free_queue.py:_score_for`:

1. **Lazy refresh**: score is computed at `append()` time only (when a
   block re-enters the free queue) and never refreshed after.
   `path_counter.record_hit` increments `n_b`, but the heap score
   stays at whatever `n_b` was at the last `append`. Lookup
   `vllm/v1/core/hima/integration.py:105-111` and
   `vllm/v1/core/hima/lpb_free_queue.py:80-110`.
2. **Depth-as-integer**: `c_pool` cost is computed via
   `c_kv_ms(depth)` where `depth` is the integer path index (1..K).
   The cost curve expects an L = *token count*. As a result
   `c_pool(depth=1) ≈ c_pool(depth=50) ≈ 0.44 ms` — effectively
   constant. LPB scoring then degenerates to `n_b × 0.44` (just an
   LFU on hit count, the depth dimension is gone).

Either bug could explain a hit-rate collapse under pressure. This
scenario discriminates between them by running a single high-pressure
cell (the W1 workload
turns=64) under each scoring variant.

## Expected outcome

- **Lazy fixes it / depth doesn't** → eager refresh recovers L1
  competitiveness. The `_evict_cost`/`peek_n_scores` heap stale-ness
  was the real issue. Implementation bug, ~1 week to fix.
- **Depth fixes it / lazy doesn't** → cost-curve semantics were
  wrong. Even with eager refresh, LFU-on-hits isn't enough; you
  need depth-aware protection. Slightly heavier fix but still
  implementation-level.
- **Both fix it (eager+depth_tokens variant)** → both bugs
  contribute. The fix combines them.
- **Neither fixes it** → the LPB design as specified in the paper
  has a deeper issue. Paper-rewrite territory.

## Code change required

Patch `vllm/v1/core/hima/lpb_free_queue.py` (and possibly
`vllm/v1/core/hima/integration.py`) to support an env-selectable
scoring mode:

```
VLLM_HIMA_LPB_SCORING ∈ {
    "lazy",                # current behaviour (default)
    "eager",               # refresh score on every record_hit
    "depth_tokens",        # c_kv_ms(depth * block_size) instead of c_kv_ms(depth)
    "eager_depth_tokens",  # both fixes combined
}
```

Implementation outline:
- `_score_for`: branch on the depth-handling based on
  `os.environ.get("VLLM_HIMA_LPB_SCORING")`.
- `HiMARuntime.record_hit`: when variant is `eager*`, additionally
  call `refresh_lpb_score(block)` for each path block currently in
  the free queue (need access to `_lpb_queues` registry — already
  present in HiMARuntime).
- Log the active variant once at engine start so the .out file makes
  it auditable.

This is Phase 10a (separate task #53), prerequisite of running this
scenario.

## Workloads

After Phase 10a lands, run the W1 workload
turns=64 cell at 2 configs (lru baseline, l1_only) × 4 scoring
variants = 8 cells.

| variant | lru | l1_only |
|---|---|---|
| lazy (current) | (= W1 cell) | (= W1 cell) |
| eager | ✓ | ✓ |
| depth_tokens | ✓ | ✓ |
| eager_depth_tokens | ✓ | ✓ |

lru baseline is included even though the scoring code path doesn't
run under LRU — it's the noise-floor sanity check that `VLLM_HIMA_LPB_SCORING`
has no effect when LPB isn't active.

## How to repro

Prereq: Phase 10a (env knob landed) + Phase 7 (the W1 driver
exists).

```bash
cd /data/yuzhou/projects/vllm-songyang
OUTDIR=dev/intralayer/verify/4_lpb_scoring_variants/runs

for variant in lazy eager depth_tokens eager_depth_tokens; do
  for mode in lru l1_only; do
    CUDA_VISIBLE_DEVICES=0 KMP_AFFINITY=disabled \
      VLLM_HIMA_LPB_SCORING=$variant \
      .venv/bin/python -u \
        dev/intralayer/driver.py \
          --turns 64 --config $mode --clients 16 \
      > "$OUTDIR/turns64_${variant}_${mode}.out" 2>&1
  done
done
```

## Status

**done (2026-05-30).** Phase 10a (env knob) landed; Phase 10b ran on the
PathA anchor-survival scenario. See [`results.md`](RESULTS.md) and
[`run.sh`](run.sh).

## Result

**The LPB scoring variant makes no difference** — lazy = eager =
depth_tokens = eager_depth_tokens, bit-identical PhaseH hit% (88.87 %) and
anchor survival (126720 cached). The two suspected bugs (lazy refresh,
depth-as-integer) are real in principle but benign: the 500×-warmed
anchor's hit count dominates regardless, and the depth-cost rescale is
monotone so it doesn't reorder eviction. L1 still beats the LRU floor
(88.87 vs 85.91 %, +2.96 pp). **Decision: keep `lazy`**; the knob remains a
diagnostic. Full table + reasoning in [`results.md`](RESULTS.md).
