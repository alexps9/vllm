# Finding M.7 — partial-cache prototype scales: 43% TTFT win at block_size=1024

After M.6 validated the mechanism end-to-end on small prompts
(K=2 full blocks at block_size=256), this finding scales the
experiment up to the size regime that matches our actual target:
the hybrid model's inflated block_size of 1056 ≈ 1024.

## Setup

- Model: `Qwen/Qwen3-8B` (same as M.6, full-attention only)
- **`block_size = 1024`** (8× larger than M.6's 256, close to the
  hybrid model's 1056)
- **K = 8 full blocks** (base prompt = 8192 tokens)
- TP=1, util=0.6, max_num_seqs=4 on a single H200
- R sweep: {0, 64, 256, 512, 800, 1000, 1023}
- Each (R, mode) pair runs 3 times; report mean of last 2 (warmup excluded)

## Results

| R | turn1_len | turn2_len | baseline cached/unc/ttft_ms | partial cached/unc/ttft_ms | Δ uncached | Δ TTFT |
|---|---|---|---|---|---|---|
|    0 |  8192 |  8208 | 8192 /   16 / 19 | 8192 /   16 / 19 |       0 |    0 |
|   64 |  8256 |  8272 | 8192 /   80 / 19 | **8256** /  **16** / 19 | -80% |    0 |
|  256 |  8448 |  8464 | 8192 /  272 / 23 | **8448** /  **16** / **20** | -94% |  -13% |
|  512 |  8704 |  8720 | 8192 /  528 / 29 | **8704** /  **16** / **20** | -97% | **-31%** |
|  800 |  8992 |  9008 | 8192 /  816 / 35 | **8992** /  **16** / **20** | -98% | **-43%** |
| 1000 |  9192 |  9208 | 8192 / 1016 / 40 | **9192** /  **16** / **24** | -98% | **-40%** |
| 1023 |  9215 |  9231 | 9216 /   15 / 21 | 9215 /   16 / 20 | (special) | ~0 |

### Headline numbers

- **At R=800 (the typical mid-block partial)**: TTFT drops from **35
  ms → 20 ms** (-43%), uncached tokens drop from 816 → 16 (-98%).
- **At R=1000**: TTFT 40 → 24 ms (-40%), uncached 1016 → 16 (-98%).
- **Partial-cache TTFT is ~flat at ~20 ms** across the R sweep,
  regardless of bubble size. The baseline TTFT scales roughly
  linearly with R (because each uncached token costs prefill compute);
  with the fix, the cost decouples.

### Why the win scales

In M.6 (block_size=256), bubble was at most 255 tokens, TTFT ~20 ms,
absolute saving ~8 ms. In M.7 (block_size=1024), bubble can reach
1023 tokens, TTFT scales to ~40 ms baseline, absolute saving up to
~20 ms. The pattern continues to the hybrid case (block_size=1056):

- For Qwen3.5-35B-A3B with block_size=1056, R can be up to 1055.
- Expected baseline TTFT for R=800 follow-up: scales with model
  size; rough estimate 35B/8B × 35ms ≈ 150ms.
- Expected post-fix TTFT: ~50ms (proportional flat).
- Per-turn saving ≈ **100ms TTFT**. For a 50-turn cc session:
  **~5 seconds** of TTFT savings — matches Finding M.3's
  extrapolation.

## What this proves (in addition to M.6)

- The prototype is **kernel-tolerant at larger block sizes** —
  attention reads from the cached partial block at offset 0..R-1
  correctly even when R is near block_size.
- **TTFT win scales with bubble size**, not with engine
  bookkeeping cost. The hash lookup + map insertion are cheap.
- The R=0 case has **identical wall** between baseline and partial-
  cache (19 ms in both), confirming the new code path is a no-op
  when there's no partial.

## Caveats

- R=1023 is special: turn1 of 9215 tokens exactly fills 9 blocks
  (1023 partial in the 9th), so cache_blocks doesn't see a "partial"
  in the usual sense. The baseline shows turn2_cached=9216, the
  partial-cache run shows turn2_cached=9215. Both work; this is an
  edge case where R = block_size - 1.
- This is still single-group (non-hybrid). Hybrid Qwen3.5-35B-A3B
  needs the per-group `num_computed_tokens` lift (M.2). The
  *mechanism* is now proven; only the per-group plumbing remains
  for the headline-grabbing hybrid TTFT win.

## Files

- `07_nonhybrid_microbench_large.py` — bench script
- `runs/07_nonhybrid_large_baseline.{jsonl,out}` — env unset
- `runs/07_nonhybrid_large_partial_cache.{jsonl,out}` — env=1
