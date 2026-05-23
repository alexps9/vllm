# dev/aginfer — LPB vs LRU, cross-engine

Cross-engine notes on HiMA's Layer-1 LPB (hits-per-byte) eviction
policy. Both vLLM and sglang have implementations on respective
forks; this directory holds the shared workload design and the
per-engine implementation + results.

## Layout

| file / dir | what's in it |
|---|---|
| [`scenarios.md`](scenarios.md)   | Engine-agnostic phase pipeline (A → B → G → E → F → H → C). Pitfalls, expected outcomes, design lessons. |
| [`vllm.md`](vllm.md)             | vLLM HiMA L1 implementation pointers + measured results. **Headline: Phase H production-pattern win, −12 % Path A, −17.7 % Path B.** |
| [`sglang.md`](sglang.md)         | sglang LPB implementation review + 4-round optimization journey + 7-variant measured results. **Headline: no regression after fixes (was +60 ms, now +11 ms residual, post-memoization within noise), but no measurable Phase H win on any workload — eviction outcomes converge with LRU by tree-structure and by hit-0-dominated tie-breaks. Prelude's single-trial −19.77 % GSP headline does not reproduce at n=3.** |
| `runs/vllm/`                     | vLLM `.jsonl` / `.out` / `summary.json` per trial. |
| `runs/sglang/`                   | sglang `.jsonl` / `.out` per trial (current = v5 mem; includes Path A baseline + Path A two-anchor variant). |
| `runs/sglang_gsp/`               | sglang GSP bench results (n=3), the prelude-headline workload that did not reproduce. |
| `runs/sglang_prefix/`            | sglang v1 (pre-fix) archive. |
| `runs/sglang_v2_heap/`           | sglang v2 (+heap +deque +real-bytes +cleanup) archive. |
| `runs/sglang_v3/`                | sglang v3 (+two-phase eviction) archive. |
| `runs/sglang_v4_baseline_for_compare/` | sglang v4 (+evict_full LPB extension) archive. |
| `runs/sglang_s30/`               | sglang scale=30 variant archive. |
| `runs/sglang_skipG_mambaonly/`   | sglang skipG with v1-style mamba-only LPB. |
| `runs/sglang_skipG_v2_both_paths/` | sglang skipG with LPB extended to evict_full. |
| `figures/`                       | vLLM `fig_lru_vs_lpb_*.png` (anchor + scenarios per sweep). |

## Bottom line

| engine | best result on dev/aginfer Path A + variants | why |
|---|---|---|
| **vLLM**   | **LPB −12 % to −17.7 %** batch TTFT (vLLM's per-block `FreeKVCacheBlockQueue` evicts the anchor under Phase F's pressure; LPB protects it; the post-pressure swarm reveals the difference) | per-block LRU doesn't track recency at the prefix-tree level, so LPB's explicit hit-count signal is needed to protect the anchor |
| **sglang** | **LPB tied with LRU within noise across 7 workload variants** (baseline scale=10, baseline v5-memoized, scale=30, skipG-v1, skipG-v2, two-anchor, GSP). Every Phase H/swarm reports byte-identical `sum_cached`. Prelude's `7c6828c9a` GSP headline (−19.77 %, n=1) does **not** reproduce at n=3 (mean −0.86 %, within ±47 ms noise). | (a) sglang's per-node radix-tree LRU already encodes prefix-level recency, (b) the radix-tree lock-ref keeps internal nodes (anchors with live child sessions) structurally untouchable regardless of policy, and (c) LPB scoring degenerates to "hit-0 first, then recency" once the real 32 MB/slot mamba bytes dominate the denominator |

The two engines behave differently on the *same* benchmark because
of structurally different LRU designs, not because of bugs in
either LPB implementation.

For sglang, "best case real perf gain" is **not achieved on any
of our 7 workloads**. The structural reason is exhaustively
documented in `sglang.md`. A workload that should expose a win
(skewed-popularity multi-anchor + tight `max-mamba-cache-size`
forcing real snapshot rotation) is identified in §"What would
expose a measurable LPB win on sglang" but not yet implemented.

## Optimizations applied to sglang LPB (full journey in `sglang.md`)

|              | LPB regression | reduction |
|---|---:|---:|
| pre-fix v1   | +60 ms | — |
| v2 (heap, bounded deque, real bytes, cleanup) | +14 ms | 77 % |
| v3 (two-phase eviction) | +13 ms | 78 % |
| v4 (evict_full LPB extension) | **+11 ms** | **81 %** |

All on Qwen3.5-35B-A3B Path A util=0.9, n=3 trials each. Total
run-wall difference between modes is now within 0.3 s on a 200 s
pipeline — essentially zero overhead at the run-wall granularity.
The remaining +11 ms is the irreducible per-batch bookkeeping cost
on a workload where the eviction-policy choice doesn't change the
cached state.

## Engine references

- **vLLM HiMA**: `https://github.com/alexps9/vllm` branch `HiMA`,
  driver `dev/compare_lru_lpb.py`, scoring
  `vllm/v1/core/hima/lpb_free_queue.py`.
- **sglang LPB**: `https://github.com/rucnyz/sglang` branch `HiMA`
  (squashed from `prelude`, with `HPB`→`LPB` rename),
  scoring `python/sglang/srt/mem_cache/mamba_radix_cache.py`,
  driver `dev/aginfer/compare_lru_lpb.py` in the sglang repo,
  env gate `SGLANG_LPB_LRU=1`. Optimization commits on top of the
  initial squash: `9bc52737e` (A+B+G+I), `076507663` (E),
  `36a16bfdc` (evict_full extension + skip-G flag).
