# dev/intralayer — LPB vs LRU, cross-engine

Cross-engine notes on HiMA's Layer-1 LPB (hits-per-byte) eviction
policy. Both vLLM and sglang have implementations on respective
forks; this directory holds the shared workload design and the
per-engine implementation + results.

## Layout

| file / dir | what's in it |
|---|---|
| [`scenarios.md`](scenarios.md)   | Engine-agnostic phase pipeline (A → B → G → E → F → H → C). Pitfalls, expected outcomes, design lessons. |
| [`vllm.md`](vllm.md)             | vLLM HiMA L1 implementation pointers + measured results. **Headline (L1-only, fresh n=3): Phase H win −8.8…−10.7 % Path A, −12.2 % Path B.** (The prior −12 %/−17.7 % "full-stack" figure was a stale-baseline phantom — see verify/1.) L1 uses recency-aware LPB eviction (verify/10). |
| [`sglang.md`](sglang.md)         | sglang LPB implementation review + 4-round optimization journey + 8-variant measured results. **Headline: no regression after fixes (post-memoization within noise) on 7 prior variants; ✓ −15.7 % mean / −25.7 % median TTFT achieved on the skewed-popularity stress (8th variant), comparable to vLLM's −10.7 %/−12.2 %. Prelude's single-trial −19.77 % GSP headline does not reproduce at n=3.** |
| [`verify/INDEX.md`](verify/INDEX.md) | **Numbered verification scenarios.** Each `N_<slug>/` is one self-contained verification (README + RESULTS + run.sh + runs/). `INDEX.md` is the current map with verdicts. |
| `runs/vllm/`                     | vLLM `.jsonl` / `.out` / `summary.json` per trial. |
| `runs/sglang/`                   | sglang `.jsonl` / `.out` per trial (current = v5 mem; includes Path A baseline + Path A two-anchor variant). |
| `runs/sglang_gsp/`               | sglang GSP bench results (n=3), the prelude-headline workload that did not reproduce. |
| `runs/sglang_skewed/`            | sglang **skewed-popularity stress (n=3): the LPB-win workload**. Zipf(α=1.5) 12 groups, `--max-mamba-cache-size 8`. |
| `runs/sglang_prefix/`            | sglang v1 (pre-fix) archive. |
| `runs/sglang_v2_heap/`           | sglang v2 (+heap +deque +real-bytes +cleanup) archive. |
| `runs/sglang_v3/`                | sglang v3 (+two-phase eviction) archive. |
| `runs/sglang_v4_baseline_for_compare/` | sglang v4 (+evict_full LPB extension) archive. |
| `runs/sglang_s30/`               | sglang scale=30 variant archive. |
| `runs/sglang_skipG_mambaonly/`   | sglang skipG with v1-style mamba-only LPB. |
| `runs/sglang_skipG_v2_both_paths/` | sglang skipG with LPB extended to evict_full. |
| `figures/`                       | vLLM `fig_lru_vs_lpb_*.png` (anchor + scenarios per sweep). |

## Verification scenarios

L1 was re-attributed from the old combined `hima_enabled` switch (which once
turned on both L1 and the now-removed L2); HiMA is now enabled per layer via
`VLLM_HIMA_L1_ENABLE`. The full scenario list with verdicts lives in
[`verify/INDEX.md`](verify/INDEX.md). Summary: **L1 wins** (Path A −8.8…−10.7 %,
Path B −12.2 %, fresh n=3), scoring variants are a no-op, recency-aware fix
verified (verify/10), **L2 removed** (≈ LRU; archived in
[`dev/archive/L2/`](../archive/L2/)), **pcache removed**.

## Bottom line

| engine | best result on dev/intralayer Path A + variants | why |
|---|---|---|
| **vLLM**   | **L1 −8.8…−10.7 % Path A / −12.2 % Path B** batch TTFT, fresh n=3 (vLLM's per-block `FreeKVCacheBlockQueue` evicts the anchor under Phase F's pressure; L1's recency-aware LPB protects it; the post-pressure swarm reveals the difference) | per-block LRU doesn't track recency at the prefix-tree level, so LPB's explicit hit-count signal is needed to protect the anchor |
| **sglang** | **LPB tied with LRU on 7 workloads BUT −16.2 % mean / −26.9 % median TTFT on the 8th** (`runs/sglang_skewed/`, n=3: Zipf(α=1.5) 12-group skewed-popularity workload, one-shot requests, `--max-mamba-cache-size 8` → forces real snapshot rotation; cache hit % jumps 30.5 % → 51.4 %). The tied workloads (baseline scale=10, baseline v5-memoized, scale=30, skipG-v1, skipG-v2, two-anchor, GSP) all violated one of the two conditions LPB needs: free-leaf snapshots and skewed hit counts. | (a) sglang's per-node radix-tree LRU already encodes prefix-level recency, (b) the radix-tree lock-ref keeps internal nodes (anchors with live child sessions) structurally untouchable regardless of policy, (c) on uniform-popularity workloads LPB tie-breaks to recency. The skewed workload removes (b) (one-shot requests = free leaves) and (c) (Zipf-biased traffic) and adds tight mamba pressure — LPB then protects the top-3 hottest snapshots simultaneously where LRU only protects the most-recent. |

The two engines behave differently on the *same* benchmark because
of structurally different LRU designs, not because of bugs in
either LPB implementation.

For sglang, the goal lands at:
- **worst case ✓** no regression (verified across 30+ trials, 7
  workload variants; baseline residual is +2.26 % within noise)
- **best case ✓** real perf gain achieved on the 8th workload
  (skewed-popularity stress, n=3): **−16.2 % mean TTFT, −26.9 %
  median TTFT, +68.7 % cache hit rate**, comparable to vLLM's
  −12 %/−17.7 % Path A/B win.

The skewed-popularity driver lives in the sglang repo at
`dev/intralayer/skewed_bench.py` + `dev/intralayer/skewed_run.sh`; per-trial
results are in `runs/sglang_skewed/`.

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
  driver `dev/intralayer/compare_lru_lpb.py` in the sglang repo,
  env gate `SGLANG_LPB_LRU=1`. Optimization commits on top of the
  initial squash: `9bc52737e` (A+B+G+I), `076507663` (E),
  `36a16bfdc` (evict_full extension + skip-G flag).
