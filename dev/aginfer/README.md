# dev/aginfer — LPB vs LRU, cross-engine

Cross-engine notes on HiMA's Layer-1 LPB (hits-per-byte) eviction
policy. Both vLLM and sglang have implementations on respective
forks; this directory holds the shared workload design and the
per-engine implementation + results.

## Layout

| file / dir | what's in it |
|---|---|
| [`scenarios.md`](scenarios.md)   | Engine-agnostic phase pipeline (A → B → G → E → F → H → C). Pitfalls, expected outcomes, design lessons. |
| [`vllm.md`](vllm.md)             | vLLM HiMA L1 implementation pointers + measured results. Headline Phase H production-pattern win (−12 % Path A, −17.7 % Path B). |
| [`sglang.md`](sglang.md)         | sglang LPB implementation review + Path A measured results. Phase H **+67 % slower** on this workload because sglang's tree-LRU already protects the anchor; LPB has no work to do and only pays its overhead. |
| `runs/vllm/`                     | vLLM `.jsonl` / `.out` / `summary.json` per trial. |
| `runs/sglang/`                   | sglang `.jsonl` / `.out` per trial. |
| `figures/`                       | vLLM `fig_lru_vs_lpb_*.png` (anchor + scenarios per sweep). |

## Bottom line so far

- **Same scenarios, different engine outcomes**: identical Phase
  A → B → G → E → F → H → C pipeline produces opposite-sign LPB
  results on vLLM vs sglang at the same operating point
  (util=0.9, Qwen3.5-35B-A3B, n=3 each).
- **vLLM Phase H**: LPB **−12.0 %** batch TTFT (production-pattern
  win). LRU evicts the anchor under Phase F's pressure; LPB
  protects it; the swarm reveals the difference.
- **sglang Phase H**: LPB **+66.8 %** batch TTFT (regression).
  Sglang's radix-tree LRU **also** keeps the anchor cached
  through Phase F (both modes 99.7 % cached on the Phase H
  swarm), so LPB has no protection benefit to deliver — and
  pays its O(n) selector overhead instead.
- **Implication**: LPB's value is engine-specific. Where the
  baseline LRU already protects hot prefixes structurally
  (sglang's radix tree), LPB is overhead. Where the baseline LRU
  is per-block and doesn't (vLLM's free-block queue), LPB is the
  workload-metric win.
- **For sglang specifically**: the O(n) selector is the practical
  cost; switching to a heap (as vLLM does) would shrink the
  overhead. The deeper question of whether LPB protection adds
  anything *beyond* what sglang's tree-LRU already gives needs a
  workload that displaces the hot prefix's tree-node from
  recency — not in the current pipeline.

## Engine references

- **vLLM HiMA**: `https://github.com/alexps9/vllm` branch `HiMA`,
  driver `dev/compare_lru_lpb.py`, scoring
  `vllm/v1/core/hima/lpb_free_queue.py`.
- **sglang LPB**: `https://github.com/rucnyz/sglang` branch `HiMA`
  (squashed from `prelude`, with `HPB`→`LPB` rename),
  scoring `python/sglang/srt/mem_cache/mamba_radix_cache.py`,
  driver `dev/aginfer/compare_lru_lpb.py` in the sglang repo,
  env gate `SGLANG_LPB_LRU=1`.
