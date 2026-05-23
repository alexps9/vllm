# dev/aginfer — LPB vs LRU, cross-engine

Cross-engine notes on HiMA's Layer-1 LPB (hits-per-byte) eviction
policy. Both vLLM and sglang have implementations on respective
forks; this directory holds the shared workload design and the
per-engine implementation + results.

## Layout

| file | what's in it |
|---|---|
| [`scenarios.md`](scenarios.md)   | Engine-agnostic phase pipeline (A → B → G → E → F → H → C) we want to run on both. Pitfalls, expected outcomes, the design lessons we learned. |
| [`vllm.md`](vllm.md)             | vLLM HiMA L1 implementation pointers + measured results (2 sweeps × 5 phases × n=3 trials = 30 engine loads). Has the headline Phase H production-pattern win. |
| [`sglang.md`](sglang.md)         | sglang LPB implementation review + comparison to vLLM design. Existing committed-run cross-check (control-shape, ties). The Phase H–style benchmark driver isn't written yet — flagged as open. |

## Bottom line so far

- **Designs match in spirit, differ in mechanics** (binary vs
  continuous scoring; heap vs O(n) selector; KV-wide vs
  mamba-only scope). See `sglang.md` for the table.
- **vLLM Phase H lands the production win**: −12 % to −18 %
  batch TTFT on concurrent swarm at util=0.9, 5–6σ, perfectly
  reproducible. Anchor protection binary across 6/6 trials per
  side. See `vllm.md`.
- **sglang side has the implementation but no Phase H driver
  yet**. Existing committed runs (single-trial Phase B-only
  shape) show tied — same shape vLLM saw before Phase H was
  added. Implementation review (`sglang.md`) finds no
  correctness blockers; two design quirks (`* 1024` size
  heuristic, mamba-only scoping) worth fixing before final
  numbers.

## Engine references

- **vLLM HiMA**: `https://github.com/alexps9/vllm` branch `HiMA`,
  driver `dev/compare_lru_lpb.py`, scoring
  `vllm/v1/core/hima/lpb_free_queue.py`.
- **sglang LPB**: `https://github.com/rucnyz/sglang` branch `hima`
  (squashed from `prelude`, with `HPB`→`LPB` rename),
  scoring `python/sglang/srt/mem_cache/mamba_radix_cache.py`,
  env gate `SGLANG_LPB_LRU=1`.
