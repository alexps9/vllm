# 05 — e2e validation (n=3) of LPB optimizations (2026-05-26)

## Setup

PathA Qwen3.5-35B-A3B, util=0.9, TP=2, phase_f_scale=10. Driver
`compare_lru_lpb.py --mode l1_only --tag _pathA_postopt --trial {1,2,3}`.
Stored in `runs/compare_l1_only_pathA_postopt_t*.{jsonl,out}`.

## Result

n=3, per-phase (ms ±sd):

| phase | LRU n=3 | L1 pre-opt n=3 | L1 POST-OPT n=3 | post-opt vs LRU |
|---|---:|---:|---:|---:|
| Phase B (per-turn TTFT) | 94 ±0 | 156 ±6 | 153 ±4 | **+62.5%** |
| Phase E (cold per-prompt) | 63 ±0 | 144 ±3 | 159 ±33 | **+152.7%** |
| Phase F (decoy per-prompt) | 63 ±1 | 150 ±14 | 140 ±3 | **+122.9%** |
| Phase G (pre-pressure swarm) | 333 ±2 | 475 ±46 | 477 ±8 | **+43.3%** |
| Phase H (post-pressure swarm) | 370 ±7 | 444 ±42 | **424 ±10** | **+14.5%** |
| Phase H hit % | 85.9 | 88.9 | 88.9 | +3.5 % ✓ |
| Phase H throughput | 1617 ±5 | 1254 ±48 | 1215 ±13 | **−24.8%** |

## What changed

Phase H TTFT: **444 → 424 ms** (−4.5 % improvement vs pre-opt L1).
Other phases essentially unchanged (within noise). Throughput
basically unchanged.

## What the microbench told us vs what e2e shows

| | microbench rotate | e2e Phase H TTFT |
|---|---|---|
| baseline (pre-opt) | 5065 ns/op (16.2× LRU) | 444 ms (+20 % vs LRU) |
| post-opt | 913 ns/op (2.8× LRU) | 424 ms (+14.5 % vs LRU) |
| improvement | **−82 %** | **−4.5 %** |

The microbench wins on the LPB queue do not translate proportionally
to e2e. The 4.5 % Phase H improvement reflects only the fraction of
e2e time spent inside LPB queue ops; the dominant remaining cost is
**HiMA infrastructure beyond the queue itself** — likely some
combination of:

1. **Coordinator overrides** (`_hima_find_longest_cache_hit`,
   `_hima_cache_blocks`) — every cache-lookup and every
   request-cache_blocks path goes through a Python wrapper that
   adds frame overhead even when the body is a near-no-op
   (e.g. Phase E has no prefix hits so `record_hit` doesn't fire,
   but the override layer still runs).
2. **`decide_admission` on every scheduling iteration** — even in
   L1-only where the function short-circuits to `OWN_FREE`, it
   still allocates an `AdmissionDecision` dataclass per call. The
   call-site in `scheduler.py:444-472` also imports `contextlib`
   plus two HiMA modules per scheduling iter (Python caches them
   but the `getattr` lookups aren't free).
3. **HiMA runtime singleton existence** itself imposes some
   per-request overhead through the various `get_runtime()` checks
   in hot paths.

## Phase-wise analysis

- **Phase B/E/F (single-allocation per call)**: regression is **per-CALL**
  fixed cost (60-150 ms extra per call), not per-allocation. Most
  likely from coordinator override layer or per-scheduling-iter HiMA
  hooks.
- **Phase G/H (batched 30-request swarm)**: regression amortizes
  better (14-43 %) because the per-call fixed cost is spread over
  many allocations. Phase H is the smallest regression because the
  batch is largest.

This shape is **diagnostic**: it tells us the remaining overhead is
per-request, not per-allocation. We've optimized the wrong layer if
the goal is parity at util=0.9.

## RESULT (partial)

Target T2 (PathA util=0.9 n=3 L1-only PhaseH within ±5 % of LRU 370 ms)
**not met**. Current: +14.5 % vs LRU. Microbench target T1 (≤3× LRU
per op) **met** (2.8×).

To close the rest of the gap, the next stage must address per-request
HiMA overhead. Candidates:

- **11f**: hoist `decide_admission` short-circuit out of `HiMARuntime`
  back into the scheduler call-site so it's a single `is None` check
  + `continue`, no dataclass alloc, no function call frame
- **11g**: profile the actual e2e with py-spy or cProfile attached
  to the EngineCore subprocess during Phase E (cold, no prefix
  hits — should be the cleanest signal)
- **11h**: investigate whether the coordinator-override pattern itself
  can be folded into the base coordinator behind an `if runtime is
  not None` short-circuit at the override site, avoiding per-call
  Python frame setup

Stage 11e closes as a partial validation — the LPB queue is no longer
the bottleneck, but L1-only still doesn't match LRU at util=0.9.
Open follow-ups go into stages 11f/11g/11h.
