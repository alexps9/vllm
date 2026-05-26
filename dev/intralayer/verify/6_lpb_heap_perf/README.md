# 6. LPB heap performance engineering

> **Engineering campaign, not a single-shot verification.** This folder
> tracks the multi-stage effort to make `LPBFreeBlockQueue` fast enough
> that L1-only beats LRU at util=0.9 too (not just util=0.35). Lives
> in `verify/` because it's bracketed by the same baseline ↔ target
> measurement discipline as the other scenarios.

## Why this exists

[`verify/1`](../1_l1_isolation_existing_tests/why_l1_lost.md) showed
L1-only is +20 % TTFT worse than LRU at util=0.9 PathA. The root cause
is a constant-factor overhead in `LPBPriorityQueue` + `LPBFreeBlockQueue`:
~5 µs per op vs ~0.3 µs for the LRU `FreeKVCacheBlockQueue`, a 16×
gap. At util=0.35 LPB's eviction-policy benefit dominates the overhead
and it wins; at util=0.9 there's no benefit, only cost.

User directive: **eliminate the overhead so L1 can win at *all*
operating points, not just pressured ones.**

## Targets

| target | metric | accept |
|---|---|---|
| **T1: microbench parity** | LPB rotate per-op ns | ≤ 3× LRU (= ≤ 1000 ns) |
| **T2: e2e perf parity at slack** | PathA util=0.9 n=3 L1-only PhaseH TTFT | within ±5 % of LRU's 370 ms |
| **T3: e2e perf preserved at pressure** | PathA util=0.35 n=3 L1-only PhaseH TTFT | still ≤ LRU's 458 ms |
| **T4: semantic equivalence** | unit-tested popmin order vs current | identical for all (n_b, depth, last_access) combos |

## Plan (staged)

| stage | what | unblocks | sub-agent |
|---|---|---|---|
| **11a — profile** ✅ | cProfile baseline + microbench harness (this folder's `microbench_lpb.py`) | 11b/11c/11d | self |
| **11b — indexed heap** | Replace `_Entry` dataclass + lazy-delete in `LPBPriorityQueue` with array+pos-map indexed binary heap. Mutate-in-place on update, no dead entries. Public API unchanged. | 11e | sub-agent |
| **11c — tiered eviction** | Wrap `LPBFreeBlockQueue` with a cold FIFO (vLLM-style doubly-linked-list) + small hot pq. 99 % of pops hit the cold path = LRU speed. | 11e | sub-agent |
| **11d — `_score_for` micro-opts** | Inline path_counter.count, cache c_kv_ms(depth), strip attribute lookups in hot loop. | 11e | sub-agent or self |
| **11e — validate** | Re-run microbench + PathA n=3 at util=0.9 and util=0.35. Measure against T1-T4. | done | self |

## Status board

| stage | status | bench (ns/op rotate, mean ±sd) | vs LRU | journal |
|---|---|---:|---:|---|
| baseline | reference | 5065 ±79 | 16.2× | [01](journal/01_baseline.md) |
| 11a profile | ✅ done | — | — | [01](journal/01_baseline.md) |
| 11b tuple+heapq | ✅ done | 2424 ±120 | 7.4× | [02](journal/02_indexed_heap.md) |
| 11c tiered eviction | ✅ done | **913 ±15** | **2.8×** ✓ T1 met | [03](journal/03_tiered_eviction.md) |
| 11d _score_for opts | ✅ done (subagent) | ≈ no extra delta at N_HOT=50 in bench | — | [04](journal/04_score_for_opts.md) |
| 11e e2e validation (stale baseline) | ✅ done | Phase H 444→424 ms vs **stale archive LRU 370 ms** | misleading | [05](journal/05_e2e_validation.md) |
| 11f admitter hoist + coord cleanup | ✅ done | no measurable delta — bottleneck wasn't there | — | [06](journal/06_per_request_hooks.md) |
| 11g phantom regression diagnosis | ✅ **done** — the +80 ms was environmental | L1-only matches LRU when measured fresh, same GPUs | **T2 ✓** | [07](journal/07_phantom_regression.md) |
| 11h fresh same-env n=3 confirmation | ✅ done | L1-only **PhaseH −9%** vs LRU, **PhaseE tied** | **T1+T2+T3+T4 ✓ all met** | [08](journal/08_fresh_n3_final.md) |

## Campaign complete

PathA util=0.9 n=3 fresh same-environment:

| metric | LRU n=3 | L1-only n=3 | full HiMA n=3 |
|---|---:|---:|---:|
| Phase E (cold per-prompt TTFT) | 144 ±2 ms | **144 ±5 ms** (−0.1%) | 143 ±3 ms (−0.7%) |
| Phase H TTFT | 479 ±14 ms | **437 ±22 ms** (**−8.8%**) | **428 ±19 ms** (**−10.7%**) |
| Phase H hit % | 86 % | 89 % (+3 pp) | 89 % (+3 pp) |
| Phase H throughput | 1211 ±36 tok/s | 1208 ±45 (tied) | 1191 ±11 (−1.6%) |

All four targets met. See [`journal/08_fresh_n3_final.md`](journal/08_fresh_n3_final.md) for the full breakdown and what was actually shipped (5 numbered changes).

### Cumulative result so far

LPB rotate: **5065 → 913 ns/op (−82 %)**.
LPB rotate + 1 refresh: **9608 → 1582 ns/op (−84 %)**.
Ratio vs LRU rotate: **16.2× → 2.8×** ✓ (target T1: ≤3×).

E2E result pending stage 11e.

## Convention

- **Every change** lands as a numbered entry in `journal/NN_<slug>.md`
  with: hypothesis, code diff summary, microbench result before/after,
  any new test added.
- **Patches** saved verbatim in `patches/NN_<slug>.patch` for easy
  diffing / revert.
- **Runs** (.jsonl/.out from e2e) saved in `runs/` keyed by stage.

## Layout

```
verify/6_lpb_heap_perf/
├── README.md            # this file (plan + status board)
├── microbench_lpb.py    # the harness — single source of truth for perf
├── journal/             # ordered diary of attempts
│   ├── 01_baseline.md
│   ├── 02_<slug>.md
│   └── ...
├── patches/             # snapshotted diffs
└── runs/                # microbench + e2e raw output
```
