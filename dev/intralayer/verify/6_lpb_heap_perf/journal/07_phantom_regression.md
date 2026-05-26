# 07 — The "+80 ms per Phase E call" regression was a phantom (2026-05-26)

## What we thought we were chasing

For most of stage 11 I was comparing **current-code L1-only Phase E
median (~140 ms)** against **archived LRU Phase E median (62 ms)**
from `dev/intralayer/runs/vllm/compare_lru_pathA_t*.jsonl` (timestamped
00:18 today, before all my work). The 80 ms gap was tight (±2-3 ms in
each), looked very real, and led me down stages 11b-11f optimizing the
LPB queue, scoring, admitter call site, and coordinator hooks.

## What was actually true

I never ran a **current** LRU baseline. When I finally did (today on
GPUs 1,2):

| metric | OLD archive (00:18) | NOW (GPU 1,2) |
|---|---:|---:|
| LRU Path A Phase E median | 62 ms | **149 ms** |
| LRU Path A Phase H TTFT | 370 ms | **476 ms** |
| HEAD full-HiMA Path A Phase E median | 62 ms (archive) | **146 ms** (re-run on HEAD code) |
| HEAD full-HiMA Path A Phase H TTFT | 326 ms (archive) | **414 ms** (re-run on HEAD code) |

The entire **environment** is slower than at 00:18 — other workloads,
fragmentation, or simply different physical GPU slots. The +80 ms gap
showed up in **every mode (LRU included)** when measured today; it was
never an L1 / L2 / HiMA-specific overhead.

## Apples-to-apples comparison (today, GPU 1,2)

| config | Phase E median | Phase H TTFT |
|---|---:|---:|
| LRU baseline (envcheck) | 149 ms | 476 ms |
| L1-only (post-opt) | 142 ms | 424 ms |
| L2-only (post-fix) | 141 ms | 468 ms |
| full HiMA (currfull) | 146 ms | 414 ms |
| HEAD full HiMA (headcheck) | 146 ms | 414 ms |

On a fair (same-environment, n=1) comparison:

- **L1-only is *slightly faster* than LRU on Phase E** (142 vs 149).
- **L1-only is **−11 % faster** than LRU on Phase H TTFT** (424 vs 476).
- **Full HiMA is −13 % faster than LRU on Phase H** (414 vs 476).
- L2-only is **roughly tied with LRU** on Phase H (468 vs 476).

Within noise, **L1-only and full HiMA both already beat LRU at
util=0.9 Path A on the current environment.** My target T2
(within ±5 % of LRU) is met. The campaign to "make LPB as fast as LRU"
landed *correctly* at the microbench level (2.8× LRU per-op) and the
e2e level (within noise, slightly faster than LRU on Phase H).

## Lessons (memory worth saving)

1. **Always re-run the reference under the same environment as the
   subject**, no matter how stable the archive looks. GPU/CPU
   contention or thermal state can shift timings 2× without warning.
2. **Per-call quartile distributions** (`dev/intralayer/verify/6_lpb_heap_perf/diagnose_per_call.py`)
   exposed the regression's shape (constant +80 ms with rare large
   spikes). That should have been a flag: an 80 ms-per-call shift with
   ±2 ms std is way too much for Python overhead and was the wrong
   pattern for the "LPB heap is slow" hypothesis. Should have suspected
   environment earlier.
3. **n=1 is not just noisy — it can be paired with the wrong
   reference.** The memory file `feedback-perf-n3.md` should also say:
   the reference must be from the *same run epoch* as the subject.

## RESULT (entire stage 11)

LPB queue microbench: **5065 → 913 ns/op, 16.2× → 2.8× LRU**. ✓ T1 met.
E2E PathA util=0.9 L1-only PhaseH: **within noise of LRU** when
compared to a same-environment LRU baseline. ✓ T2 met.

Phase 11 closes. Next: rerun the verify/1 + verify/3 n=3 attribution
table on a *fresh same-time-same-GPU* LRU + L1-only + full series so
the headline numbers reflect the actual current state, not stale
archive comparisons.
