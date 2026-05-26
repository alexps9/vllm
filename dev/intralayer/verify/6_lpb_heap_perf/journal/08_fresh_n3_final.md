# 08 — Fresh same-environment n=3 (the authoritative comparison) (2026-05-26)

## Setup

After [journal/07](07_phantom_regression.md) revealed that all prior
"L1-only vs LRU" comparisons were against stale archive data (00:18
today, different GPUs), this stage runs **LRU, L1-only, and full HiMA
back-to-back on the same GPU pair (1,2), n=3 each**, so the LRU
baseline reflects the *current* environment.

9 cells, ~85 min total. Runs in
`dev/intralayer/verify/6_lpb_heap_perf/runs/fresh_n3/`.

## Result

PathA, util=0.9, TP=2, phase_f_scale=10, n=3 same environment:

| metric | LRU n=3 | L1-only n=3 | delta | full HiMA n=3 | delta |
|---|---:|---:|---:|---:|---:|
| Phase B (cc per-turn TTFT) | 140 ±6 ms | 137 ±2 ms | −1.9% | 137 ±1 ms | −2.0% |
| **Phase E (cold per-prompt TTFT)** | **144 ±2 ms** | **144 ±5 ms** | **−0.1%** ✓ | 143 ±3 ms | −0.7% |
| Phase F (decoy per-prompt TTFT) | 143 ±1 ms | 141 ±2 ms | −1.0% | 142 ±3 ms | −0.4% |
| Phase G (pre-pressure swarm) | 449 ±32 ms | 477 ±7 ms | +6.2% * | 474 ±11 ms | +5.5% * |
| **Phase H TTFT (post-pressure swarm)** | **479 ±14 ms** | **437 ±22 ms** | **−8.8%** ✓ | **428 ±19 ms** | **−10.7%** ✓ |
| Phase H hit % | 86 ±0 | 89 ±0 | **+3 pp** ✓ | 89 ±0 | +3 pp |
| Phase H throughput | 1211 ±36 tok/s | 1208 ±45 tok/s | −0.3% | 1191 ±11 tok/s | −1.6% |
| final anchor | 89 % | 89 % | tied | 89 % | tied |

\* Phase G is documented "tied by design" at util=0.9 (`scenarios.md`)
— pre-pressure swarm doesn't differentiate modes since the anchor
survives in all configs. The +6% spread on G is within LRU's own
±32 ms noise.

## Verdict — all four targets ✓

| target | accept | actual |
|---|---|---|
| **T1: microbench parity** | LPB rotate ≤ 3× LRU | **2.8× LRU** (913 ±15 ns/op vs 327 ±12) ✓ |
| **T2: e2e parity at slack** | PhaseH within ±5% of LRU | L1-only **−8.8%** vs LRU (better than parity) ✓ |
| **T3: preserve win at pressure** | util=0.35 still beat LRU | n=3 at util=0.35 showed L1-only −12% Phase G, −5% Phase H ✓ |
| **T4: semantic equivalence** | unit tests pass | `test_indexed_heap_equiv.py` + `test_tiered_wrapper.py` all pass ✓ |

Phase E specifically (the "regression" we chased) — **−0.1% vs LRU**,
i.e. exactly tied within sub-ms noise. The +80 ms per-call gap I was
optimizing for was an artifact of comparing against an archive from
00:18 today taken under different GPU conditions.

## What we actually shipped

1. **LPBPriorityQueue rewrite** ([journal/02](02_indexed_heap.md)) —
   `_Entry` dataclass → tuple-keyed lazy-delete heap. C-level
   comparison. −52% per-op.
2. **Tiered cold-FIFO + hot-heap** ([journal/03](03_tiered_eviction.md))
   — most popleft hits a vLLM `FreeKVCacheBlockQueue` linked-list
   fast path; only hot blocks (n_b > 0) touch the heap. Additional
   −62% per-op on top of (1).
3. **`_score_for` + `path_count.count` micro-opts**
   ([journal/04](04_score_for_opts.md)) — bound-method + curve cache;
   deadline gate on `path_count.count` to skip the deque walk when
   nothing is ripe.
4. **Per-request hook cleanup** ([journal/06](06_per_request_hooks.md))
   — admitter consultation skipped in L1-only mode; `contextlib.suppress`
   removed from coordinator's refresh loop. Negligible e2e impact (the
   bottleneck wasn't here), but removed a real waste.
5. **`maybe_get_free_queue_factory` factory-gate fix** (from earlier
   in the campaign, in `vllm/v1/core/hima/integration.py:364`) — the
   LPB queue factory was returning the LPB queue whenever a HiMA
   runtime existed; now gated on `hima_l1_enabled`. This was a real
   bug discovered along the way that affected L2-only mode in
   `verify/3`.

## Files / artifacts

- Code changes (uncommitted): `vllm/v1/core/hima/{intra_pool/lpb_queue.py, intra_pool/path_count.py, lpb_free_queue.py, integration.py, coordinator_hima.py}`, `vllm/v1/core/sched/scheduler.py`
- Patches: `dev/intralayer/verify/6_lpb_heap_perf/patches/{02_indexed_heap, 03_tiered_eviction, 04_score_for_opts}.patch`
- Tests: `test_indexed_heap_equiv.py`, `test_tiered_wrapper.py`
- Bench: `microbench_lpb.py`
- Run.sh wrappers: not yet added for verify/6, deferred (campaign closing)

## Implications for sibling verify scenarios

verify/1 (L1 isolation) + verify/3 (L2 isolation) — the headline
"L1-only +20% TTFT regression" claim was driven by stale archive
comparisons. Both readme files should be updated to use this fresh
same-environment data as the authoritative attribution.

verify/2 (Songyang W1 repro) — its driver still pends (Phase 7),
but the design should be aware that **L1-only is no longer the
suspected regression source for util=0.9 workloads** — the LPB queue
is now within noise of LRU. The W1 collapse hypothesis shifts to
admitter or partial-cache interaction.

## RESULT

Campaign complete. Stage 11 closes successfully. **LPB heap is as
fast as LRU at the e2e level on util=0.9 PathA** (Phase E tied,
Phase H 9% better, throughput tied), and microbench is at 2.8× LRU
per-op (vs original 16.2×, an 82% reduction in per-op overhead).
