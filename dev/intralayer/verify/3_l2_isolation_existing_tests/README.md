# 3. L2 isolation of existing intralayer tests

## What we're verifying

Mirrors [`verify/1`](../1_l1_isolation_existing_tests/README.md) but
with **`--mode l2_only`** (admitter + bisection budgeter + cross-pool
planner active, LPB intra-pool queue *off* — i.e. default LRU queue
remains in use).

Without this, [`verify/1`](../1_l1_isolation_existing_tests/README.md)'s
Case 3 outcome ("L1-only sits between LRU and full-stack") cannot be
interpreted: it would be consistent with *either* additive L1+L2
contributions *or* a pure L1×L2 interaction. L2-only on the same
workloads tells us how much of the prior Phase H "−12% Path A / −17.7%
Path B" win is attributable to the inter-pool layer alone.

**Phase 6 (rewriting `dev/intralayer/vllm.md`) is blocked on this
result.** Updating the headline without L2-only data risks a wrong
attribution that this scenario would then immediately contradict.

## Expected outcome

Three plausible cases, parallel to verify/1 but for L2:

1. **L2-only matches LRU baseline** → L2 was a no-op on these
   workloads. Prior Phase H win is *all* L1.
2. **L2-only matches full-stack** → all of the win is L2 doing the
   work (admitter/budgeter are protecting the anchor via remap or
   eviction-cost shaping).
3. **L2-only sits between LRU and full-stack** → both layers
   contribute; with verify/1 we can decompose additive vs
   interaction.

## Workloads to rerun

| driver | cells | notes |
|---|---|---|
| `compare_lru_lpb.py --mode l2_only` | Path A + Path B × 3 trials = 6 cells | **Skip Phase G** in analysis: documented "tied by design" (`scenarios.md`); no new signal under l2_only either |
| `e2e_l1_pressure_curve.py --mode l2_only` | 1 sweep over pressure levels | Burst is a subset of pressure curve; rerun only this one |

LRU baseline cells (`compare_lru_path{A,B}_t{1,2,3}` in
`dev/intralayer/runs/vllm/`) remain valid — don't rerun.

## How to repro

Prereq: Phase 1 (sub-flags) ✅ + Phase 3 (`--mode l2_only` in drivers) ✅.
**Data file prereq**: `dev/intralayer/cc_long_traces.jsonl` must exist
(the canonical conversation-trace dataset).

```bash
cd /data/yuzhou/projects/vllm-songyang
OUTDIR=dev/intralayer/verify/3_l2_isolation_existing_tests/runs
mkdir -p "$OUTDIR"

# Path A (Qwen3.5-35B-A3B, util=0.9, TP=2, phase-f-scale=10)
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=5,6 KMP_AFFINITY=disabled \
    .venv/bin/python -u dev/intralayer/compare_lru_lpb.py \
      --mode l2_only --tag _pathA --trial $trial \
      --util 0.9 --tp 2 --phase-f-scale 10 \
    > "$OUTDIR/compare_l2_only_pathA_t${trial}.out" 2>&1
  cp "dev/intralayer/runs/vllm/compare_l2_only_pathA_t${trial}.jsonl" "$OUTDIR/"
done

# Path B (Qwen3.5-122B-A10B, util=0.9, TP=4)
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1,2,3,4 KMP_AFFINITY=disabled \
    .venv/bin/python -u dev/intralayer/compare_lru_lpb.py \
      --mode l2_only --tag _pathB --trial $trial \
      --util 0.9 --tp 4 --phase-f-scale 10 \
      --model "Qwen/Qwen3.5-122B-A10B" \
    > "$OUTDIR/compare_l2_only_pathB_t${trial}.out" 2>&1
  cp "dev/intralayer/runs/vllm/compare_l2_only_pathB_t${trial}.jsonl" "$OUTDIR/"
done

# Pressure curve
CUDA_VISIBLE_DEVICES=5,6 KMP_AFFINITY=disabled \
  .venv/bin/python -u dev/intralayer/e2e_l1_pressure_curve.py \
    --mode l2_only \
    --out "$OUTDIR/e2e_l1_pressure_curve_l2_only.jsonl" \
  > "$OUTDIR/e2e_l1_pressure_curve_l2_only.out" 2>&1
```

## Status

**Path A n=3 ✅ done** but the result is **pre-fix and tainted**.
A bug discovered in [`verify/1`'s why_l1_lost.md](../1_l1_isolation_existing_tests/why_l1_lost.md)
shows that the `LPBFreeBlockQueue` factory wasn't gated on
`hima_l1_enabled` — so L2-only mode was silently using the LPB queue,
which means our "L2-only" data is actually "L1+L2 stealth" with L2
admitter on top. **The fix landed on 2026-05-26 in `integration.py:364`.**
Path A n=3 needs to be re-run under the fix. A single-trial quick
check (t99 post-fix) confirms the LRU queue is now in effect under
L2-only mode (hit% dropped from 88.9 % to 85.9 %, matching LRU
baseline).

## Result — Path A n=3 (L2-only data is STALE; pending fresh re-measure)

### About the baseline

The L2-only n=3 row below was measured at ~00:18 on 2026-05-26
against an LRU baseline captured at the same time. After
[journal/07](../6_lpb_heap_perf/journal/07_phantom_regression.md)
revealed that environment-level timings shifted by ~80 ms per call
between that epoch and the verify/6 measurement epoch, the comparison
"L2-only TTFT vs the 00:18 LRU baseline" is **not directly comparable**
to the fresh n=3 LRU / L1-only / full HiMA results from
[journal/08](../6_lpb_heap_perf/journal/08_fresh_n3_final.md). The
fresh n=3 sweep did **not** include an L2-only column — that re-measure
is still pending. The numbers are kept below, flagged as stale, so the
attribution shape is still readable, but **do not subtract stale-LRU
from fresh-LRU columns** without re-running L2-only.

Joint attribution table (mixed-epoch — see caveat above):

| config | n | PhaseH TTFT (ms ±sd) | vs (stale-)LRU | hit% | tput (tok/s ±sd) | final anchor | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LRU baseline (stale archive, 00:18) | 3 | 370 ±7 | 0.0 % (ref) | 85.9 % | 1617 ±5 | 89.2 % | stale env |
| LRU baseline (fresh n=3, journal/08) | 3 | **479 ±14** | (ref) | 86 % | 1211 ±36 | 89 % | authoritative |
| L1-only ([verify/1](../1_l1_isolation_existing_tests/README.md), fresh n=3) | 3 | **437 ±22** | **−8.8 %** vs fresh-LRU | 89 % ← LPB queue | 1208 ±45 | 89 % | authoritative |
| **L2-only (STALE, pending re-measure)** | 3 | 468 ±9 | +26.6 % vs stale-LRU | 85.9 % ✓ matches LRU | 1229 ±19 | 89.2 % | needs fresh n=3 |
| full HiMA (fresh n=3, journal/08) | 3 | **428 ±19** | **−10.7 %** vs fresh-LRU | 89 % | 1191 ±11 | 89 % | authoritative |
| full-stack "lpb" (stale archive) | 3 | 326 ±7 | −12.0 % vs stale-LRU | 88.9 % | 1607 ±1 | 89.2 % | stale env |

(For history: the pre-fix "L2-only" run had hit% = 88.9 % — the LPB
queue fingerprint — because the factory-gate bug silently installed
the LPB queue under L2-only mode too. Pre-fix files have been
overwritten with post-fix data.)

### Verdict (provisional, pending fresh L2-only re-measure)

The stale-baseline reading was: "**both** L1 and L2 individually lose
to LRU; only the L1×L2 combination wins (−12 %)". The fresh n=3
(journal/08) overturned the L1 half of that reading — L1 alone now
beats LRU by 8.8 % at util=0.9, within ~1σ of full HiMA's 10.7 %. The
remaining question is whether L2-only also rehabilitates under fresh
same-environment measurement, or whether `decide_admission` overhead
genuinely costs ~10–25 % TTFT at util=0.9. Until L2-only is re-run on
the fresh GPU pair back-to-back with LRU / L1-only / full HiMA, the
table's "+26.6 %" cell should be treated as **possibly an environment
artefact rather than an admitter-overhead signal**.

Provisional reading (unchanged by fresh data, since L1-only and
full HiMA both kept hit% = 89 %):

1. **L2-only matches LRU on cached state** (hit% = 85.9 % in both
   epochs), confirming the `maybe_get_free_queue_factory` fix
   correctly disables the LPB queue under L2-only mode.
2. L2-only's std dev is tiny (±9 ms) — admitter overhead is
   *consistent* per call.
3. Whether L2-only's *level* is genuinely above LRU or merely
   reflecting the stale-epoch baseline is **the question fresh n=3
   needs to answer**.

### Interaction hypothesis (now lower priority)

The earlier hypothesis — "admitter's DEFER actions reduce the count
of allocations the LPB queue must process, which is how full-stack
beats both individual layers" — was constructed to explain the
stale-baseline Case 4. With L1 alone now beating LRU on fresh data,
the interaction story may not be needed at all. Reassess after the
L2-only fresh re-measure.

## Result (final)

_(awaiting (a) fresh n=3 L2-only on the verify/6 GPU pair back-to-back
with LRU / L1-only / full HiMA, and (b) Path B + pressure curve.
Until (a) lands, the joint attribution above is provisional.)_
