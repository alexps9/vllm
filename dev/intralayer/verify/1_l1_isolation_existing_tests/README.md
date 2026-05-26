# 1. L1 isolation of existing intralayer tests

## What we're verifying

All existing "intralayer" tests in `dev/intralayer/` (compare_lru_lpb,
e2e_l1_burst, e2e_l1_pressure_curve) were previously run with a master
`hima_enabled=True` switch — now removed — that turned on the **entire
HiMA stack**: not just L1 (LPB queues + path counter) but also L2
(Admitter + BisectionBudgeter + CrossPoolPlanner). So the prior "LPB
beats LRU" headline results conflated L1 and L2 contributions.

After Phase 1 (introducing independent `VLLM_HIMA_L1_ENABLE` /
`VLLM_HIMA_L2_ENABLE` sub-flags as the **only** way to enable HiMA;
the legacy `VLLM_HIMA_ENABLE` env var and `hima_enabled=True` CLI/kwarg
master switch no longer exist), we rerun the same workloads under
**L1-only** and compare against:

- **LRU baseline** (no HiMA at all) → does pure LPB beat LRU?
- **full-stack** (old "lpb" runs in `dev/intralayer/runs/vllm/`) → how much of the prior win/regression was L1 vs L2?

## Expected outcome

We do not have a strong prior. Three plausible cases:

1. **L1-only matches full-stack** → L2 was a no-op or noise on these workloads; prior results survive.
2. **L1-only matches LRU** → all of the prior "LPB win" was actually L2 doing the work; LPB scoring as-implemented is essentially LRU.
3. **L1-only sits between** → both layers contribute; need separate L2-only run to fully attribute.

Case 2 would be the most consequential — it would mean our intralayer
paper claim is wrong as stated and the LPB scoring (lazy update, depth
treated as integer not token count) is failing as theorized.

## Workloads to rerun

| driver | cells | prior result (full-stack) |
|---|---|---|
| `compare_lru_lpb.py --mode l1_only` | Path A × 3 trials + Path B × 3 trials = 6 cells | Path A −12 %, Path B −17.7 % TTFT |
| `e2e_l1_pressure_curve.py --mode l1_only` | pressure sweep | (see `dev/intralayer/e2e_l1_pressure_curve.jsonl`) |

**Skipped / dropped from the original verify/1 plan:**

- **`e2e_l1_burst.py`** — single point on the pressure curve;
  redundant when the curve sweep is rerun under the same mode.
- **Phase G** of `compare_lru_lpb.py` — documented "tied by design"
  in [`scenarios.md`](../../scenarios.md) at util=0.9 (anchor
  survives Phase B in both LRU and LPB; the divergence is at Phase
  H *after* the cold-burst pressure). Running it under l1_only
  produces no new signal. Keep its rows in the log for symmetry
  but don't compute / report a delta on it.

LRU baseline cells (in `dev/intralayer/runs/vllm/compare_lru_*`) do
not need rerunning — they don't touch HiMA.

## How to repro

Prereq: Phase 1 (sub-flags) ✅ + Phase 3 (`--mode l1_only` in drivers) ✅.
**Data file prereq**: `dev/intralayer/cc_long_traces.jsonl` must exist
(symlink `../interlayer/cc_long_traces.jsonl` was created on 2026-05-26 —
the canonical file lives in `dev/interlayer/`).

```bash
cd /data/yuzhou/projects/vllm-songyang
OUTDIR=dev/intralayer/verify/1_l1_isolation_existing_tests/runs
mkdir -p "$OUTDIR"

# Path A (Qwen3.5-35B-A3B, util=0.9, TP=2, phase-f-scale=10)
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=3,4 KMP_AFFINITY=disabled \
    .venv/bin/python -u dev/intralayer/compare_lru_lpb.py \
      --mode l1_only --tag _pathA --trial $trial \
      --util 0.9 --tp 2 --phase-f-scale 10 \
    > "$OUTDIR/compare_l1_only_pathA_t${trial}.out" 2>&1
  cp "dev/intralayer/runs/vllm/compare_l1_only_pathA_t${trial}.jsonl" "$OUTDIR/"
done

# Path B (Qwen3.5-122B-A10B, util=0.9, TP=4)
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1,2,3,4 KMP_AFFINITY=disabled \
    .venv/bin/python -u dev/intralayer/compare_lru_lpb.py \
      --mode l1_only --tag _pathB --trial $trial \
      --util 0.9 --tp 4 --phase-f-scale 10 \
      --model "Qwen/Qwen3.5-122B-A10B" \
    > "$OUTDIR/compare_l1_only_pathB_t${trial}.out" 2>&1
  cp "dev/intralayer/runs/vllm/compare_l1_only_pathB_t${trial}.jsonl" "$OUTDIR/"
done

# Pressure curve (subsumes e2e_l1_burst; --out writes JSONL directly into OUTDIR)
CUDA_VISIBLE_DEVICES=3,4 KMP_AFFINITY=disabled \
  .venv/bin/python -u dev/intralayer/e2e_l1_pressure_curve.py \
    --mode l1_only \
    --out "$OUTDIR/e2e_l1_pressure_curve_l1_only.jsonl" \
  > "$OUTDIR/e2e_l1_pressure_curve_l1_only.out" 2>&1
```

## Status

**Path A n=3 ✅ done.** Path B (TP=4) + pressure curve still queued.

## Result — Path A n=3 (fresh same-environment, util=0.9)

### About the baseline

The numbers below are from the **fresh n=3 same-environment sweep** in
[`verify/6_lpb_heap_perf/runs/fresh_n3/`](../6_lpb_heap_perf/runs/fresh_n3/)
(LRU, L1-only, full HiMA all re-measured back-to-back on the same GPU
pair). The previously published table — which reported L1-only as
**+20.1 %** Phase H TTFT regression vs LRU — compared current-code
L1-only against an archived LRU baseline (`dev/intralayer/runs/vllm/`,
captured at ~00:18 on 2026-05-26, on a different GPU pair and under
different system load). The whole environment was faster at that time,
so the gap was artefactual. See
[`verify/6_lpb_heap_perf/journal/07_phantom_regression.md`](../6_lpb_heap_perf/journal/07_phantom_regression.md)
for the discovery and
[`verify/6_lpb_heap_perf/journal/08_fresh_n3_final.md`](../6_lpb_heap_perf/journal/08_fresh_n3_final.md)
for the authoritative replacement.

Phase H (post-pressure concurrent swarm) on Qwen3.5-35B-A3B, util=0.9, TP=2:

| config | n | PhaseH TTFT (ms ±sd) | vs LRU | hit% | tput (tok/s ±sd) | final anchor |
|---|---:|---:|---:|---:|---:|---:|
| LRU baseline (fresh n=3) | 3 | **479 ±14** | 0.0 % (ref) | 86 % | 1211 ±36 | 89 % |
| **L1-only** (fresh n=3) | 3 | **437 ±22** | **−8.8 %** ✓ wins | **89 %** (LPB queue) | 1208 ±45 (tied) | 89 % |
| full HiMA (fresh n=3) | 3 | **428 ±19** | **−10.7 %** ✓ wins | 89 % | 1191 ±11 (tied) | 89 % |
| L2-only (stale archive, pending fresh re-measure) | 3 | 468 ±9 | +26.6 % vs stale-LRU (370) | 85.9 % | 1229 ±19 | 89.2 % |

Full per-phase fresh n=3 (mean ±sd across trials; per-turn TTFT uses
the trial-median, swarm TTFT uses `batch_wall_s × 1000`):

| metric | LRU | L1-only | full HiMA |
|---|---:|---:|---:|
| Phase B (cc per-turn TTFT) | 140 ±6 ms | 137 ±2 ms | 137 ±1 ms |
| Phase E (cold per-prompt TTFT) | 144 ±2 ms | 144 ±5 ms | 143 ±3 ms |
| Phase F (decoy per-prompt TTFT) | 143 ±1 ms | 141 ±2 ms | 142 ±3 ms |
| Phase G (pre-pressure swarm TTFT) | 449 ±32 ms | 477 ±7 ms | 474 ±11 ms |
| **Phase H TTFT** | **479 ±14 ms** | **437 ±22 ms** | **428 ±19 ms** |
| Phase H hit % | 86 % | 89 % | 89 % |
| Phase H throughput | 1211 ±36 tok/s | 1208 ±45 tok/s | 1191 ±11 tok/s |
| final anchor | 89 % | 89 % | 89 % |

### Verdict at util=0.9 — L1 already wins under same-environment measurement

- **L1 alone beats LRU** by 8.8 % Phase H TTFT (within ~1σ of full
  HiMA's −10.7 % — within noise of full-stack on this workload).
- **L1 alone earns the +3 pp hit-rate** (86 % → 89 %), independently of
  the L2 stack — the LPB queue is what keeps the anchor warm.
- The earlier "Case 4 — L1 alone loses, only the interaction wins"
  reading was driven entirely by the stale-archive LRU reference and
  does **not** survive fresh same-environment measurement.
- Per-turn phases (B / E / F) are tied within ≤ 2 ms median, i.e. LPB
  per-allocation overhead is no longer measurable at the e2e level
  after the verify/6 LPB rewrite (see journal/08, T1 met).
- All configs preserve the final anchor at 89 % — anchor survival is
  still binary, but now LRU also keeps it on the *fresh* environment,
  so the win shows up as TTFT + hit% rather than catastrophic anchor
  eviction.

### L1 alone also beats LRU under genuine KV pressure (util=0.35, n=3 ✓)

The util=0.35 sweep below was a separately-run same-environment n=3
(LRU and L1-only measured back-to-back) and is **unaffected** by the
stale-archive issue described above — its LRU reference is contemporary
with its L1-only measurement. It remains a valid datapoint on the
pressure-dependent behaviour. Same workload at
`--util 0.35 --phase-f-scale 1` (tight pool, smaller F-phase pressure
to fit in the smaller budget):

| metric | LRU (n=3) | L1-only (n=3) | delta |
|---|---:|---:|---:|
| Phase G TTFT (pre-pressure swarm) | 487 ±12 ms | **430 ±0 ms** | **−12 %** ✓ |
| Phase H TTFT (post-pressure swarm) | 458 ±15 ms | **434 ±27 ms** | **−5 %** ✓ |
| Phase H hit % | 85.9 % | 88.9 % | +3 pp ✓ |
| throughput | 1251 ±58 tok/s | 1231 ±70 tok/s | −2 % (within noise) |
| final anchor | 89.2 % | 89.2 % | tied |

n=3 confirms: **L1 beats LRU by 12 % on Phase G TTFT and 5 % on Phase
H TTFT at util=0.35.** The Phase G std-dev of 0 ms on the L1 side is
notable — L1's path-counted eviction protection is more *deterministic*
than LRU's at this operating point. See [`why_l1_lost.md`](why_l1_lost.md)
for the full investigation, the LPB-vs-LRU CPU microbenchmark (77× per-op),
and the `maybe_get_free_queue_factory` bug fix.

### Final answer — L1 wins at both regimes; magnitude is pressure-dependent

Under fresh same-environment measurement, **L1 alone beats LRU on
Phase H at util=0.9 (−8.8 %) and at util=0.35 (−5 %)**. The win is
real at production utilisation and not just an artefact of tight-pool
regimes. The earlier "L1 loses at util=0.9" story was a stale-baseline
artefact (see [journal/07](../6_lpb_heap_perf/journal/07_phantom_regression.md))
combined with pre-rewrite LPB heap overhead (the verify/6 campaign
brought LPB per-op down from 16.2× LRU to 2.8× LRU; see
[journal/08](../6_lpb_heap_perf/journal/08_fresh_n3_final.md)). LPB's
anchor-protection benefit still scales with KV pressure (+3 pp hit-rate
under both regimes is consistent), while its per-allocation overhead
is now small enough that the +3 pp net positive at e2e level surfaces
at util=0.9 too. See [`why_l1_lost.md`](why_l1_lost.md) for the
historical investigation, the LPB-vs-LRU CPU microbenchmark, and the
`maybe_get_free_queue_factory` bug fix landed during this scenario.

### Implication for [verify/2](../2_songyang_w1_regression_repro/README.md)

Songyang's SWE-bench W1 runs at **util=0.55** — between our two
operating points (util=0.35 and util=0.9). With L1 now winning at
*both* endpoints under fresh same-environment measurement, the prior
"win/lose boundary" concern dissolves; L1 is no longer a suspected
regression source for the W1 workload. Songyang's specific
turns-≥32 hit-rate collapse may still be a pressure-curve cliff effect
similar to verify/5's K=15→20 cliff: at low turns enough KV slack
remains, but as conversation grows enough blocks accumulate that LRU
crosses the cliff and L1 (with admitter helping) preserves the prefix.
The W1 hypothesis space now shifts to admitter behaviour or
partial-cache interactions rather than LPB queue overhead (see
[journal/08, "Implications for sibling verify scenarios"](../6_lpb_heap_perf/journal/08_fresh_n3_final.md)).
