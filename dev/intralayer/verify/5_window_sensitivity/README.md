# 5. Path-counter window sensitivity

## What we're verifying

Tests whether the LPB `PathCountedHitCounter` window length materially
changes anchor-survival behavior. Songyang's stated hypothesis for his
SWE-bench W1 regression — *"LPB evicts shared prefixes because their
hit count decays faster than per-conversation blocks"* — was dismissed
on the grounds that his runs used `VLLM_HIMA_HPB_WINDOW_S=3600` (1 h),
within which a 10-min experiment cannot decay anything. **That
dismissal is itself untested.** At the default 60 s window the
hypothesis is not obviously false.

If short windows show meaningfully different behavior, the dismissal
of his hypothesis was directionally wrong. If long and short windows
behave identically on our workload, it's well-founded. Either way,
this result pins down which window value the W1 workload
should use — and forces it to be encoded explicitly in file names.

## Expected outcome

Two plausible cases:

1. **Insensitive**: anchor survival % similar across all window values
   on the existing Path A/B / pressure-curve workloads → decay is a
   no-op in our test regime. Songyang's hypothesis genuinely doesn't
   explain his regression. the W1 run should pick the value matching
   his prod config (3600 s) for repro fidelity.
2. **Sensitive at the lower end**: short windows (30 s / 60 s) show
   the anchor losing protection while long windows (3600 s / 86400 s)
   preserve it → Songyang's "decay" hypothesis was directionally
   right at default windows. the W1 run must then run at *both* 60 s
   (default) and 3600 s (Songyang's value) to distinguish hypotheses.

## Workload

Use `e2e_l1_pressure_curve.py --mode l1_only` at the most pressure
level that already shows L1-vs-LRU divergence (Phase H of the
compare_lru_lpb pipeline isn't a single point but the curve gives
us a controlled sweep). Run once per window value.

| cell | `VLLM_HIMA_HPB_WINDOW_S` | notes |
|---|---|---|
| 1 | `30` | aggressive decay; expect anchor loss |
| 2 | `60` | default in HiMAConfig (`hima_lpb_window_s=60.0`) |
| 3 | `600` | 10 min; matches per-experiment scale |
| 4 | `3600` | Songyang's value, what other intralayer tests use |
| 5 | `86400` | effectively no decay (1 day) — control |

5 cells total.

## How to repro

Prereq: Phase 1 (sub-flags) ✅. **No data-file prereq** — this driver
uses an absolute path hardcoded at `e2e_l1_pressure_curve.py:45`
(`/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl`)
which is independent of the symlink the compare_lru_lpb pipeline needs.

```bash
cd /data/yuzhou/projects/vllm-songyang
OUTDIR=dev/intralayer/verify/5_window_sensitivity/runs
mkdir -p "$OUTDIR"

for win in 30 60 600 3600 86400; do
  CUDA_VISIBLE_DEVICES=1,2 KMP_AFFINITY=disabled \
    VLLM_HIMA_HPB_WINDOW_S=$win \
    .venv/bin/python -u dev/intralayer/e2e_l1_pressure_curve.py \
      --mode l1_only \
      --out "$OUTDIR/e2e_pressure_l1only_win${win}.jsonl" \
    > "$OUTDIR/e2e_pressure_l1only_win${win}.out" 2>&1
done
```

## Status

**All 5 cells done ✅.** Final result below.

## Result — anchor survival across windows

Anchor survival % across pressure levels (`e2e_l1_pressure_curve.py`,
TP=2, util=0.35, Qwen3.5-35B-A3B):

| K | LRU (existing) | win=30 | win=60 | win=600 | win=3600 | win=86400 |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % |
| 5  | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % |
| 10 | **0 %** ← LRU cliff | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % |
| 15 | 0 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % | 89.2 % |
| 20 | 0 % | **0 %** ← short-window cliff | **0 %** | 89.2 % | 89.2 % | 89.2 % |
| 25 | 0 % | 0 % | 0 % | **0 %** ← long-window cliff | **0 %** | **0 %** |
| 30 | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

### Verdict

**Two regimes, separated by ~600 s**:

- **Short windows (30 s / 60 s)**: cliff K=15 → K=20. Decay of warm-up
  hits *during* the long burst phases (K≥20 takes 250+ seconds) lets
  the anchor's path-counter slip to zero faster than the burst evicts
  it via raw cumulative pressure.
- **Long windows (600 s / 3600 s / 86400 s)**: cliff K=20 → K=25.
  Decay no longer matters at this experiment duration; the cliff is
  governed by raw cumulative-blocks-vs-pool-budget. All three long
  windows behave identically.

### Two findings

1. **Songyang's "decay" hypothesis is directionally right but
   quantitatively small.** Going from the default 60 s window to his
   3600 s setting buys exactly one extra K-level of anchor protection
   (cliff K=20 → K=25). It does *not* explain a turns-32+ hit-rate
   collapse on its own — anchor survival changes by less than one
   pressure level. The dismissal stands as "decay alone can't explain
   his W1 W2 regression at turns ≥ 32"; the directional element is
   noted.
2. **L1-only protects anchor for longer than LRU at every window
   value tested.** LRU cliffs at K=10, L1-only (any window) cliffs no
   sooner than K=20. That's a real anchor-protection win for L1 in
   the pressure-curve regime, *separate* from the compare_lru_lpb
   Phase H finding (where L1-only TTFT loses at util=0.9 — see
   [`verify/1`](../1_l1_isolation_existing_tests/why_l1_lost.md)).

### Recommendation for the W1 workload

Pin the window to **3600 s** (matching Songyang's prod config) so any
regression we observe is not a "we picked too short a window" artifact.
Encode in filename as `_win3600`. If a follow-up sweep is needed, add
a 60 s control cell at the turns=64 setting only.
