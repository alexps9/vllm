# 2. Songyang SWE-bench W1 regression repro

> **Scaffold only — no driver code, no result data yet.** This README
> documents the *design* of the scenario. The driver
> (`driver.py`) is to be written as part of Phase 7 (task #50) and
> [`verify/5`](../5_window_sensitivity/README.md) must finish first to
> pin the `VLLM_HIMA_HPB_WINDOW_S` value. Until both land, this folder
> contains only this README + an empty `runs/`.

## What we're verifying

Songyang's SWE-bench W1 runs (Qwen3.5-35B-A3B hybrid, 16 concurrent clients,
util=0.55) show a sharp prefix-cache hit-rate collapse at turns ≥ 32:

| turns | baseline hit% | HiMA hit% | p95 TTFT delta |
|---:|---:|---:|---:|
| 4  | 47.9 | 46.3 | tied |
| 8  | 65.0 | 65.2 | tied |
| 16 | 75.5 | 75.5 | tied |
| **32** | 84.2 | **44.2** | **+193 %** |
| **64** | 91.1 | **56.5** | **+217 %** |

Songyang's stated hypothesis: *"LPB evicts shared prefixes before LRU
because their hit count decays faster than per-conversation unique
blocks in the windowed counter."* The hypothesis does not survive code
reading — the path-counter window is 3600 s in his run, so no decay
fires within a 10-min experiment. Real root cause is unknown without
isolation, since his run had **both L1 and L2 of HiMA on plus
`VLLM_PARTIAL_CACHE_ENABLED=1`** (3 mechanisms stacked).

We reproduce the regression on a controlled testbed and attribute it
to one of: L1 (LPB scoring), L2 (admitter / budgeter / planner),
partial-cache, or interaction effects.

## Expected outcome

Two hypotheses to discriminate between:

1. **Regression is L1-only**: even with L2 off + partial-cache off, turns=64 L1-only run shows hit% collapse. → confirms LPB scoring bug (lazy refresh + depth-as-integer treating `c_pool` as constant).
2. **Regression needs L2**: L1-only is tied with LRU baseline; full-HiMA (L1+L2) reproduces collapse. → admitter/budgeter is the culprit.

Negative result (regression doesn't reproduce on Qwen3-8B) would mean
the issue is specific to hybrid mamba + long context, not L1 scoring.

## Workload design

Mock SWE-bench W1 traffic on **Qwen3-8B** (single-group full-attention;
hybrid would add the `partial-cache bypass on len(kv_cache_groups) != 1`
confound which we want to keep out of this scenario):

- 16 concurrent clients
- per-client conversation of `turns` round-trips
- prompt grows ~1k tokens per turn (mocks SWE-bench tool-history accumulation)
- `--gpu-memory-utilization` tuned so 16 × turns × ~2k context oversubscribes the KV pool (small pool → force evictions)
- **`VLLM_HIMA_HPB_WINDOW_S` pinned to the value picked by [`verify/5`](../5_window_sensitivity/README.md)**;
  encoded in run filenames as `_win<value>` so this never drifts silently
- turn settings: turns=8 is a single sanity cell (4 turns × 1 config = "should be tied" baseline); turns=32 and turns=64 run the full config matrix

6 configs × 2 turn settings (32, 64) + 1 turns=8 sanity = **13 cells**, ~3 min each.

| config | flags | what it tests |
|---|---|---|
| A: LRU | (nothing) | baseline |
| B: L1-only | `VLLM_HIMA_L1_ENABLE=1` | LPB scoring alone |
| C: L2-only | `VLLM_HIMA_L2_ENABLE=1` | admitter+budgeter alone (LRU queue intact) |
| D: full | `VLLM_HIMA_L1_ENABLE=1 VLLM_HIMA_L2_ENABLE=1` | L1+L2 combined |
| E: L1 + pcache | B + `VLLM_PARTIAL_CACHE_ENABLED=1` | L1 × partial-cache interaction |
| F: L2 + pcache | C + `VLLM_PARTIAL_CACHE_ENABLED=1` | L2 × partial-cache interaction |
| G: full + pcache | D + `VLLM_PARTIAL_CACHE_ENABLED=1` | Songyang's actual config (modulo hybrid → single-group) |

Columns A/G are mandatory at every turn level; B/C/D/E/F are mandatory
only at turns=32 and turns=64 (turns=8 is the "should be tied" sanity
that anchors the matrix). C and F separately cover Gap A and Gap D
identified by the verify-plan audit.

## Verdict criteria

Reference = config A (LRU) at the same turn count. Threshold: ≥ 20 pp
drop in prefix-cache hit% at turns=64.

- **Repro confirmed** if any of {D, G} crosses the threshold.
- **L1 attribution** if B (L1-only) crosses too; L1 scoring is the culprit. Pass-through to [`verify/4`](../4_lpb_scoring_variants/README.md) for bug isolation.
- **L2 attribution** if C (L2-only) crosses and B doesn't; admitter / budgeter is the culprit.
- **Interaction (L1×L2)** if B and C both tie with A but D crosses; combined-stack only.
- **Partial-cache amplifies** if {E, F, G} cross by ≥10 pp more than {B, C, D} respectively.
- **Repro failed (negative)** if all configs at turns=64 within ±5 pp of A. Next step is **hybrid-model repro** (Qwen3.5-35B-A3B, 2× H200 TP=2) — capability-gated; see [`verify/6_hybrid_repro`](../6_hybrid_repro/README.md) **(not yet built; will be scoped only if turns=64 returns negative on Qwen3-8B)**.

## How to repro

Prereq: Phase 1 (sub-flags) landed.

```bash
cd /data/yuzhou/projects/vllm-songyang
mkdir -p dev/intralayer/verify/2_songyang_w1_regression_repro/runs

# (driver.py to be written as part of Phase 7)
for turns in 8 32 64; do
  for config in lru l1_only full_hima full_hima_pcache; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
      dev/intralayer/verify/2_songyang_w1_regression_repro/driver.py \
      --turns $turns --config $config --clients 16 \
      | tee dev/intralayer/verify/2_songyang_w1_regression_repro/runs/turns${turns}_${config}.out
  done
done
```

## Status

pending — blocked by Phase 1 (sub-flags) and Phase 7 (driver).

## Result

_(filled in after Phase 7 completes)_
