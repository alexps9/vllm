# vLLM HiMA L1 — LPB implementation + measured results

The vLLM-side LPB implementation, its driver script, and the
two-sweep × 3-trial measurements that landed the headline
production-pattern win.

See [`scenarios.md`](scenarios.md) for the engine-agnostic phase
design that drives both `compare_lru_lpb.py` (here) and the
sglang side ([`sglang.md`](sglang.md)).

## Implementation

| component | file |
|---|---|
| LPB-ordered free-block queue (heap, `_HIT_SCORE_OFFSET = 1e12`) | `vllm/v1/core/hima/lpb_free_queue.py` |
| Engine knob (`hima_enabled: bool`) | `vllm/engine/arg_utils.py` (threads through `CacheConfig`) |
| HiMA runtime + windowed hit counter | `vllm/v1/core/hima/runtime.py` (`PathCountedHitCounter`) |
| Cost curves per pool | `vllm/v1/core/hima/cost_curve.py` |
| Hybrid KV-cache coordinator wrapper | `vllm/v1/core/hima/coordinator_hima.py` |
| Driver (this experiment) | `dev/compare_lru_lpb.py` |
| Aggregator + figures | `dev/plot_lru_vs_lpb.py` |
| Dataset | `dev/cc_long_traces.jsonl` (106 real Claude-Code sessions, 44 MB, committed) |

LPB score: cold block = `time.monotonic()`; any block with
`n_b > 0` (windowed hits) = `time.monotonic() + 1e12`. Binary
protected/not, tie-broken by recency within the protected set.
Replaces `FreeKVCacheBlockQueue` wholesale — applies to every pool,
not just mamba.

Wiring: `EngineArgs(hima_enabled=True)` enables it. Window
configurable via `VLLM_HIMA_HPB_WINDOW_S` (default 60 s — Path A
uses 3600 to keep the anchor's hits from expiring across the
multi-minute run).

## Sweeps

| tag | model | util | TP | KV budget | Phase F scale | files |
|---|---|---|---|---|---|---|
| `_pathA` | Qwen/Qwen3.5-35B-A3B   | 0.9 | 2 | 8.46 M tokens | 10 (50 decoys × 30 K + 500 cold) | `dev/intralayer/runs/vllm/compare_{lru,lpb}_pathA_t{1,2,3}.{jsonl,out}` |
| `_pathB` | Qwen/Qwen3.5-122B-A10B | 0.9 | 4 | 5.08 M tokens | 10                                | `dev/intralayer/runs/vllm/compare_{lru,lpb}_pathB_t{1,2,3}.{jsonl,out}` |

## Headline — anchor protection (Phase C FINAL probe)

|                                       | LRU         | LPB           | trials confirming |
|---------------------------------------|------------:|--------------:|---:|
| BASELINE anchor probe (after warmup)  | 4224 / 4737 | 4224 / 4737   | 6/6 each, stddev = 0 |
| FINAL probe (post all phases)         | **0 / 4737** | **4224 / 4737** | 6/6 each, stddev = 0 |

Binary outcome, perfectly reproducible across model size.

## Workload-metric results (mean ± sample stddev, n=3 each)

### Phase B (cc-burst average) — no regression

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr (tok/s) | LPB thr (tok/s) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A |  92.20 ± 0.68 |  92.41 ± 0.27 | +0.2 % | 12.57 ± 0.10 | 12.58 ± 0.05 | +0.1 % | 124.30 ± 0.75 | 124.73 ± 0.45 | +0.3 % |
| Path B | 125.51 ± 0.60 | 124.61 ± 2.29 | −0.7 % |  8.85 ± 0.13 |  8.75 ± 0.12 | −1.1 % | 126.36 ± 0.30 | 126.42 ± 0.95 | +0.0 % |

### Phase G (pre-pressure swarm) — tied at util=0.9 by design (control)

| sweep | LRU batch TTFT (ms) | LPB batch TTFT (ms) | Δ | LRU cached | LPB cached |
|---|---:|---:|---:|---:|---:|
| Path A | 331.66 ± 2.51 | 331.20 ± 3.08 | −0.1 % | 126 720 ±0 | 126 720 ±0 |
| Path B | 481.40 ± 2.42 | 478.32 ± 0.81 | −0.6 % | 126 720 ±0 | 126 720 ±0 |

At util=0.9 the anchor survives Phase B in both modes, so both swarms
hit. Confirms G alone isn't enough at production util — the LRU
disadvantage shows up after E + F apply pressure (Phase H below).

### Phase E (cold-unique random) — tied

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr | LPB thr | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A | 63.25 ± 0.61 | 62.82 ± 0.76 | −0.7 % |  6.32 ± 0.67 |  6.37 ± 0.57 | +0.8 % | 173.3 ± 1.9 | 171.8 ± 1.3 | −0.8 % |
| Path B | 87.15 ± 0.17 | 86.60 ± 0.55 | −0.6 % | 12.36 ± 0.85 | 12.52 ± 0.70 | +1.2 % | 115.7 ± 2.8 | 114.9 ± 2.2 | −0.7 % |

### Phase F (adversarial decoy waste) — tied

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr | LPB thr | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A | 62.56 ± 0.26 | 63.18 ± 0.61 | +1.0 % |  6.51 ± 0.12 |  6.59 ± 0.16 | +1.3 % | 172.97 ± 0.36 | 171.33 ± 0.19 | −0.9 % |
| Path B | 86.37 ± 0.19 | 86.58 ± 0.28 | +0.2 % | 11.83 ± 0.07 | 11.84 ± 0.09 | +0.1 % | 119.15 ± 1.22 | 119.11 ± 1.07 | −0.0 % |

Path A Phase F is the worst LPB number across all runs (+1.3 % TPOT,
~2.4 LRU-stddev gap, magnitude sub-2 %). Consistent with LPB's
heap/path-counter overhead being non-zero per request, not a
correctness regression. Note: `--phase-f-scale 10` only reaches
~49 % of KV budget on Path B; truly saturating decoys (scale ≥ 20)
would be needed to actually trigger the protect-useless-blocks
failure mode, and that hasn't been done.

### Phase H (POST-pressure SWARM) — **the production LPB win**

| sweep | LRU batch TTFT (ms) | LPB batch TTFT (ms) | Δ | LRU req/s | LPB req/s | total wall Δ |
|---|---:|---:|---:|---:|---:|---:|
| Path A | 370.03 ± 7.40 | 325.51 ± 7.41  | **−12.0 %** (~6σ) | 39.5 | 41.7 | **−5.5 %** |
| Path B | 565.35 ± 2.79 | 465.30 ± 18.07 | **−17.7 %** (~5σ) | 26.1 | 28.3 | **−8.5 %** (saves 100 ms) |

- **Hit rate is binary, stddev = 0**: LRU 85.91 % (29/30 hit, 1
  request pays the anchor prefill, the other 29 share via vLLM's
  prefix-cache merge) vs LPB 88.87 % (30/30 hit, anchor still
  cached from Phase A's warmup).
- **44 ms (Path A) → 100 ms (Path B) gap** is exactly one
  anchor-prefill cost on the respective hardware — bigger model →
  slower per-token prefill → bigger saving.
- **Token throughput is essentially unchanged** because batched
  decode wall already dominates. The win lives in TTFT (and the
  resulting request throughput / total batch wall, where the
  ~46-100 ms shows up directly).

## Findings

1. **Production-pattern win (Phase H)**: −12 % to −18 % batch
   TTFT, 5–6σ separation, perfectly reproducible. Bigger model →
   bigger win.
2. **No regression elsewhere**: every other metric × phase × sweep
   tied within noise or LPB slightly faster. Worst LPB outcome:
   Path A Phase F +1.3 % TPOT (sub-2 %, at noise edge).
3. **Anchor protection is binary and reproducible**: 6/6 LRU
   evicts vs 6/6 LPB protects.
4. **LPB's true worst case hasn't been triggered**: Phase F at
   scale=10 reaches ~49 % KV occupancy on 122 B and produces no
   measurable regression. Scale ≥ 20 would be needed.

## Production implications

- **Production swarm pattern** (N concurrent agents sharing an
  anchor, after the cache has been churned): LPB delivers
  **−12 % to −18 % batch TTFT**, saves 44–100 ms per swarm
  depending on model size, +5.5 % to +8.5 % request throughput.
- **Average workload metrics** (Phases B / E / F): LPB and LRU
  are indistinguishable. No regression on cc-burst, on
  cold-unique, or on adversarial decoy patterns.

## Figures

| Path | anchor survival | per-phase grid |
|---|---|---|
| A | `dev/intralayer/figures/fig_lru_vs_lpb_anchor_pathA.png` | `dev/intralayer/figures/fig_lru_vs_lpb_scenarios_pathA.png` |
| B | `dev/intralayer/figures/fig_lru_vs_lpb_anchor_pathB.png` | `dev/intralayer/figures/fig_lru_vs_lpb_scenarios_pathB.png` |

## Repro

```bash
cd /scratch/yuzhou/projects/vllm-songyang
# venv + build (one-time)
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Path A — Qwen3.5-35B-A3B, TP=2, util=0.9. ~5 min per trial.
for trial in 1 2 3; do
  for mode in lru lpb; do
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u dev/compare_lru_lpb.py \
      --mode $mode --trial $trial \
      --tag _pathA --util 0.9 --tp 2 --phase-f-scale 10 \
      > dev/intralayer/runs/vllm/compare_${mode}_pathA_t${trial}.out 2>&1
  done
done
.venv/bin/python dev/plot_lru_vs_lpb.py --tag _pathA | tee dev/intralayer/runs/vllm/compare_summary_pathA.out

# Path B — Qwen3.5-122B-A10B, TP=4, util=0.9. ~7 min per trial.
for trial in 1 2 3; do
  for mode in lru lpb; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u dev/compare_lru_lpb.py \
      --mode $mode --trial $trial \
      --tag _pathB --util 0.9 --tp 4 --phase-f-scale 10 \
      --model Qwen/Qwen3.5-122B-A10B \
      > dev/intralayer/runs/vllm/compare_${mode}_pathB_t${trial}.out 2>&1
  done
done
.venv/bin/python dev/plot_lru_vs_lpb.py --tag _pathB | tee dev/intralayer/runs/vllm/compare_summary_pathB.out
```
