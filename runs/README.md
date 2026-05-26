# HiMA Evaluation on Qwen3.5-35B-A3B

Branch: `hima` · Model: `Qwen/Qwen3.5-35B-A3B` · Hardware: 8× RTX PRO 6000 Blackwell 96 GB (TP=2)

## Overview

Two workloads × two server configs (baseline vs HiMA) × two pressure settings (original / aggressive).

| | original | aggressive |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | **0.55** |
| `max_num_seqs` | 64 | **256** |
| W1 clients | 4 | **16** |
| W1 common_prefix | 512 tok | **2048 tok** |
| W2 turns/agent | 4 | **16** |
| W2 mock obs size | ~0.1 KB | **~1.5–3 KB** |

**HiMA env flags** (hima mode only):
```
VLLM_PARTIAL_CACHE_ENABLED=1  VLLM_PARTIAL_CACHE_MIN_R=256
VLLM_HIMA_ENABLE=1            VLLM_HIMA_HPB_WINDOW_S=3600
```

---

## Workloads

**W1** — vLLM native multi-turn benchmark (`benchmark_serving_multi_turn.py`).  
Sweeps `num_turns ∈ {4, 8, 16, 32, 64, 128}`. Measures server-side prefix-cache hit rate
and client TTFT via Prometheus deltas.

**W2** — Mini-SWE-Bench-Lite concurrent agent driver.  
Loads real problem statements from `princeton-nlp/SWE-Bench_Lite`, replays as K-turn agents
with realistic mock tool observations (no real shell). Sweeps `concurrency ∈ {1,2,4,8,16,32,64,128}`.

---

## Reproduce

```bash
export REPO=/path/to/vllm       # repo root (venv at $REPO/.venv)
export MODEL=/path/to/Qwen3.5-35B-A3B

cd "$REPO"
git checkout hima

# aggressive sweep (util=0.55, 16 clients, 16-turn agents)
MODEL=$MODEL REPO=$REPO VLLM_UTIL=0.55 VLLM_MAX_SEQS=256 VLLM_MAX_LEN=65536 VLLM_TP=2 \
  bash runs/scripts/run_sweep.sh baseline /tmp/out

MODEL=$MODEL REPO=$REPO VLLM_UTIL=0.55 VLLM_MAX_SEQS=256 VLLM_MAX_LEN=65536 VLLM_TP=2 \
  bash runs/scripts/run_sweep.sh hima /tmp/out

# original sweep (util=0.85, 4 clients)
MODEL=$MODEL REPO=$REPO \
  bash runs/scripts/run_sweep.sh baseline /tmp/out_orig 4 "4 8 16 32 64 128"
MODEL=$MODEL REPO=$REPO \
  bash runs/scripts/run_sweep.sh hima    /tmp/out_orig 4 "4 8 16 32 64 128"

# plot
source "$REPO/.venv/bin/activate"
python runs/scripts/plot.py /tmp/out      /tmp/out/figures
python runs/scripts/plot.py /tmp/out_orig /tmp/out_orig/figures
```

---

## Scripts

| file | role |
|---|---|
| `scripts/start_server.sh` | start vLLM serve (baseline or hima), controlled by env vars |
| `scripts/workload1.sh` | W1 multi-turn sweep; prefix sizes auto-selected from `VLLM_UTIL` |
| `scripts/workload2.py` | W2 concurrent SWE-Bench-Lite driver |
| `scripts/run_sweep.sh` | end-to-end driver: server → W1 → W2 → shutdown |
| `scripts/plot.py` | generate W1/W2 plots + markdown summary table |

---

## Results

### Original sweep (util=0.85, 4 clients, turns 4–128)

Figures: `data/figures/original/w1.png`, `data/figures/original/w2.png`

**W1:** Hit rate rises monotonically with turns (41% → 93%). HiMA ≈ baseline throughout.  
**W2:** Hit rate 14–34% across conc 1–64. HiMA ≈ baseline; partial-cache did not trigger
(low turn count → small partial-block tails).

### Aggressive sweep (util=0.55, 16 clients, 16-turn agents, conc 1–128)

Figures: `data/figures/aggressive/w1.png`, `data/figures/aggressive/w2.png`

**W1 (baseline):**

| turns | hit% | p95 TTFT ms | preempt |
|---:|---:|---:|---:|
| 4 | 47.9% | 3200 | 0 |
| 8 | 65.0% | 967 | 0 |
| 16 | 75.5% | 691 | 0 |
| 32 | 84.2% | 356 | 0 |
| 64 | 91.1% | 424 | 0 |

**W1 (HiMA vs baseline):**

| turns | baseline hit% | HiMA hit% | p95 TTFT Δ |
|---:|---:|---:|---:|
| 4 | 47.9% | 46.3% | +3.9% |
| 8 | 65.0% | 65.2% | +24.8% |
| 16 | 75.5% | 75.5% | +3.9% |
| 32 | 84.2% | **44.2% 🔴** | **+193%** |
| 64 | 91.1% | **56.5% 🔴** | **+217%** |

**W2 (HiMA vs baseline, 16 turns/agent):**

| conc | baseline hit% | HiMA hit% | runtime Δ |
|---:|---:|---:|---:|
| 1–2 | ~80% | ~79% | +31–38% |
| 4 | 82.1% | **57.6% 🔴** | +31% |
| 8 | 80.5% | 79.0% | +65% |
| 16 | 81.3% | **32.0% 🔴** | +33% |
| 32 | 81.1% | **42.0% 🔴** | +80% |
| 64 | 80.8% | **13.3% 🔴** | +142% |
| 128 | 19.5% | 16.4% | +7% |

### Key findings

1. **LPB (L1) hurts at util=0.55 + high concurrency.** With reduced KV pool, LPB evicts shared prefixes before LRU because their "hit count" decays faster than per-conversation unique blocks in the windowed counter.

2. **Partial cache (L2) never triggered** (`dedup_hit=0` throughout). Two likely reasons:
   - Qwen3.5-35B-A3B may map to two KV-cache groups (full-attn + mamba), but `_try_partial_extension` requires exactly one group.
   - `VLLM_PARTIAL_CACHE_MIN_R=256` threshold was not met for our workload's block-alignment pattern.

3. **HiMA's design scenario** requires: one high-value anchor prefix warmed 500× **before** cold-burst traffic evicts it under LRU. That pattern does not arise in diverse concurrent workloads — it's reproducible via `dev/compare_lru_lpb.py` Phase A→H pipeline (see `dev/intralayer/vllm.md`).
