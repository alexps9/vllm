# HiMA Evaluation on Qwen3.5-35B-A3B

Branch: `hima` · Model: `Qwen/Qwen3.5-35B-A3B` · Hardware: 8× RTX PRO 6000 Blackwell 96 GB (TP=2)

## Overview

Two workloads × four server configs (baseline / l1_only / l2_only / full) × two pressure settings (original / aggressive).
- **baseline**: stock vLLM
- **l1_only**: HiMA L1 (LPB intra-pool eviction + path counter)
- **l2_only**: HiMA L2 (admitter + budgeter + planner)
- **full**: both L1 and L2

| | original | aggressive |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | **0.55** |
| `max_num_seqs` | 64 | **256** |
| W1 clients | 4 | **16** |
| W1 common_prefix | 512 tok | **2048 tok** |
| W2 turns/agent | 4 | **16** |
| W2 mock obs size | ~0.1 KB | **~1.5–3 KB** |

**HiMA env flags** (per layer; set both sub-flags for the full stack):
```
VLLM_HIMA_L1_ENABLE=1   VLLM_HIMA_L2_ENABLE=1   VLLM_HIMA_HPB_WINDOW_S=3600
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
  bash runs/scripts/run_sweep.sh full /tmp/out

# original sweep (util=0.85, 4 clients)
MODEL=$MODEL REPO=$REPO \
  bash runs/scripts/run_sweep.sh baseline /tmp/out_orig 4 "4 8 16 32 64 128"
MODEL=$MODEL REPO=$REPO \
  bash runs/scripts/run_sweep.sh full    /tmp/out_orig 4 "4 8 16 32 64 128"

# plot
source "$REPO/.venv/bin/activate"
python runs/scripts/plot.py /tmp/out      /tmp/out/figures      baseline,full
python runs/scripts/plot.py /tmp/out_orig /tmp/out_orig/figures baseline,full
```

---

## Scripts

| file | role |
|---|---|
| `scripts/start_server.sh` | start vLLM serve in mode `baseline` / `l1_only` / `l2_only` / `full`, controlled by env vars |
| `scripts/workload1.sh` | W1 multi-turn sweep; prefix sizes auto-selected from `VLLM_UTIL` |
| `scripts/workload2.py` | W2 concurrent SWE-Bench-Lite driver |
| `scripts/run_sweep.sh` | end-to-end driver: server → W1 → W2 → shutdown |
| `scripts/plot.py` | generate W1/W2 plots + markdown summary table |

---

## Results

### Original sweep (util=0.85, 4 clients, turns 4–128)

Figures: `data/figures/original/w1.png`, `data/figures/original/w2.png`

**W1:** Hit rate rises monotonically with turns (41% → 93%). HiMA ≈ baseline throughout.  
**W2:** Hit rate 14–34% across conc 1–64. HiMA ≈ baseline.

### Intralayer-only sweep (util=0.55, baseline vs hima_l1, n=1)

Isolates L1 LPB from L2. Same aggressive workload as below.  
Data: `data/intralayer_aggressive/{baseline,hima_l1}/{w1,w2}/`  ·  Figures: `data/figures/intralayer_aggressive/`

**W1** (multi-turn):

| turns | baseline hit% | hima_l1 hit% | baseline p95 TTFT | hima_l1 p95 TTFT |
|---:|---:|---:|---:|---:|
| 4 | 47.8% | 46.4% | 2482 | 1754 |
| 8 | 64.8% | 65.0% | 1074 | 966 |
| 16 | 75.5% | 75.6% | 742 | 700 |
| 32 | 84.1% | **46.8% 🔴** | 354 | **949 🔴** |
| 64 | 91.0% | **58.5% 🔴** | 401 | **1266 🔴** |

**W2** (concurrent agents, 16 turns each):

| conc | baseline hit% | hima_l1 hit% | baseline tok/s | hima_l1 tok/s |
|---:|---:|---:|---:|---:|
| 1 | 81.1% | 81.1% | 187 | 189 |
| 2 | 80.7% | 76.9% | 219 | 214 |
| 4 | 80.5% | 79.7% | 296 | 303 |
| 8 | 81.4% | 80.1% | 375 | 376 |
| 16 | 80.8% | **25.1% 🔴** | 422 | **287 🔴** |
| 32 | 81.0% | **39.8% 🔴** | 564 | **337 🔴** |
| 64 | 80.4% | **21.3% 🔴** | 686 | **346 🔴** |
| 128 | 19.9% | 15.6% | 358 | 319 |

**Verdict**: L1 LPB alone reproduces the same degradation pattern as full HiMA — the regression is in **L1**, not in L2. LPB ties baseline at low load (W1 turns ≤ 16, W2 conc ≤ 8) but collapses cache hit rate at moderate-to-high load (W1 turns ≥ 32, W2 conc ≥ 16). HiMA's intended win pattern requires a single high-value anchor warmed before cold-burst pressure (see `dev/intralayer/vllm.md`), which neither W1 nor W2 produces.

### Aggressive sweep, full HiMA (util=0.55, 16 clients, 16-turn agents, conc 1–128)

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

1. **L1 LPB is the source of the regression.** The intralayer-only sweep (`hima_l1` mode, no L2) reproduces the same hit-rate collapse and TTFT spike as full HiMA on W1 turns ≥ 32 and W2 conc ≥ 16. With a reduced KV pool, LPB's windowed hit counter decays faster on shared prefixes than on per-conversation unique blocks, so LPB evicts the wrong blocks under diverse concurrent traffic.

2. **HiMA's design scenario** requires: one high-value anchor prefix warmed 500× **before** cold-burst traffic evicts it under LRU. That pattern does not arise in diverse concurrent workloads — it's reproducible via `dev/intralayer/compare_lru_lpb.py` Phase A→H pipeline (see `dev/intralayer/vllm.md`).
