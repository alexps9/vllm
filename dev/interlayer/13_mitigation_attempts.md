# Finding M.13 — bq mitigation doesn't help; root cause is dispatch fragmentation

After M.12 hypothesized that increasing `batch_queue_size` would close
the +14% regression, M.13 tested it directly: did NOT help. The fix
must address a different aspect of the problem.

## Setup

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
# 4 fresh runs back-to-back on the same GPU after reboot:
for env in "" "VLLM_PARTIAL_CACHE_ENABLED=1" "VLLM_BATCH_QUEUE_SIZE=3" "VLLM_BATCH_QUEUE_SIZE=3 VLLM_PARTIAL_CACHE_ENABLED=1"; do
    env $env CUDA_VISIBLE_DEVICES=1 KMP_AFFINITY=disabled \
        .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py
done
```

## Results

| mode | TTFT (s) | full (s) | tok/s | hit% | bubble |
|---|---|---|---|---|---|
| bq=2 baseline | 5.81 | 14.24 | 156.3 | 91.59 | 51442 |
| bq=2 partial  | 4.84 | 16.13 | 138.0 | 98.99 |  6147 |
| **bq=3 baseline** | 5.84 | 14.21 | 156.6 | 91.59 | 51442 |
| **bq=3 partial**  | 4.85 | 16.14 | 137.9 | 98.99 |  6147 |

| comparison | TTFT Δ | full Δ | tok/s Δ |
|---|---|---|---|
| bq=2 partial vs baseline | -16.78% | **+13.28%** | -11.72% |
| bq=3 partial vs baseline | -16.99% | **+13.59%** | -11.97% |
| bq=3 vs bq=2 (baseline)  |  +0.49% |  -0.21% |  +0.21% |
| bq=3 vs bq=2 (partial)   |  +0.23% |  +0.07% |  -0.07% |

bq=3 has zero effect, in both baseline and partial modes. The
M.12 mitigation hypothesis is **falsified**.

## Why bq=3 doesn't help

The cc workload is single-stream (one cc session at a time, with
`max_num_seqs=4` engine config). Each turn issues a request, the
engine processes it (prefill + decode), then the next turn arrives.
At any moment there's typically **only 1 active request in flight**.

vLLM v1's batch_queue holds <= N batches in flight. With 1 active
request, the queue holds AT MOST 1 batch — so making the queue
deeper (2 → 3) doesn't change anything. The scheduler can't fill
the deeper queue with phantom requests.

## Re-classifying the main-thread samples

Categorizing py-spy main-thread samples by engine function and
converting to wall-time-attributed ms (samples × ms_per_sample, where
ms_per_sample = wall_ms / total_main_samples for each run):

| category | baseline% | partial% | Δ% | Δ ms |
|---|---|---|---|---|
| **cuda_synchronize** | 38.56 | 40.89 | +2.33 | **+1154 ms** |
| **inductor_compiled (Python dispatch)** | 18.60 | 19.36 | +0.76 | **+497 ms** |
| engine_step | 1.25 | 1.89 | +0.64 | +131 ms |
| kv_cache_update | 1.88 | 2.27 | +0.40 | +103 ms |
| execute_model_other | 4.83 | 4.86 | +0.03 | +102 ms |
| prepare_inputs | 2.80 | 2.79 | -0.01 | +55 ms |
| sample | 8.66 | 7.85 | -0.80 | +41 ms |
| attention_forward | 5.49 | 4.93 | -0.56 | +17 ms |
| other | 10.94 | 9.71 | -1.23 | +17 ms |
| attn_metadata | 2.84 | 2.41 | -0.42 | -13 ms |
| block_table_copy | 4.16 | 3.03 | -1.13 | -103 ms |

Sum of positive deltas: ~+2.1 s (matches the wall regression).

Top 2 contributors:
- **cudaSynchronize wait: +1154 ms** (main thread blocked on GPU)
- **Python dispatch in inductor wrappers: +497 ms** (more calls to
  compiled kernels because more batches)

Together ~1.65 s of the ~2 s wall regression.

## Refined root cause

The previous M.11/M.12 hypothesis was "partial cache fragments
batches → more sync points". M.13 refines this:

**Partial cache reduces work per scheduling step** (because cached
partial tokens are skipped during prefill). Per the cc workload's
single-stream structure, the engine still issues the **same number
of scheduling steps** per turn (prefill chunk + decode steps). But
each step now does **less GPU work**, while paying the **same fixed
overhead** per step:

1. CUDA stream synchronize (~50-100 µs fixed Python overhead per
   sync point, × all the sync points)
2. PyTorch Inductor compiled-kernel dispatch wrapper (~10-50 µs of
   Python per launch)

When GPU work per step shrinks, these fixed overheads become a
larger fraction of wall time. async_scheduling tries to hide them
by keeping 2 batches in flight, but for single-stream workloads
there isn't a "next batch" to overlap with, so the main thread
blocks.

The total work (decode steps) is similar between modes; the
**ratio of fixed overhead to useful work** is what shifts.

## What WOULD help (un-tested)

- **Multi-stream workloads** where multiple requests can fill the
  batch queue — partial cache shouldn't regress because batches
  stay big enough to amortize fixed overhead.
- **Coalesce admissions** at the scheduler level — multiple new
  requests submitted as one batch admission to avoid per-admission
  fragmentation.
- **Skip partial-cache application** for tiny extensions where the
  prefill savings don't outweigh the per-step overhead — heuristic-
  based, would need calibration per model/hardware.
- **Architectural: reduce per-launch fixed overhead** in the engine
  (out of scope for this work).

## Recommendation

The partial-cache regression is inherent to **single-stream + already-
small prefill** workloads. For these:

- TTFT win is real (-17%), useful for interactive latency.
- Throughput cost is real (-14%), undesirable for bulk generation.

For **multi-stream + larger workloads** (typical production), the
regression should self-amortize and partial-cache should be a
net win. This needs validation but is out of scope for this
investigation.

The `VLLM_PARTIAL_CACHE_ENABLED` env var stays opt-in. Operators
should profile their workload before enabling.

## Committed artifacts

- `dev/interlayer/runs/m13/{bq2,bq3}_{baseline,partial}.{jsonl,out}`
  — 4 fresh runs
- `vllm/v1/executor/uniproc_executor.py` — `VLLM_BATCH_QUEUE_SIZE`
  env override (kept for future experiments)
- This doc

## Methodology note: GPU contention is real

The host has unkillable zombie processes from prior runs (kill -9
returns success but the process stays in R state with 0% CPU).
Multiple users sharing GPU also matters — use `nvidia-smi
--query-compute-apps=pid,used_memory` to pick a free GPU. M.13 used
GPU 1 because GPU 0 had a 115GB tenant.
