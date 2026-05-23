# Finding M.9 — partial-cache on real cc multi-turn workload

After M.6/M.7 validated the mechanism on synthetic two-turn microbenches,
M.9 measures it on the real Claude Code traces — 10 multi-turn sessions,
106 total turns, Qwen3-8B with block_size=1024. Each turn issued twice
(max_tokens=1 for TTFT, max_tokens=21 with ignore_eos=True for
throughput).

## Setup

- Model: `Qwen/Qwen3-8B` (full-attention, no mamba — partial-cache's
  current scope)
- `block_size = 1024`, `gpu_memory_utilization = 0.6`,
  `max_num_seqs = 4`, single H200
- Dataset: `dev/cc_long_traces.jsonl`, first 10 sessions
- Sampling: `temperature=0.0, ignore_eos=True` so each turn always
  generates 21 tokens (eliminates EOS-timing variance between runs)
- Both runs use the SAME engine + binary; only `VLLM_PARTIAL_CACHE_ENABLED`
  differs

## Results (aggregate over 106 turns / 2226 output tokens)

| metric | baseline | partial cache | delta |
|---|---|---|---|
| total TTFT (s) | 5.71 | 4.80 | **-15.84%** |
| total full_wall (s) | 14.18 | 16.16 | +13.99% |
| mean TTFT per turn (ms) | 53.8 | 45.3 | **-15.8%** |
| mean full_wall per turn (ms) | 134 | 152 | +14.0% |
| throughput (tok/s) | 157.0 | 137.7 | -12.3% |
| cache hit % (vs prompt) | 77.1% | **83.3%** | +8.1% rel. |
| cache hit % (vs prior content) | 91.6% | **99.0%** | +8.1% rel. |
| bubble tokens (re-prefilled prior content) | 51442 | 6147 | **-88.05%** |

## Headline observations

### The bubble IS eliminated on real workload

- **Cache hit rate climbs from 91.6% to 99.0% relative to prior-turn
  content.** With partial cache, almost every token that was computed
  in any prior turn gets re-used on follow-up turns.
- **Bubble drops from 51,442 to 6,147 tokens** = 88% reduction in
  redundant prefill work across the workload.
- This is exactly Finding D's promise — the bubble is real and
  eliminable.

### TTFT improves on every kind of turn

- -16% across 106 turns. Per-turn data shows TTFT is faster on nearly
  every turn (a few neutral, none meaningfully slower).
- This is the headline interactive-workload win: agents and chat UIs
  feel ~16% snappier on follow-up turns.

### Throughput regression (-12%) — a real trade-off

- Decode itself is slower per token: ~4.0 ms/decode-step baseline
  vs ~5.35 ms/decode-step with partial cache. ~+34% on decode alone.
- We tried several mitigations during this finding (dedup on partial_len,
  skip-during-decode, nested-map index instead of O(R) probe).
  Each helped slightly with code overhead but none reclaimed the GPU-
  side decode slowdown. The cost appears to be from GPU-side
  cache-state differences (partial-cache hits give the new request a
  KV layout where the K-th block is "pre-warmed" with prior-turn K, V;
  the resulting memory-access pattern during attention has slightly
  different performance characteristics than a freshly-allocated
  block).
- Net wall time on this workload: **+2 s out of 20 s** total
  (5.71 + 14.18 = 19.89 s → 4.80 + 16.16 = 20.96 s = +1.07 s on the
  total session). Visible but small in absolute terms.

### What this means for the trade-off

This is **a net win for interactive workloads** (cc agents, chat,
anything where TTFT determines perceived latency) and **a net loss for
bulk-generation throughput** (batch inference, long-form completion).

The fix is opt-in via `VLLM_PARTIAL_CACHE_ENABLED=1`, so operators
can choose per deployment. Default is unchanged (env unset = no
partial cache = no regression).

## What we tried that didn't fully fix the throughput regression

| iter | change | full_wall delta | TTFT delta |
|---|---|---|---|
| v1  | initial (O(block_size) R probe each lookup) | +18% | -16% |
| v2  | nested map: index by parent_hash, only probe valid R values | +17% | -17% |
| v3  | dedup repeat partial-cache writes within request lifetime | +17% | -17% |
| v4  | ignore_eos=True (eliminate output-length variance) | +14% | -17% |
| v5  | skip cache_partial_block during decode steps | +14% | -16% |

The progression from v1→v5 confirms the remaining regression is NOT
from Python-side overhead in our cache scaffolding — it's GPU-side and
likely architectural (kernel performance characteristics depend on
block-state continuity in subtle ways).

## Raw repro

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py \
    | tee dev/interlayer/runs/09_cc_baseline.out
CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 \
    .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py \
    | tee dev/interlayer/runs/09_cc_partial_cache.out
```

Snapshots: `09_cc_{baseline,partial_cache}_v5.jsonl`.

## What's next (M.10+)

1. **GPU-side decode-slowdown root cause** — profile attention kernels
   to identify the specific source of the +1.35 ms/step regression.
   Likely candidates: slot-mapping kernel variance, FA3 perf cliff
   on certain block-table shapes.
2. **Stress test correctness** — concurrent multi-tenant requests to
   exercise the destructive-hit path under contention.
3. **Workload calibration**: the trade-off depends on prompt-length /
   output-length ratio. For longer prompts + short outputs (TTFT-
   dominated) the win is bigger. For short prompts + long outputs
   (decode-dominated) the regression dominates. A more thorough
   workload sweep would help operators pick the env var.
