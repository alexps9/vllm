# Finding M.15 — `VLLM_PARTIAL_CACHE_MIN_R` threshold heuristic works

M.13 ruled out batch_queue_size mitigation. M.14 ruled out multi-stream
amortization. M.15 tests M.13's last suggested mitigation: **skip
partial-cache hits whose R is below a threshold** because tiny hits
don't recover enough prefill work to amortize the per-launch fixed
overhead.

Result: **this works**. r=256 is a Pareto improvement over r=0 on
the cc workload — beats the default partial-cache on BOTH TTFT and
throughput simultaneously.

## Implementation

`vllm/v1/core/kv_cache_manager.py` — `_try_partial_extension`:

```python
min_r = int(os.environ.get("VLLM_PARTIAL_CACHE_MIN_R", "0"))
for R, _candidate_block in candidates:
    if R > max_extension or R < min_r:  # ← threshold gate
        continue
    ...
```

Default `min_r=0` preserves existing behavior. Operator sets the env
var to enable threshold-gating.

## Sweep on cc workload (single-stream, Qwen3-8B, block_size=1024)

| config | TTFT (s) | full (s) | tok/s | hit% | bubble | ΔTTFT | Δfull |
|---|---|---|---|---|---|---|---|
| baseline (env unset)       | 5.81 | 14.24 | 156.3 | 91.59 | 51442 | — | — |
| partial r=0 (default)     | 4.84 | 16.13 | 138.0 | 98.99 |  6147 | -16.78% | +13.28% |
| partial r=128             | 4.78 | 16.12 | 138.1 | 98.94 |  6481 | **-17.74%** | +13.19% |
| **partial r=256**         | **4.79** | **15.66** | **142.1** | 98.54 |  8919 | **-17.50%** | **+9.98%** |
| partial r=512             | 4.91 | 15.33 | 145.2 | 97.28 | 16655 | -15.54% | +7.65% |
| partial r=800             | 5.34 | 15.19 | 146.5 | 94.90 | 31208 |  -8.06% | +6.67% |
| partial r=1024 (≈off)     | 5.75 | 14.26 | 156.1 | 91.59 | 51442 |  -1.00% |  +0.12% |

## Interpretation

- **r=0 (current default)**: applies every partial hit ≥ 1 token. Big
  TTFT win (-17%), big throughput hit (+13%).
- **r=256**: skips hits with R < 256. Still catches most of the
  benefit (hit% 98.54% vs 98.99%; only 2772 extra bubble tokens out
  of 51442 = 5%). TTFT actually **improves** (-17.5%) because we
  avoid the small-extension cases that were costing more in overhead
  than they saved. Throughput regression nearly halves (+10% vs +13%).
- **r=512**: continues to recover throughput (+7.65%) at modest TTFT
  cost (-15.5% vs -17.5%).
- **r=800**: most throughput recovery (+6.67%) but TTFT win cut in
  half (-8%).
- **r=1024** (= block_size): effectively disables partial cache.
  Baseline performance recovered. Confirms the env var works.

The "knee" of the curve is around r=256-512. r=256 is a Pareto
improvement over the un-thresholded partial cache: better TTFT AND
better throughput.

## Why r=256 outperforms r=0 on TTFT

The TTFT pass measures wall time of `LLM.generate(max_tokens=1)`.
For r=0, every turn applies the partial-cache hit, even when R is
small (say 32). The Python overhead (lookup, dict ops, KV cache
manager bookkeeping) for applying that small hit COSTS MORE than
the 32-token prefill savings. So small-R hits make TTFT worse
overall.

At r=256, those small-R hits are skipped, the small-R requests fall
back to baseline (re-prefill), and TTFT for THAT subset of turns is
faster than the "apply" path. Net per-turn TTFT improves.

For larger R values (256+), the prefill savings dominate the
overhead so applying is still net positive on TTFT.

## Trade-off matrix updated (now including thresholded partial)

For cc-like single-stream interactive workloads:

| workload need | best config |
|---|---|
| best TTFT, accept throughput cost | `VLLM_PARTIAL_CACHE_ENABLED=1 VLLM_PARTIAL_CACHE_MIN_R=256` |
| balanced TTFT + throughput | `VLLM_PARTIAL_CACHE_ENABLED=1 VLLM_PARTIAL_CACHE_MIN_R=512` |
| max throughput, sacrifice TTFT | env unset (no partial cache) |

For multi-stream / bulk throughput workloads: env unset is best
(M.14 showed multi-stream amplifies regression — threshold likely
helps there too but not tested).

## Why this is the "right" knob

The threshold acts as a cost/benefit gate at admission time. Every
partial-cache hit has:
- **Benefit**: R tokens of attention prefill saved (≈ R × per-token-
  prefill-cost)
- **Cost**: fixed per-launch overhead from smaller batches downstream
  (≈ constant, regardless of R)

For small R, cost > benefit. For large R, benefit > cost. The
threshold cleanly separates the regimes.

Calibration depends on:
- Model size (per-token prefill cost scales)
- Hardware (per-sync overhead constant)
- Workload (single-stream vs multi-stream amplifies cost)

`r=256` is a reasonable starting point for Qwen3-8B / H200 / single
stream. Operators should calibrate for their setup.

## Files

- `vllm/v1/core/kv_cache_manager.py` — VLLM_PARTIAL_CACHE_MIN_R env var
- `dev/interlayer/runs/m15/single_partial_r{256,512,800,1024}.{jsonl,out}`
- This doc

## What's still untested

- Multi-stream + threshold combination — likely also improves
- Other workloads (RAG with long prefill, bulk gen, etc.) — different
  optimum

## r=128 vs r=256 — which is "best"?

| metric | r=128 | r=256 |
|---|---|---|
| TTFT delta | **-17.74%** | -17.50% |
| full delta | +13.19% | **+9.98%** |
| throughput delta | -11.66% | **-9.18%** |
| hit% | 98.94% | 98.54% |

r=128 is marginally better on TTFT (~0.2 percentage points) but the
throughput regression is essentially unchanged from r=0 (no
mitigation). r=256 sacrifices that tiny TTFT difference but cuts
throughput regression by a meaningful 3 percentage points (24%
relative reduction in regression magnitude).

**r=256 is the recommended starting point**. r=128 is only worth
considering if you have absolutely zero tolerance for TTFT
regression and don't care about throughput at all.
