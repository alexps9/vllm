# 0_page_bubble — RESULTS

**Verdict: the page-size bubble is real and large.** On Qwen3.5-35B-A3B the
forced uniform page inflates attention block_size to 1056 tokens, producing
**42.6% workload-weighted KV waste on 106 real Claude-Code sessions**
(vs 0.69% at the natural block_size=16). p95 session >130%.

## D — counterfactual waste vs block_size (re-validated 2026-05-31, offline)

106 real CC sessions, incremental per-message tokenization, waste =
Σ(ceil(cumlen/B)·B − cumlen) / Σ cumlen.

| block_size | workload-weighted | total waste tokens | median | p95 |
|---:|---:|---:|---:|---:|
| 16 (vLLM default) | 0.69% | 76,113 | 0.3% | 2.1% |
| 32 | 1.34% | 148,081 | 0.7% | 4.0% |
| 64 | 2.61% | 288,977 | 1.4% | 7.7% |
| 128 | 5.26% | 582,289 | 2.7% | 15.4% |
| 256 | 10.52% | 1,163,921 | 5.1% | 31.3% |
| 512 | 20.95% | 2,316,945 | 10.7% | 62.8% |
| **1056 (forced)** | **42.62%** | **4,713,841** | **20.7%** | **129.6%** |
| 2112 | 85.10% | 9,411,985 | 42.8% | 270.1% |

**Byte-identical to the original (commit `438ad0397`)** — the measurement
reproduces exactly on current code + traces. Raw:
[`runs/counterfactual_revalidate.out`](runs/counterfactual_revalidate.out),
[`runs/counterfactual_block_size.json`](runs/counterfactual_block_size.json),
fig `runs/fig_block_size_counterfactual.png`.

Reading: at the forced 1056 block, for every 100 tokens actually used,
~42.6 token-slots of KV are allocated-but-empty (~30% of allocated KV is
dead). The worst sessions (short, ragged) allocate >2× their real tokens.

## A / B / C — live-engine findings (from commit `438ad0397`; not yet re-run on GPU this session)

- **A. Inflation = 1056.** Engine prints verbatim: *"Setting attention
  block size to 1056 tokens to ensure that attention page size is >= mamba
  page size."* Root cause: SSM temporal state stored in **fp32** (not bf16),
  making the mamba page large. 66× the vLLM default of 16.
- **B. Hit length rounds to 1056.** On the running engine, 13/13 prompt
  lengths (100..10000) matched `floor((L-1)/1056)*1056` exactly — prefix
  cache only reuses at 1056-token boundaries.
- **C. Per-turn waste ≈ 528 tokens.** Multi-turn agent partial-block waste
  averages block_size/2 = 528 tokens/turn, independent of session length
  (80/80 turns across 50/150/500/3000-tok regimes).

> Status: **D re-validated offline this session** (the headline). A/B/C were
> validated on the live engine in the original study; a GPU re-run
> (`hit_rate_microbench.py`, `multi_turn_waste.py`) is the remaining
> confirmation — cheap, deferred.

## So what

This is the **memory** cost of the bubble (wasted KV capacity → fewer
concurrent reqs / smaller effective prefix cache). The earlier pcache effort
targeted the *compute* cost (recomputing the ragged tail) and failed on
hybrid because mamba is block-granular (can't resume mid-block;
[`08_hybrid_architectural_blocker.md`](08_hybrid_architectural_blocker.md)).
The memory bubble is what the interlayer effort should target — by breaking
the uniform-page constraint (see [`../design.md`](../design.md) "The lever").
