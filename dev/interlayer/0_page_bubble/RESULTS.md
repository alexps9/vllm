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

## A / B / C — live-engine findings (RE-CONFIRMED on GPU 2026-06-01, Qwen3.5-35B-A3B, TP=2)

- **A. Attention page = 1056.** Live engine prints verbatim (both TP
  workers): *"Setting attention block size to 1056 tokens to ensure that
  attention page size is >= mamba page size"* + *"Padding mamba page size by
  0.76% to ensure mamba page size and attention page size are exactly
  equal."* Root cause: SSM temporal state stored in **fp32** → large mamba
  page. **Nuance:** `cache_config.block_size` *reports* 16, but the
  attention kernel page and the prefix-cache reuse granularity are **1056**
  (the `gcd`/`hash_block_size`); B confirms the waste lives at 1056.
- **B. Hit length rounds to 1056.** Live `num_cached_tokens` on re-issue
  matched `floor((L-1)/1056)·1056` exactly — L=1087/1088/1500/2000 all →
  1056 (4/4). Short prompts (<1056) get **0** cached → 100% recompute.
- **C. Per-turn waste ≈ block_size/2 = 528 tokens**, independent of turn
  size (live, 4 regimes × 20 turns):

  | regime (tok/turn) | avg waste/turn | waste / new tokens added |
  |---|---:|---:|
  | micro (50)  | 560.5 | **1121%** |
  | short (150) | 459.9 | 307% |
  | medium (500)| 552.6 | 110% |
  | long (3000) | 543.6 | 18% |

  The `waste/new-tokens` column is the agent-traffic killer: small-increment
  multi-turn (the common agent pattern) wastes >10× the tokens it adds.

> Status: **all four findings re-confirmed** — A/B/C on the live engine
> (2026-06-01), D offline byte-identical. Raw:
> `runs/{hit_rate,multi_turn,inspect_sizes}_reconfirm.out`.

## So what

This is the **memory** cost of the bubble (wasted KV capacity → fewer
concurrent reqs / smaller effective prefix cache). The earlier pcache effort
targeted the *compute* cost (recomputing the ragged tail) and failed on
hybrid because mamba is block-granular (can't resume mid-block;
[`08_hybrid_architectural_blocker.md`](08_hybrid_architectural_blocker.md)).
The memory bubble is what the interlayer effort should target — by breaking
the uniform-page constraint (see [`../design.md`](../design.md) "The lever").
