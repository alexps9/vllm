# page_bubble — RESULTS

> ⚠️ **CORRECTED + interlayer CLOSED — read [`../../POSTMORTEM.md`](../../POSTMORTEM.md).**
> The "42.6%" headline below was **mislabeled as memory**; it is actually a
> **recompute ratio** (`Σ(L mod 1056) / Σ(new content per turn)` —
> `counterfactual_block_size.py:126-136`): the partial-block *tail* recomputed
> on each re-issue ÷ new content. The **true memory bubble (internal
> fragmentation), live-measured** (`live_bubble_snapshot.py`): **~5% at short/mid
> resident contexts, ~0.5% at 100k** — not 42.6%. The recompute tail is real
> (~30 ms/turn, `ttft_tail_vs_context.py`) but **mamba-bound / unfixable** by
> the attention sub-block approach on hybrid (`08_hybrid_architectural_blocker.md`).
> Findings A–D below are accurate *as measurements*; only the "memory" framing
> was wrong. The interlayer effort is closed (see POSTMORTEM).

**Verdict (original, framing corrected above):** the forced uniform page
inflates attention block_size to 1056 tokens; the partial-block tail recomputed
each re-issue is **42.6% of new content** on 106 real Claude-Code sessions
(vs 0.69% at block_size=16) — a **recompute**, not memory, quantity.

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

## So what (CORRECTED — see [`../../POSTMORTEM.md`](../../POSTMORTEM.md))

This 42.6% is the **recompute** cost (the partial-block tail re-prefilled each
re-issue), **not** a memory quantity — the original wording here ("the memory
cost… the memory bubble is what the interlayer effort should target") was wrong.
The **true memory bubble is ~0.5–5%** (live-measured, `live_bubble_snapshot.py`).
And the recompute cost is exactly what the earlier pcache effort targeted and
**cannot be fixed on hybrid** — mamba is block-granular (can't resume mid-block;
[`08_hybrid_architectural_blocker.md`](08_hybrid_architectural_blocker.md)), so
finer *attention* allocation doesn't help. **The interlayer effort is closed.**
The real gap vs sglang is prefix-cache granularity (sglang's radix caches the
exact turn-end; vLLM rounds to 1056), which is a vLLM-core re-architecture
outside scope. Full reasoning + decision: [`../../POSTMORTEM.md`](../../POSTMORTEM.md).
