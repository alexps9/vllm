# page_bubble — prove the page-size bubble exists in vLLM

## Claim

On a hybrid model (attention + mamba), vLLM forces a **uniform block byte
size** = `max(attention_page, mamba_page)`. The mamba state (fp32, one
indivisible state per req) is the larger page, so vLLM **inflates the
attention block_size** — to **1056 tokens** on Qwen3.5-35B-A3B (66× the
default 16). Attention KV is then allocated in 1056-token blocks that real
requests fill only fractionally → **blocks all allocated, but a large share
of slots inside them empty.** That internal fragmentation is the bubble.

## What's measured (4 findings)

| # | finding | how |
|---|---|---|
| **A** | attention page / prefix-cache granularity inflates to **1056** (engine verbatim: *"Setting attention block size to 1056 tokens…"*); root = fp32 SSM state. (`cache_config.block_size` reports 16, but the kernel page / hit granularity is 1056) | `inspect_sizes.py` (offline) + live engine log |
| **B** | prefix-cache hit length rounds to **1056 multiples** (`floor((L-1)/1056)*1056`) | `hit_rate_microbench.py` (live engine, 13/13 prompt lengths) |
| **C** | per-turn partial-block waste ≈ **block_size/2 = 528 tokens/turn**, independent of session length | `multi_turn_waste.py` (live engine, 80/80 turns) |
| **D** | on **106 real Claude-Code sessions**: workload-weighted waste **42.6%**, p95 **>130%** | `real_session_waste.py` / `counterfactual_block_size.py` (offline; tokenizer + traces) |

The counterfactual (D) is the headline — it isolates the bubble as a pure
function of block_size:

| block_size | workload-weighted waste | p95 session |
|---:|---:|---:|
| 16 (vLLM default) | 0.69% | 2.1% |
| 256 | 10.5% | 31% |
| 512 | 21.0% | 63% |
| **1056 (forced, hybrid)** | **42.6%** | **130%** |

## Repro

```bash
# D — offline, no GPU (re-validated 2026-05-31, byte-identical to the
# original commit 438ad0397):
.venv/bin/python dev/interlayer/0_feasibility/page_bubble/counterfactual_block_size.py
.venv/bin/python dev/interlayer/0_feasibility/page_bubble/real_session_waste.py

# A — offline size arithmetic:
.venv/bin/python dev/interlayer/0_feasibility/page_bubble/inspect_sizes.py

# A/B/C — need a live hybrid engine on Qwen3.5-35B-A3B (GPU):
.venv/bin/python dev/interlayer/0_feasibility/page_bubble/hit_rate_microbench.py
.venv/bin/python dev/interlayer/0_feasibility/page_bubble/multi_turn_waste.py
```

Trace dataset: `dev/intralayer/cc_long_traces.jsonl` (106 real CC sessions).
See [`RESULTS.md`](RESULTS.md) for the full re-validated numbers and
[`08_hybrid_architectural_blocker.md`](08_hybrid_architectural_blocker.md)
for why the prior pcache fix could not address this on hybrid.

## Provenance

Restored from commit `438ad0397` ("empirical study of vLLM hybrid-model
BlockPool inflate"), which was deleted with the pcache tree (`7d974f6f6`)
when pcache (the *wrong* fix) was removed. The bubble proof is sound and
motivates the interlayer effort (see [`../../design.md`](../../design.md)).
