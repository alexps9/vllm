# vLLM hybrid-model KV cache size & prefix-cache behavior — investigation log

This directory holds the scripts and raw outputs from an empirical study of how
vLLM's `BlockPool` sizing behaves on `Qwen/Qwen3.5-35B-A3B` (a Gated DeltaNet +
Full-Attention hybrid model) and what that implies for prefix-cache hit rate
under different agent workloads.

The study was driven by a chain of questions:

1. What is `attn_page_size_1_token` / `mamba_page_size` / `block_size` etc.,
   and how do they relate in vLLM's unified `BlockPool`?
2. After the hybrid-padding inflate logic kicks in, how big does `block_size`
   actually become?
3. Does the coarsened `block_size` hurt the prefix cache?
4. How bad is that hit-rate hit in practice — short prefixes, long prefixes,
   long-running agents?

The answer to (2)/(3)/(4) was surprising enough that we wanted everything
reproducible end-to-end, hence this dir.

---

## Environment

* Working dir: `/data/yuzhou/projects/vllm-songyang`
* venv: `.venv/` (local, project-local — created with `uv venv --python 3.12`)
* vLLM install: `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
  (editable, points at `vllm/` in this repo)
* GPUs: 8× NVIDIA H200 144GB each (we used TP=2)
* Model: `Qwen/Qwen3.5-35B-A3B` (already in `/scratch/yuzhou/.cache/huggingface/`)

To run anything below, all you need is the venv:

```bash
.venv/bin/python dev/<script>.py
```

The hit-rate scripts spin up the engine (TP=2, mamba_cache_mode=align), which
takes ~3 min for model load + CUDA-graph capture, then runs in a few seconds.

---

## Scripts

### `inspect_sizes.py` — offline math, no GPU

Replicates the inflate arithmetic from `vllm/platforms/interface.py:613-674`
with no engine load. Prints `attn_page_size_1_token`, `mamba_page_size`,
inflated `block_size`, `lcm_block_size`, and a predicted hit-length table.

Important fix vs. an earlier draft: the **SSM temporal state is stored in
`float32` (not bf16)** for numerical stability. The conv state is in bf16.
Missing this halves `mamba_page_size` and gives the wrong inflated block
size. The dtypes come from `vllm.model_executor.layers.mamba.mamba_utils.
gated_delta_net_state_dtype` (defaults `mamba_ssm_cache_dtype="auto"` →
follows `mamba_cache_dtype` → follows `model_dtype` for conv, but
explicit fp32 for the SSM state).

Run:

```bash
.venv/bin/python dev/inspect_sizes.py qwen3.5-35b
```

### `trace_inflate.py` — replays inflate using vLLM's own helpers

To cross-check `inspect_sizes.py` against vLLM ground truth without
starting the engine: it builds a `vllm_config` via `EngineArgs.
create_engine_config()`, then constructs `FullAttentionSpec` and
`MambaSpec` exactly as `_align_hybrid_block_size` does, and prints
every intermediate.

This is the script that revealed the **fp32 SSM** detail (
`dtypes = (torch.bfloat16, torch.float32)`).

Run:

```bash
.venv/bin/python dev/trace_inflate.py
```

### `hit_rate_microbench.py` — single-shot prefix-cache hit verification

Spins up Qwen3.5-35B with `enable_prefix_caching=True`. For each target
prefix length `L`, builds a deterministic prompt of exactly `L` tokens,
primes the cache, re-issues, and reads `num_cached_tokens` off the
`RequestOutput`. Compares against `floor((L-1)/1056) × 1056`.

Run:

```bash
.venv/bin/python dev/hit_rate_microbench.py
```

### `multi_turn_waste.py` — multi-turn agent: per-turn partial-block waste

Simulates a multi-turn agent by extending a token-level prompt by a fixed
increment each "turn", issuing each turn as a fresh request (so cache hit
must come from prior turns' committed blocks, not from kept-alive KV).
Records `num_cached_tokens` at every turn and compares against the
partial-block-aware prediction. Tests five turn-size regimes
(50 / 150 / 500 / 3000 / 8000 tokens per turn, 20 turns each).

Run:

```bash
.venv/bin/python dev/multi_turn_waste.py
```

Outputs are saved to `dev/multi_turn_waste.out`.

### `real_session_waste.py` — apply the formula to real Claude Code traces

Loads 106 multi-turn agentic sessions from
`/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl`
(hyperswitch repo, Anthropic block format), tokenizes them, applies the
empirically-validated formula `floor((L - 1) / 1056) × 1056`, and reports
per-session and workload-weighted partial-block waste. No GPU needed —
the formula's correctness was nailed down in Finding C.

Run:

```bash
.venv/bin/python dev/real_session_waste.py
```

Outputs are saved to `dev/real_session_waste.out`.

---

## Findings

### A. The inflate produces `block_size = 1056` (×66 over vLLM default 16)

Authoritative numbers for `Qwen/Qwen3.5-35B-A3B`, TP=2:

```
attn_page_size_1_token   = num_kv_heads(per-worker=1) × (head_size + head_size_v)
                         × dtype_size(bf16) = 1 × (256+256) × 2 = 1024 B
mamba_page_size          = conv (bf16, 24 KiB) + ssm (fp32, 1024 KiB)
                         = 1048 KiB per worker
kernel_block_alignment   = 16
attn_block_size_inflated = 16 × ceil(1073152 / (16 × 1024))
                         = 16 × 66 = 1056 tokens
mamba_page_size_padded   = 1056 KiB        (waste 8 KiB / block = 0.76%)
lcm_block_size           = 1056            (prefix cache hit granularity)
```

Engine confirms verbatim:

```
INFO ... interface.py:645  Setting attention block size to 1056 tokens to
                           ensure that attention page size is >= mamba page size.
INFO ... interface.py:669  Padding mamba page size by 0.76% to ensure that
                           mamba page size and attention page size are exactly equal.
```

The often-quoted "vLLM padding 16-30%" figure from the HiMA paper is **not
the mamba block-tail waste** (that's 0.76% here) — it's the cumulative
effect of `block_size = 1056` on (a) short-prefix prefix-cache misses and
(b) per-request last-block waste, summed across the workload.

### B. Hit-rate microbench — `lcm = 1056` matches reality precisely

| shared prefix L | predicted hit | actual `num_cached_tokens` |
|---:|---:|---:|
| 100 | 0 | **0** |
| 300 | 0 | **0** |
| 500 | 0 | **0** |
| 543 | 0 | **0** |
| 544 | 0 | **0** |
| 545 | 0 | **0** |
| 800 | 0 | **0** |
| 1087 | 1056 | **1056** |
| 1088 | 1056 | **1056** |
| 1500 | 1056 | **1056** |
| 2000 | 1056 | **1056** |
| 5000 | 4224 | **4224** |
| 10000 | 9504 | **9504** |

Formula: `predicted = floor((L - 1) / 1056) × 1056`. **13/13 match.**

For shared prefixes < 1056 tokens (system prompts, tool definitions —
typical for agentic workloads), **vLLM has zero prefix-cache hit on this
model.** For long prefixes the waste drops to ~1% (just the last partial
block).

### C. Multi-turn agent — partial-block waste accumulates per turn

We simulate **four turn-size regimes, 20 turns each** — covering chat
ping-pong, ReAct loops, medium tool-call traffic, and long-form generation.
At each turn we issue a *fresh* request with the cumulative prompt and
read `num_cached_tokens`. The prediction is
`expected = floor((prev_total - 1) / 1056) × 1056`. **All 80 turns across
all four scenarios matched the prediction exactly (`diff = +0`).**

The longest scenario (`long_turn`, 3000 tok/turn) reaches 59K-token
context over 20 turns. For traffic closer to real long-horizon agents
(100K+ context across many turns), see Finding D below — where we apply
the validated formula to 106 actual Claude Code sessions.

The interesting column is `partial_block_waste = prev_total - cached` —
the number of tokens we *did* compute on the prior turn but the engine
*did not* cache, because they were stuck in a partial last block when the
prior request ended. Those tokens must be re-prefilled on every subsequent
turn until the next block boundary is crossed.

#### Raw per-turn rows (excerpts from `dev/multi_turn_waste.out`)

**Scenario 1 — micro_turn (50 tok/turn, base 1500, 20 turns)**

```
 turn | prompt_len | cached | expected | diff | partial_block_waste
    0 |       1500 |      0 |        0 |  +0  |         0
    1 |       1550 |   1056 |     1056 |  +0  |       444
    ...
   11 |       2050 |   1056 |     1056 |  +0  |       944    ← cache stuck at 1056
   12 |       2100 |   1056 |     1056 |  +0  |       994
   13 |       2150 |   2112 |     2112 |  +0  |        38    ← crosses 2×1056
   14 |       2200 |   2112 |     2112 |  +0  |        88
   ...
   19 |       2450 |   2112 |     2112 |  +0  |       338
```

Cache stuck at 1056 for 12 turns; one block-crossing event at turn 13.
Adding 50 tokens of content costs ~560 tokens of re-prefill per turn.

**Scenario 4 — long_turn (3000 tok/turn, base 2000, 20 turns, max 59K)**

```
    1 |       5000 |   1056 |  +0  |       944
    2 |       8000 |   4224 |  +0  |       776
    3 |      11000 |   7392 |  +0  |       608
    4 |      14000 |  10560 |  +0  |       440
    5 |      17000 |  13728 |  +0  |       272
    6 |      20000 |  16896 |  +0  |       104    ← lowest waste — best alignment
    7 |      23000 |  19008 |  +0  |       992    ← cycle restart
    ...
   18 |      56000 |  52800 |  +0  |       200
   19 |      59000 |  55968 |  +0  |        32    ← excellent alignment
```

Waste oscillates between 32 and 1040 with a roughly periodic pattern
(period = 1056 / gcd(3000, 1056) turns). Each turn straddles 2-3 block
boundaries.

#### Summary (20 turns each)

| scenario                  | base | final ctx | avg waste/turn | **waste / new content** |
|---------------------------|-----:|----------:|---------------:|------------------------:|
| micro_turn (50 tok/turn)  | 1500 |   2,450   |     560 tokens |   **1121%** |
| short_turn (150 tok/turn) | 2000 |   4,850   |     460 tokens |    306% |
| medium_turn (500 tok/turn)| 3000 |  12,500   |     553 tokens |    110% |
| long_turn (3000 tok/turn) | 2000 |  59,000   |     544 tokens |     18% |

**Key observations:**

1. **Per-turn waste is bounded** at ~`block_size / 2 = 528 tokens` regardless
   of how long the conversation has been running — the absolute number does
   NOT grow with turn count.
2. **Long-turn agents (3000 tok/turn, ~60K ctx) are mostly fine** for
   partial-block waste — only ~18% overhead. For these workloads the
   dominant cost is memory pressure / preempt risk, not prefix-cache
   efficiency.
3. **Short-turn agents are catastrophic**:
   - Tool-call loops, ReAct-style agents, IDE assistants → 50-500 tok/turn
   - Re-prefill **3× to 11× the new content** every turn
4. **Block-boundary crossing creates periodic reset points** — visible as
   "waste suddenly drops to <100" rows in every scenario.

#### Implication

The HiMA paper's "vLLM padding 16-30%" figure is best interpreted as a
**workload-averaged number** that bakes in this turn-size distribution.

- For agentic workloads dominated by **long-horizon agents** (single
  session, few large tool outputs), this overhead is ~7-20% — closer to
  the paper's lower bound, and not necessarily fatal.
- For workloads dominated by **swarms of short-turn agents** (parallel
  ReAct loops, tool ping-pong, IDE-style assists), the partial-block
  waste **dwarfs** the new content — this is where the paper's claim of
  "vLLM bubble" bites hardest.

This is exactly the regime where a smaller `block_size` (à la the SGLang
two-pool design HiMA advocates) would help most — the partial-block waste
is `block_size / 2` on average, so halving `block_size` halves the waste.

### D. Real Claude-Code traffic — `cc_long_traces.jsonl` (106 sessions)

Synthetic turn schedules are useful for validating the formula, but the
question that actually matters is: **on real traffic, what's the waste?**

Source data: `/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl`,
106 multi-turn agentic sessions on the hyperswitch (Rust) repository in the
Anthropic-blocks format (text / tool_use / tool_result).

We flatten messages to strings, tokenize incrementally with the Qwen3.5-35B
tokenizer (one `<|im_start|>role\n...<|im_end|>\n` chunk per message), then
apply the formula
`expected_cache_hit_at_turn_N = floor((cumulative_tokens_after_turn_{N-1} - 1) / 1056) × 1056`
which we already empirically validated to be exact (Finding C).

**Aggregate (across 106 sessions):**

| metric                       |   min |   p25 |   p50 |   p75 |   p95 |    max |
|------------------------------|------:|------:|------:|------:|------:|-------:|
| turns / session              |    15 |    22 |    46 |   127 |   246 |    396 |
| max context (tokens)         | 74K   | 101K  | 108K  | 115K  | 123K  | 132K   |
| median new tokens / turn     |   143 |   345 |   397 |   485 |  1474 |   6182 |
| avg partial-block waste/turn |   347 |   498 |   526 |   543 |   596 |    748 |
| **per-session waste %**      | **4.2%** | **10.1%** | **20.7%** | **69.5%** | **129.6%** | **219.6%** |

- **Per-turn waste is remarkably stable at ~526 tokens** (close to the
  theoretical `block_size / 2 = 528`) regardless of session length or
  turn-size distribution — matching the synthetic Finding C.
- **Per-session waste %** spans 5×: from 4% (long-horizon agent, few big
  turns) to 220% (tool-call-heavy agent, many tiny turns).
- The **workload-weighted aggregate** — what the engine actually sees on
  average across all traffic — is

  ```
  Σ partial-block waste tokens : 4,713,841
  Σ new content tokens         : 11,060,335
  → workload-weighted waste pct: 42.62%
  ```

  i.e. **for every 100 tokens of new content the agent produces, vLLM
  re-prefills another 43 tokens that should have been a cache hit**.

This is **substantially worse than the HiMA paper's "vLLM padding 16-30%"
headline figure** for this particular workload, and is concentrated in the
long tail (p75 already past 70%, p95 over 130%).

**Per-session sample (first 10 of 106):**

```
 idx | turns | max_ctx  | p50_new | avg_waste | waste_pct
   0 |    95 |  107,882 |     512 |       527 |     48.2%
   1 |   205 |  102,793 |     320 |       534 |    107.4%   ← tool-heavy
   2 |   124 |  101,671 |     436 |       551 |     67.3%
   3 |    53 |  118,227 |     408 |       569 |     25.1%
   4 |    51 |  111,667 |     580 |       448 |     20.9%
   5 |    41 |  111,366 |     384 |       524 |     19.3%
   6 |    17 |   92,008 |     818 |       556 |     10.1%
   7 |    21 |  113,397 |    1131 |       489 |      8.6%
   ...
  15 |    15 |  131,532 |    5154 |       391 |      4.2%   ← long-horizon
```

The contrast between `idx=1` (205 turns, 320 tok/turn, 107% waste) and
`idx=15` (15 turns, 5154 tok/turn, 4.2% waste) is the cleanest empirical
proof of the partial-block bubble's workload-sensitivity: **same engine,
same model, same block_size; only the turn-shape changes, and the waste
moves 25×**.

---

## Reproduction — quick start

```bash
cd /data/yuzhou/projects/vllm-songyang

# (1) offline math, instant
.venv/bin/python dev/inspect_sizes.py qwen3.5-35b

# (2) replay inflate via vLLM helpers, ~10s
.venv/bin/python dev/trace_inflate.py

# (3) loads the model — ~3 min boot, then ~30s of probing
.venv/bin/python dev/hit_rate_microbench.py

# (4) multi-turn waste (synthetic schedules) — same load cost, ~1 min of turns
.venv/bin/python dev/multi_turn_waste.py | tee dev/multi_turn_waste.out

# (5) apply formula to real Claude Code traces — ~30s, tokenizer only, no GPU
.venv/bin/python dev/real_session_waste.py | tee dev/real_session_waste.out
```

---

## Source-of-truth references (vLLM)

* `vllm/platforms/interface.py:505-674` — `_align_hybrid_block_size`
  (inflate + mamba padding)
* `vllm/v1/kv_cache_interface.py:130-180` — `AttentionSpec`,
  `FullAttentionSpec` (`page_size_bytes`)
* `vllm/v1/kv_cache_interface.py:531-560` — `MambaSpec`
* `vllm/v1/core/kv_cache_coordinator.py:471-485` — `lcm_block_size` and
  "we don't support partial block cache hit yet"
* `vllm/v1/core/kv_cache_utils.py:569-632` — `resolve_kv_cache_block_sizes`
  (hash_block_size logic)
* `vllm/model_executor/layers/mamba/mamba_utils.py:213-234` —
  `gated_delta_net_state_shape`
* `vllm/model_executor/layers/mamba/mamba_utils.py:108-116` —
  `gated_delta_net_state_dtype` (the fp32 SSM choice)
