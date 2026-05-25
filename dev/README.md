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

### `e2e_replay.py` + `plot_results.py` — end-to-end replay + figures

Replays 30 cc sessions through a live vLLM engine, recording every
request's `(prompt_len, num_cached_tokens, partial_block_waste,
new_content_tokens)` to `dev/e2e_replay.jsonl`. Issues an anchor probe
between every session so we have a time-series of "anchor cached %"
(see Finding E.1 — note the probe artifact caveat).

`plot_results.py` consumes the JSONL and writes six figures to
`dev/figures/`.

### `e2e_l1_burst.py` + `plot_l1_burst.py` — focused L1 test

Same engine config but replays sessions 1..29 with NO intermediate
anchor probes (probes only at BASELINE and FINAL). This removes the
methodology artifact in Finding E.1 and gives the clean L1
demonstration in Finding E.2.

`plot_l1_burst.py` reads BOTH `e2e_replay.jsonl` (for E.1's flat blue
line) and `e2e_l1_burst.jsonl` (for E.2's 89.2 % → 0 % drop) and
produces the headline `fig_l1_anchor_eviction.png`.

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

### E. End-to-end L1 validation: anchor eviction on real cc workload

What about the **HiMA L1 claim** — that LRU eviction on the free-block
queue actively drops *heavily-hit anchor blocks* when a cold burst of
unrelated traffic arrives? Two scripts test it end-to-end:

#### E.1 — `dev/e2e_replay.py`: probe between every session

Setup: anchor = session 0's first user message (4,737 tokens ≈ 5 KV
blocks); warm 5×; then replay sessions 1..29 with an anchor probe
issued *between* every session. vLLM config: TP=2, util=0.35,
`mamba_cache_mode=align`, `max_num_seqs=64`. KV budget at startup was
**1,079,362 tokens = ~1022 blocks**.

**Result**: anchor stays at `cached = 4224/4737` (= `floor(4737/1056) ×
1056` — every cacheable block survives) for **all 29 probes**. See
`dev/figures/fig_anchor_survival.png` (flat line at ~89%).

**Caveat — methodology artifact**: every probe HITS the anchor and
pushes its blocks back to the *tail* of the free-block LRU queue. The
measurement itself prevents the eviction it is trying to observe. The
result is a (boring) tautology: "blocks that we keep touching don't
get evicted."

Total cold-burst pressure over the 29 sessions was
**1,571,394 new tokens ≈ 1,488 new blocks** — well above the 1,017
block threshold that should evict a 5-block anchor at the tail. So
without probe refresh, eviction is *theoretically* forced.

#### E.2 — `dev/e2e_l1_burst.py`: no intermediate probes (the real test)

To remove the probe artifact: warm anchor 5×, probe once (BASELINE),
then replay sessions 1..29 with **no probes in between**, then probe
once at the end (FINAL).

**Result** (`dev/e2e_l1_burst.out`):

```
BASELINE anchor probe (post-warm): cached=4224/4737 (89.2%)
Phase B: replaying sessions 1..29 with NO probes in between
  session  1: 52 turns, cum_new_tokens=  59,999, cum_new_blocks(lb)=   37
  session  2: 37 turns, cum_new_tokens= 119,509, cum_new_blocks(lb)=   82
  ...
  session 29: 14 turns, cum_new_tokens=1,571,394, cum_new_blocks(lb)= 1229

FINAL anchor probe (after 29 sessions of cold burst):
  anchor_cached = 0/4737 (0.0%)
  cum_new_tokens during workload = 1,571,394
  cum_new_blocks (lower bound)   = 1229
  KV budget (blocks)             = 1022 (vLLM reported)

VERDICT: anchor FULLY EVICTED → L1 claim reproduced
         (LRU dropped a high-value heavily-hit block under pressure).
```

**The anchor goes from `4224/4737` cached at BASELINE to `0/4737` at
FINAL.** Even though we hit it 5 times at the start of the experiment,
vLLM's LRU-on-release-time policy doesn't remember those hits — only
the most recent release matters. The ~1.57M tokens of subsequent
cold-burst content released ~1229 new blocks to the tail of the free
queue, pushing the anchor from the tail (where the 5 warm hits left
it) all the way to the head, where it got popped.

See `dev/figures/fig_l1_anchor_eviction.png` for the side-by-side
comparison of E.1 (probe artifact masks eviction) vs E.2 (true
eviction visible).

This is exactly the failure mode HiMA L1 (LPB scoring) is designed to
prevent: under LPB, the anchor's hit count of 5 would give it a high
score, so it would NOT be evicted even when other blocks "look" more
recently used by release time.

### F. L2 visualization: per-turn bubble accumulating over the workload

The `dev/e2e_replay.py` run produces 706 session_turn rows across 29
real cc sessions. `dev/plot_results.py` consumes that JSONL and emits
six figures into `dev/figures/`:

| figure | content |
|---|---|
| `fig_cumulative_waste.png` | partial-block waste vs new content, cumulative; **final workload-weighted waste = 22.5 %** annotated on chart (lower than the 42.6 % across all 106 sessions in Finding D because this subset happens to have longer mean turns) |
| `fig_per_turn_breakdown.png` | every request's prompt as a stacked bar of {cache hit, partial-block re-prefill, new content prefill}; reveals the sawtooth structure: prompts grow turn-by-turn within a session then reset to ~0 at the session boundary |
| `fig_per_session_waste.png` | per-session waste % sorted ascending; ranges from **4 % (sessions with 7-8 turns)** to **67 % (sessions with 67-78 turns)**; mean = 21.7 % |
| `fig_hit_rate.png` | per-request cache-hit fraction, chronological, colored by session — shows the "session start = cold" pattern and the rapid climb to 95%+ within 2-3 turns |
| `fig_anchor_survival.png` | anchor cached % across probes (flat at ~89 %; see E.1 caveat) |
| `fig_l1_anchor_eviction.png` | **The L1 headline figure**: blue line (E.1, with probes) flat at 89.2 %; red line (E.2, no probes) drops from 89.2 % to **0 %** after 29 sessions of cold burst |
| `fig_dashboard.png` | 2×2 summary grid of the four most informative panels |

The headline plot is `fig_cumulative_waste.png` — the cumulative
partial-block waste accumulates **linearly** in the workload's new
content, reaching 22.5 % of new content over 29 real sessions
(353,790 wasted tokens / 1,571,394 new content tokens).

`fig_per_session_waste.png` clearly shows that **session turn-count is
the dominant driver of waste %** — short-turn sessions hit the bubble
hardest. Matches the synthetic Finding C and the real-traffic Finding D.

### G. Counterfactual: how much of the bubble is the inflate's fault?

Findings A-F have shown:
  - `block_size` is inflated from the vLLM default of 16 to **1056** on
    this model (Finding A);
  - real cc workload loses **42.62 %** of new content to partial-block
    waste at `block_size=1056` (Finding D).

How much of that 42.62 % is *the inflate's fault* vs the workload's
fault? `dev/counterfactual_block_size.py` answers it directly: apply
the *exact same* formula and the *exact same* 106 cc sessions, varying
only `block_size`:

| block_size | workload-weighted waste % | total waste tokens |
|---:|---:|---:|
| 16   |  **0.69 %** |     77,275 |
| 32   |  1.34 %     |    149,016 |
| 64   |  2.61 %     |    288,977 |
| 128  |  5.26 %     |    582,289 |
| 256  | 10.52 %     |  1,163,921 |
| 512  | 20.95 %     |  2,316,945 |
| **1056** | **42.62 %** | **4,713,841** |
| 2112 | 85.10 %     |  9,411,985 |

**The waste % roughly doubles every time `block_size` doubles** (a clean
log-linear relationship — see `dev/figures/fig_block_size_counterfactual.png`).

At the vLLM default `block_size=16`, the same workload would lose only
**0.69 %** of new content — essentially zero. So the entire ~42 % bubble
is the inflate's fault, not the workload's. If vLLM weren't forced to
inflate `block_size` to match the fp32 mamba state (Finding A), the cc
bubble would be **62 × smaller** in token terms (77K vs 4.7M wasted
tokens across the 106-session corpus).

This is exactly the counterfactual that HiMA's Path-A (split BlockPool
into separate per-spec stores) would deliver.

### H. L1 anchor-eviction *pressure curve* — the cliff is at K ≈ 7 sessions

Finding E.2 showed the anchor evicts after 29 sessions of cold burst.
**At what pressure does it actually break?** `dev/e2e_l1_pressure_curve.py`
sweeps K ∈ {0, 5, 10, 15, 20, 25, 30}, drawing each phase's cold-burst
sessions from a disjoint pool (sessions 1..5 for K=5, 6..15 for K=10,
etc.). Each phase: re-warm anchor 5×, replay K silent sessions, probe.

**Result** (`dev/e2e_l1_pressure_curve.out` /
`dev/figures/fig_l1_pressure_curve.png`):

| K | cum new tokens | cum new blocks (lb) | anchor cached % |
|---:|---:|---:|---:|
|  0 |          0 |     0 | **89.2 %** (= floor(4737/1056)·1056) |
|  5 |    279,895 |   222 | **89.2 %** (still alive) |
| **10** |   **540,235** | **429** | **0 %** ← *cliff* |
| 15 |    810,834 |   613 |  0 % |
| 20 |  1,117,292 |   831 |  0 % |
| 25 |  1,304,043 |   989 |  0 % |
| 30 |  1,573,910 | 1,223 |  0 % |

The transition is **sharp**: anchor survives 5 sessions of cold burst
(222 new blocks lower-bound), then is **completely gone after 10**
(429 new blocks lower-bound). That's about a 250-block shove on a
~1022-block KV budget. The actual cache state at K=10 includes prior
phases' content too, so the real cumulative cache pressure at the
eviction cliff is somewhere around 500-800 blocks — well below the
1022-block budget. **vLLM's LRU evicts the anchor before the cache is
even full, because the anchor is the oldest "tail" entry by release
time.**

This is the precise L1 failure mode HiMA-LPB is designed to fix: under
LPB scoring, the anchor's hit count (5 warm hits) would give it a
higher score than the cold-burst blocks (each hit 0-1 times) — so even
at high cumulative pressure, the anchor would survive while less-valued
blocks evict first.

### I. Wall-clock TTFT cost of the partial-block bubble

The 4.71 M wasted tokens from Finding D are a token-count number. What's
the user-visible cost in seconds? `dev/ttft_cost_of_bubble.py` measures
actual prefill latency at lengths {512, 1k, 2k, 4k, 8k, 16k} on
Qwen3.5-35B-A3B / TP=2 / H200, takes the median of 3 trials per length,
and fits a linear model.

**Measured prefill latency** (`dev/ttft_cost_of_bubble.out`):

| prompt_len (tokens) | median wall (ms) | apparent tps |
|---:|---:|---:|
|    512 |  52 |    9,861 |
|  1,024 |  53 |   19,217 |
|  2,048 | 108 |   18,965 |
|  4,096 | 114 |   35,823 |
|  8,192 | 146 |   56,218 |
| 16,384 | 276 |   59,463 |

**Linear fit:** `wall_s = 53.1 ms + L × 13.34 µs/token`
  → marginal prefill rate ≈ **75 K tokens/sec** (after the fixed
  ~53 ms scheduler/launch overhead).

**Applying the marginal slope to the bubble:**

  - **29-session e2e_replay subset** (Finding F, 22.5 % waste):
    - 353,790 wasted tokens × 13.34 µs = **4.7 seconds** total TTFT cost
    - = **6.7 ms per request** averaged over 706 requests

  - **Full 106-session cc corpus** (Finding D, 42.6 % waste):
    - 4,713,841 wasted tokens × 13.34 µs = **62.9 seconds** total
      TTFT cost = **1.0 minute** of pure GPU prefill wall-clock
    - = **0.6 second per session** averaged across all 106 sessions
    - Distributed across ~10K requests in the corpus, that's
      **~6 ms TTFT overhead per request** — silent but cumulative.

In counterfactual terms (Finding G), at the vLLM default
`block_size=16` the same 106-session corpus would waste only 77K
tokens × 13.34 µs ≈ **1 second total** instead of 63 seconds.
**The inflate costs ~62 seconds of user-visible prefill latency per
106-session cc workload.**

> *(Note: at the longest probed length (16K tokens), L=32K crashed
> the script due to filler shortage — see `dev/ttft_cost_of_bubble.out`
> for the traceback. The 6 data points up to 16K are sufficient for the
> linear fit; the same slope extrapolates cleanly to 32K territory
> as a per-token marginal cost.)*

### J. HiMA L1 wiring through to a runnable engine knob (LRU vs LPB)

Findings A-I established the *failure modes* of vLLM's default LRU
free-block queue (anchor eviction under cold burst, partial-block
bubble, TTFT loss). HiMA's L1 fix — LPB scoring — was already coded in
`vllm/v1/core/hima/lpb_free_queue.py` but it took three plumbing fixes
before it actually flipped behavior on the running engine:

  1. **`EngineArgs.hima_enabled`** added (`vllm/engine/arg_utils.py`)
     and threaded into `CacheConfig`. Without this, `LLM(...,
     hima_enabled=True)` was silently ignored and the engine never
     called `enable_runtime()`, so `LPBFreeBlockQueue` was never
     constructed.
  2. **`HiMACoordinator.__init__` auto-invoke**
     (`vllm/v1/core/hima/coordinator_hima.py`): `HiMACoordinator.__new__`
     returned an instance of `HiMACoordinatorImpl` (a dynamically-built
     subclass of `HybridKVCacheCoordinator`). Because that instance is
     NOT an instance of `HiMACoordinator`, Python skipped `__init__`,
     so `block_pool` was never set and `KVCacheManager.__init__` crashed
     with `AttributeError: 'HiMACoordinatorImpl' object has no attribute
     'block_pool'`. Fix: call `instance.__init__(*args, **kwargs)`
     explicitly inside `__new__`.
  3. **`_hima_find_longest_cache_hit` signature**: matched the
     `(self, request)` shape, but the post-pull
     `HybridKVCacheCoordinator.find_longest_cache_hit` takes
     `(self, block_hashes, max_cache_hit_length)`. Fix: thread both
     positional args through.
  4. **`CostCurves.cost(pool, depth)` API mismatch**: the score function
     called `rt.cost_curves.cost(pool_kind, depth)`, but `CostCurves`
     exposes `c_kv_ms(L)` / `c_m_ms(L)` per pool, not a unified `cost`.
     Fix: pick the right method by `pool_kind`.
  5. **Score-scale inversion** (`vllm/v1/core/hima/lpb_free_queue.py`):
     the *real* bug. Cold blocks fell back to `time.monotonic()` ≈ 1e9
     while hit blocks got `n_b × c_kv_ms(depth)` ≈ 10² — so under
     min-heap-pop semantics, **the anchor was always evicted before any
     cold block**. Fix: add a constant `_HIT_SCORE_OFFSET = 1e12` to hit
     scores so cold < hit unconditionally, restoring the paper's
     intended ordering.

After those five fixes, `hima_enabled=True` actually flips behavior.

### K. End-to-end LRU vs LPB on cc workload — see `dev/intralayer/`

The big LRU vs LPB study (workload design, multi-sweep runs on
Qwen3.5-35B-A3B / Qwen3.5-122B-A10B, Phase H production-pattern
−12 % to −18 % batch TTFT win, sglang cross-implementation review)
moved out of this README to keep the engine-vs-engine notes
together:

- **[`dev/intralayer/scenarios.md`](aginfer/scenarios.md)** — shared
  benchmark design (A → B → G → E → F → H → C), pitfalls, expected
  outcomes. Engine-agnostic.
- **[`dev/intralayer/vllm.md`](aginfer/vllm.md)** — vLLM HiMA L1
  implementation pointers, driver (`dev/compare_lru_lpb.py`),
  measured results table (2 sweeps × n=3, headline Phase H
  numbers, anchor protection binary across 6/6 trials each).
- **[`dev/intralayer/sglang.md`](aginfer/sglang.md)** — sibling
  sglang implementation review (`rucnyz/sglang@HiMA`), design
  comparison, correctness notes, Phase H driver still open.

TL;DR for vLLM specifically: at production util=0.9 in the
concurrent swarm pattern that fires after Phase F's pressure has
evicted the anchor under LRU, LPB delivers
**−12.0 % batch TTFT on Qwen3.5-35B-A3B** (~6σ) and
**−17.7 % on Qwen3.5-122B-A10B** (~5σ, saves 100 ms per
30-request swarm). On every other phase × metric LPB is tied
with LRU within trial noise.

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

# (6) e2e replay on real cc traces, anchor probe between sessions
#     (~5 min: 3 min model load, ~2 min for 706 reqs across 29 sessions)
.venv/bin/python dev/e2e_replay.py | tee dev/e2e_replay.out
.venv/bin/python dev/plot_results.py    # writes dev/figures/*.png

# (7) e2e L1 burst test (no intermediate probes — the real L1 demonstration)
.venv/bin/python dev/e2e_l1_burst.py | tee dev/e2e_l1_burst.out
.venv/bin/python dev/plot_l1_burst.py   # writes dev/figures/fig_l1_anchor_eviction.png

# (8) Counterfactual block_size sweep on cc data (no GPU, ~30s)
.venv/bin/python dev/counterfactual_block_size.py | tee dev/counterfactual_block_size.out

# (9) L1 pressure curve (anchor survival vs cold-burst K), ~25 min
.venv/bin/python dev/e2e_l1_pressure_curve.py | tee dev/e2e_l1_pressure_curve.out
.venv/bin/python dev/plot_l1_pressure.py

# (10) Wall-clock TTFT cost of bubble (~5 min)
.venv/bin/python dev/ttft_cost_of_bubble.py | tee dev/ttft_cost_of_bubble.out
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
