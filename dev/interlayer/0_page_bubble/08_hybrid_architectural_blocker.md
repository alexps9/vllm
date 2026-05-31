# Finding M.8 — hybrid bubble elimination is an *architectural* problem, not a plumbing one

Previous documents (M.2, SESSION_HANDOFF) sketched hybrid support as
"~400 LOC of per-group `num_computed_tokens` plumbing through the
scheduler/manager/runner stack". After implementing the non-hybrid
case end-to-end (M.4–M.7) and re-examining the data flow, that
estimate was **wrong** — the issue isn't plumbing, it's a fundamental
architectural property of how prefill works in hybrid models.

## Why attention's partial-cache works on non-hybrid

For a single attention layer processing prefill at positions
`[K*block_size+R : N]` after a cache hit at `K*block_size+R`:

- **Input needed**: hidden states `H_p` for `p ∈ [K*block_size+R, N)`
- **State needed**: K, V for `p ∈ [0, K*block_size+R)` (✓ in KV cache)
- **Output produced**: hidden states `H'_p` for `p ∈ [K*block_size+R, N)`

Attention can skip the R cached positions because **subsequent
layers don't need attention's past hidden states** — they only need
the K, V values, which are cached. Attention's hidden state output
for past positions is consumed at THIS layer and never referenced
again.

For full-attention-only models (chain of attention + FFN + RMSNorm
layers), this holds at every layer: FFN at position `p` only depends
on its own input at position `p`, which only needs attention's
output at position `p`. So skipping R positions ripples through the
whole model — nothing past `K*block_size+R` is computed.

## Why it breaks on hybrid

Insert a mamba (or GDN/SSM) layer between two attention layers.
Mamba at position `p` needs:

- **Input**: hidden state `H_p` from the previous layer
- **State**: SSM state at `p-1` (the recurrent state)
- **Output**: hidden state `H'_p`

For mamba to skip position `p`, it needs its SSM state at `p`
already cached. vLLM caches SSM state only at **full-block
boundaries** — `state[block_id]` is the state at position
`(block_id + 1) * block_size - 1`. There is no SSM state at
sub-block positions.

So mamba **cannot** skip the R cached positions just from the
existing state cache. It has to *re-compute* the SSM state by
prefilling those R tokens.

But to prefill R tokens, mamba needs its **input** at those R
positions — the hidden states from the layer before it. Which means
the layer before it (typically attention) has to **produce** those
hidden states. Which means attention has to compute Q at those R
positions. Which is exactly the work the partial-cache was supposed
to skip.

**Circular dependency**: every layer's "skip R" requires the
previous layer to "not skip R".

## The dependency graph in code terms

For a forward pass with `num_computed_tokens = K*block_size + R`,
the model runner generates input tokens `tokens[K*block_size+R : N]`
and runs each layer on those. The first layer (embedding) outputs
hidden states for those positions only. Each subsequent layer takes
those hidden states as input and produces hidden states for the
same position range.

For a mamba layer in the middle, it needs **both** the hidden state
input for those positions AND the SSM state at position
`K*block_size+R-1`. Without the latter, mamba's output for position
`K*block_size+R` is wrong (it would use stale state from position
`K*block_size-1` and skip R tokens of recurrence).

To fix this WITHOUT model changes, the forward pass must START at
`K*block_size` (mamba's safe state), not at `K*block_size+R`.
Meaning attention at the first layer must compute Q at positions
`[K*block_size : K*block_size+R-1]` too. Which defeats the cache.

## What WOULD make hybrid work

Two possible directions, both touching model code or kernels:

### Option A: Cache mamba SSM state at partial boundaries

Add a parallel "partial mamba state cache" indexed the same way as
the partial KV cache:

```
cached_partial_mamba_state: dict[partial_key, SsmStateBundle]
```

Storage cost: ~1 MB per layer per partial entry on Qwen3.5-35B-A3B
(48 GDN layers × ~1 MB state ≈ 48 MB per partial; 100 partials =
4.8 GB). Tolerable on H200 but not free.

Required mamba kernel changes:
- On prefill, optionally output the intermediate SSM state at
  position `K*block_size + R - 1` to a designated slot (not just
  the per-block slot)
- On future prefill, optionally initialize from a designated slot
  (not just the per-block slot)

These changes are inside the GDN/mamba CUDA kernels (in
`vllm/model_executor/layers/mamba/`). Not a small surface — and
each mamba variant (Mamba1, Mamba2, GDN, KDA, etc.) would need
similar surgery.

### Option B: Cache full inter-layer hidden states

For each layer, cache the OUTPUT hidden state at the partial
boundary. The forward pass for a future request would START from
position `K*block_size + R` and INITIALIZE each layer's "previous
hidden state" from the cached value.

Storage cost: HUGE. For Qwen3.5-35B-A3B with 4096-dim hidden:
- 64 layers × 4096 dims × bf16 = 0.5 MB per position per partial
- For R values, only the LAST partial position is cached, so 0.5 MB per partial entry
- 100 partials = 50 MB. Actually cheap.

But "initialize each layer's previous hidden state" isn't how
mamba's recurrence works — mamba carries SSM state, not hidden state.
So Option B is essentially Option A under the hood for mamba layers.

### Option C: Sub-block KV layout for both attention AND mamba

Refactor the cache layout so that mamba's state storage uses the
SAME granularity as the partial cache (e.g., every 16 tokens). This
ALWAYS stores intermediate state, eliminating the need for a
separate partial cache. Tradeoff: a lot more memory per request
(mamba state is large — 1 MB at full-block is already 66x the
attention page size).

Far more invasive than A or B; probably never worth it.

## What is NOT the answer

The earlier sketch in M.2 / SESSION_HANDOFF described per-group
`num_computed_tokens` as a ~400 LOC scheduler/manager/runner change.
That change alone (without ALSO solving the mamba state problem)
delivers **zero perf win** on hybrid models — because mamba
would either silently corrupt its state (if `num_computed_tokens`
diverges between groups without state caching) or fall back to the
single-group min (defeating the partial cache).

The 400 LOC was the visible part of the iceberg; the rest is mamba
kernel work.

## Recommendation

For real hybrid bubble elimination, **Option A** (sub-block mamba
state cache) is the right path. Scope:

1. Identify the mamba kernel entry points (Triton kernels in
   `vllm/model_executor/layers/mamba/mamba2.py`,
   `qwen3_5/qwen3_5_gated_delta_net.py`, etc.).
2. Add an optional "output state to designated slot" parameter to
   the prefill kernel.
3. Add an optional "init state from designated slot" parameter.
4. Wire a per-(group, partial_key) state slot table in the model
   runner; populate on partial-cache insert, consume on partial-
   cache hit.
5. Combine with the existing partial-attention cache (M.4/M.5).

Cost estimate: 1-2 weeks for an experienced vLLM developer, mostly
in the mamba kernel + model runner glue. Per mamba variant.

For the cc workload (Qwen3.5-35B-A3B with Gated DeltaNet), the
specific kernel is the GDN forward at
`vllm/model_executor/layers/mamba/mamba2_metadata.py` and friends.

## What this means for the immediate roadmap

- **Non-hybrid bubble fix**: shipped and validated end-to-end (M.6/M.7).
  43% TTFT win on follow-up turns at R=800, block_size=1024. Real.
- **Hybrid bubble fix**: blocked on Option A above, not on per-group
  `num_computed_tokens` plumbing. The per-group plumbing alone would
  not help.
- **CC workload measurement**: doable today on a non-hybrid model
  (Qwen3-8B) to characterize how the partial-cache mechanism behaves
  on realistic multi-turn traffic. The cc traces themselves are
  model-agnostic. Finding M.9 (this is the natural next step).
- **Stress test for cross-request correctness**: the destructive-hit
  fix from this commit cycle handles the obvious race; multi-tenant
  stress test is M.10.

In short: hybrid is "next quarter's project" requiring mamba kernel
work; non-hybrid is shipping today with measured wins; the right
follow-on this session is M.9 (cc workload on Qwen3-8B).
