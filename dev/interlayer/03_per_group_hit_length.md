# Finding M.2 — partial-cache requires per-group cache-hit length on hybrid models

After confirming the bubble empirically (Finding M.1), I dug into the
implementation path and hit a hard architectural constraint that makes
the "attention-side partial cache" idea less straightforward than the
01_design_space.md sketch implied.

## The actual constraint

`coordinator.find_longest_cache_hit()` at
[`vllm/v1/core/kv_cache_coordinator.py:503-507`] returns ONE
`num_new_computed_tokens` for the whole request, applied uniformly
across all KV-cache groups:

```python
def find_longest_cache_hit(...) -> tuple[tuple[list[KVCacheBlock], ...], int]:
```

The tuple-of-blocks is per group, but the int is global. Downstream,
`kv_cache_manager.allocate_slots()` propagates this single int to
`request.num_computed_tokens`. The model runner then builds attention
metadata with this single value for every group's seq_lens/query_start_loc.

This single-int assumption is **load-bearing** for the partial-cache
fix on hybrid models:

- **Attention** (block_size=1056 inflated) could in principle hit at
  `K * 1056 + R` with a partial block. To skip re-prefilling the R
  tokens, num_computed_tokens for the attention group would need to
  be `K * 1056 + R`.
- **Mamba** (block_size=1056) can only resume from the last full-block
  state (at `K * 1056`). For its group, num_computed_tokens must be
  `K * 1056`.

If we set num_computed_tokens to `K * 1056 + R` globally, mamba's
state is *wrong* (no SSM state cached at position `K * 1056 + R`).
If we set it to `K * 1056` globally, attention has to re-prefill the
R tokens regardless of whether they're cached — defeating the
partial-cache value.

The "kernel-side skip" workaround (attention layer notices the first
R offsets of block_table[K] already have valid KV and skips them)
would require touching the FlashAttention/FlashInfer kernels, which
is way outside the scope of this work.

## The right answer: per-group num_computed_tokens

The fix is to **make `num_computed_tokens` per-group throughout the stack**.
This is invasive but actually plausible because:

1. `gpu_model_runner._build_attn_group_metadata()` at
   [`vllm/v1/worker/gpu_model_runner.py:2328`] already builds attention
   metadata per group. The only missing piece is a per-group
   `num_computed_tokens_cpu` to feed into each group's
   `CommonAttentionMetadata`.

2. The single-int assumption only spans the scheduler/manager surface
   (cache hit length lookup → request state → metadata builder input).
   Replace one int with a `list[int]` keyed by group id, propagate.

3. Attention metadata already carries `seq_lens` per request; making
   it per (group, request) is a `list[Tensor]` instead of `Tensor`
   change in the model-runner metadata-build path.

Estimated surface area:

| File | Change | LOC |
|---|---|---|
| `kv_cache_utils.py` | `PartialBlockHash`, partial-hash helper | ~50 |
| `block_pool.py` | `cached_partial_block_map`, `cache_partial_block()`, evict path | ~100 |
| `single_type_kv_cache_manager.py` | `cache_blocks` caches partial; `find_longest_cache_hit` (FullAttn) extends with partial; (Mamba) does not | ~80 |
| `kv_cache_coordinator.py` | `find_longest_cache_hit` returns `list[int]` per group instead of single int | ~30 |
| `kv_cache_manager.py` | Propagate per-group hit lengths to allocate_slots; relax block-aligned `num_computed_tokens` requirement (the limitation called out at L217-218) | ~50 |
| `request.py` | `num_computed_tokens: dict[int, int]` per group | ~20 |
| `scheduler.py` | Adapt running-request tracking | ~30 |
| `gpu_model_runner.py` | Per-group metadata: `_build_attn_group_metadata` takes per-group num_computed_tokens | ~30 |
| `core_async_scheduler.py` etc. | Similar adaptations | ~30 |

Total: ~400 LOC across 8-10 files. Plus a microbench (~80 LOC) and a
flag wiring (~10 LOC).

## What this commit doesn't yet do

This commit documents the finding but **does not yet prototype the
per-group change**. That's the next step. Spike plan:

1. **Smoke test (no real change)**: replace the inflated `block_size`
   with `min(group_block_sizes)` in the FullAttentionManager's
   per-group view, see if anything obvious breaks. Confirm that
   `num_computed_tokens` is the single bottleneck, not something
   deeper.
2. **Per-group num_computed_tokens (cold path only, no partial cache yet)**:
   change the type from `int` to `list[int]` keyed by group, but
   always populate with the same value. Should be a no-op. Run the
   existing dev/compare_lru_lpb.py to confirm.
3. **Add partial cache** behind a flag, populate the attention group's
   hit length with `K * block_size + R` and leave mamba at `K * block_size`.
4. **Re-run 02_partial_cache_micro.py**: expect `turn2_uncached ≈ 16`
   for every R (attention re-prefill skipped; mamba re-prefills R+16
   but that's much cheaper).
5. End-to-end on the cc workload.

## Alternative: defer hybrid, prototype on pure-attention models first

Pure-attention models (no mamba/SSM) have `block_size = 16` (no inflation).
The partial-block bubble is at most 15 tokens, so the absolute win is
small. But the implementation is much simpler — no per-group surgery
needed (only 1 group). This could be a useful smoke test for the
partial-cache mechanism itself, decoupled from the per-group
infrastructure.

The risk is that on pure-attention models the change won't show
measurable TTFT improvement (the bubble is too small relative to
overall prefill cost), making it hard to claim victory.

## Recommendation for the next session

Start with **smoke test** (option 1 above) to confirm the per-group
infrastructure assumption. Then either:
- Go big: implement per-group `num_computed_tokens` + partial cache (one
  comprehensive PR).
- Go small: prototype partial cache on pure-attention models and
  document the hybrid path as "needs per-group num_computed_tokens".

I'd recommend the small path first — get a working partial-cache
mechanism on pure-attention models, prove it, then propose the
per-group lift as a follow-on. That keeps each step reviewable.
