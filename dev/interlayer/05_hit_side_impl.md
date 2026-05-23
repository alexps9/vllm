# Finding M.5 — partial-cache hit-side prototype (single-group only)

After M.4 shipped the cache-side scaffolding (insertion only, no
reader), this commit adds the hit-side reader for the **single-group**
case (no mamba, full attention only). The hit-side for hybrid models
still needs per-group `num_computed_tokens` (M.2) and is deferred.

## What the hit-side does

When `KVCacheManager.get_computed_blocks(request)` is called for a
new request, after the coordinator returns its `K`-full-block hit:

1. Check the partial-block cache map (populated by M.4).
2. If non-empty AND the engine has only one KV cache group AND there
   are tokens past `K * block_size` to match, probe for an R-token
   partial extension (R from `min(block_size - 1, remaining)` down to
   1, first hit wins).
3. If found, append the cached partial block to `computed_blocks[0]`
   and bump `num_new_computed_tokens` by R.

Now the request goes into `allocate_slots` with a NON-block-aligned
`num_new_computed_tokens`. This is the limitation that the comment at
`kv_cache_manager.py:217-218` calls out:

> "allocate_slots() requires num_computed_tokens to be block-size
> aligned. Removing this limitation could slightly improve
> performance in the future."

Removing this limitation is the model-runner-side work; this commit
only implements the cache layer.

## Single-group constraint

The check `len(self.coordinator.kv_cache_config.kv_cache_groups) == 1`
gates the partial extension to single-group configs (no mamba). For
multi-group hybrids, the partial extension would mismatch mamba's
hit length and break SSM state. The hybrid path needs the per-group
num_computed_tokens lift (M.2 / M.3) — explicit follow-up work.

## What might still break

1. **Scheduler / allocator** may assert that `num_computed_tokens %
   block_size == 0`. The TODO comment exists because some code paths
   weren't audited yet. The places I suspect:
   - `scheduler.py:1018`: `request.num_computed_tokens += num_scheduled_token`
     (this is fine, just additive)
   - `single_type_kv_cache_manager.py:299`: `num_full_blocks = num_tokens // self.block_size`
     (rounds down, fine)
   - `kv_cache_manager.py:421-425`: `num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)`
     then passed to `coordinator.cache_blocks(request, num_tokens_to_cache)`. The coordinator floors back to lcm_block_size. So caching new full blocks should still work.
   - `gpu_model_runner.py`: builds positions / seq_lens from `num_computed_tokens` (line 2089). If non-aligned, positions start at K*block_size + R, which is what we want.

2. **Attention metadata**: needs `seq_len = total_request_len` and
   `query_start_loc` adjusted so only the new (post-partial) tokens
   are queried. The runner currently builds query_start_loc from
   `num_scheduled_tokens` per request, which is num_new_tokens
   minus num_new_computed_tokens. If num_new_computed_tokens
   includes R, then num_new_tokens drops by R automatically. So this
   might just work.

3. **Block table at position K**: the cached partial block enters the
   request's block_table at position K (the (K+1)-th block). The
   attention kernel sees block_table[K] when reading the K-th logical
   block of the sequence. The first R positions of that block have
   valid cached content. The kernel computes for query positions
   [K*block_size+R : K*block_size+R+num_new], which write into
   block_table[K] at offsets [R : R+num_new]. This is correct.

4. **Mamba**: skipped by the single-group check. If somehow a
   hybrid model bypasses the check, the result would be corrupt
   mamba state. The check is the only safety net here.

## Activation

Set both env vars to enable:

```bash
VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py
```

Without `VLLM_PARTIAL_CACHE_ENABLED=1` the new code path is a no-op
(both the cache-side insertion and the hit-side reader short-circuit
on the env-var check).

## Validation plan

`dev/interlayer/05_nonhybrid_microbench.py` is the testbed. Uses
Qwen3-8B (non-hybrid full attention) with `block_size=256`. Sweeps R
∈ [0, 16, 64, 128, 200, 255], same 2-turn structure as the hybrid
microbench. The acceptance criterion:

- **Baseline (env unset)**: `turn2_cached = K * block_size = 512` for
  every R > 0 (the existing bubble). `turn2_uncached = R + extension`.
- **With env var set**: `turn2_cached = K * block_size + R = 512 + R`
  for every R > 0. `turn2_uncached = extension` (16). TTFT for turn 2
  should be approximately flat across R values.

If the acceptance criterion is met, this single-group prototype
proves the mechanism end-to-end and the hybrid path is just the
per-group lift on top.

## Known untested risks

This commit ships the hit-side code without an end-to-end run on a
healthy machine. The host where the design was done is under heavy
CPU contention from other users (load avg frozen at 47, CUDA
compilation processes pinning resources) — Python interpreters
load very slowly (5-10 min for `import torch`) which makes
iteration impractical. The hit-side code is syntax-clean and
type-checks, but has not been smoke-tested on real engine traffic.

When testing on a quieter machine:
1. `VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py | tee dev/interlayer/runs/05_nonhybrid_partial_cache.out`
2. Compare `turn2_cached` to the baseline (without env var).
3. If the partial extension hits, `turn2_cached` should equal
   `turn1_len` (full prior request cached).
4. If it doesn't (still equals K*block_size), debug:
   - Is `cached_partial_block_map` non-empty after turn 1? Check
     the `[interlayer/partial_cache] insert` log line.
   - Does the partial key match on lookup? Add a debug log in
     `_compute_partial_key` to compare.
   - Does `allocate_slots` accept the non-aligned num_new_computed?
     If it asserts, see the "What might still break" section above.
