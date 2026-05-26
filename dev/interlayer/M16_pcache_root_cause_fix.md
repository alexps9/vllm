# M.16 — partial-cache adoption breaks subsequent full-block caching (root cause + fix, 2026-05-26)

## Symptom

verify/2 isolation experiment on Qwen3-8B (single-group full attention,
util=0.55, 32-turn growing conversation):

| config | clients | hit% (last turn) | p95 TTFT |
|---|---|---:|---:|
| LRU | 1 | 96.8 % | 98 ms |
| **pcache** | 1 | **93.6 %** (−3.2 pp) | **134 ms** (+37 %) |
| LRU | 16 | 96.8 % | 906 ms |
| **pcache** | 16 | **93.6 %** (−3.2 pp) | **1326 ms** (+46 %) |

Regression appears only at the very last turn (t30/t31, prompt ≈ 32 K
tokens); turns 0-28 are bit-identical with LRU.

## Triangulation

Two diagnostic env vars on the same workload exonerated all paths
except the actual extension-application:

| variant | hit% | meaning |
|---|---:|---|
| `pcache` (default) | 93.6 % | regression reproduces |
| `pcache_probe_only` (`VLLM_PARTIAL_CACHE_PROBE_ONLY=1`) | 96.8 % | insert + lookup run, extension not applied — clean |
| `pcache_insert_off` (`VLLM_PARTIAL_CACHE_INSERT_OFF=1`) | 96.8 % | insert disabled → lookup misses → clean |
| `pcache_force_miss` (`VLLM_PARTIAL_CACHE_FORCE_MISS=1`) | 96.8 % | lookup forced None → clean |

So the regression triggers **only when the partial extension actually
gets applied** to the request's computed blocks.

## Root cause

`vllm/v1/core/single_type_kv_cache_manager.py:235` (before fix):

```python
req_blocks.extend(new_computed_blocks)
self.num_cached_block[request_id] = len(req_blocks)
```

`num_cached_block` is used by the subsequent `cache_blocks()` call to
decide which blocks still need their full-block hash computed and
inserted into `cached_block_hash_to_block`:

```python
if num_cached_blocks >= num_full_blocks:
    return  # nothing new to cache
```

`len(req_blocks)` was a valid proxy for "blocks already cached as
full" while every entry in `new_computed_blocks` was either a null
skip-slot or a block that had already been hashed. The interlayer
partial-cache (M.4/M.5) broke that invariant:
`_try_partial_extension` appends a block that is **partially** cached
(has R valid tokens, but its full-block hash has never been computed).
The proxy then counts that partial block as "already cached", so
when the request later fills it to a full 1056 tokens, `cache_blocks`
returns early and the now-full block **never enters
``cached_block_hash_to_block``**. Future turns can't find it as a
full-block hit.

## Why the regression surfaces at the LAST turn only

With `hash_block_size = 16 < block_size = 1056`, vLLM's prefix-cache
finds the longest contiguous prefix at 16-token granularity. As long
as the partial block from turn N matches turn N+1's content for at
least R tokens, the prefix-cache reaches into the partial block via
hash-granular matching. Even if the block isn't full-cached, hit %
appears normal because the *content* prefix matches at hash
granularity.

The regression surfaces when:
1. The adopted partial block was extended to a FULL block at turn N
2. The next turn N+1 has a prompt that re-uses ALL of turn N's
   content as a prefix (which is what our growing-context workload
   produces every turn)
3. AND turn N+1's prompt extends past turn N's full-block boundary
   (i.e. requires the adopted block to be a *full-cache hit* to keep
   the hit% up)

At intermediate turns, the hash-granular matching covers most of the
deficit. At the *last* turn (when context length lines up such that
the adopted block is squarely on a block boundary), the deficit is
exactly one block worth of tokens (~1056), which manifests as a
−3 pp hit% step.

## Fix

`vllm/v1/core/single_type_kv_cache_manager.py:235` (after fix):

```python
req_blocks.extend(new_computed_blocks)
self.num_cached_block[request_id] = sum(
    1 for b in req_blocks
    if b.is_null or b.block_hash is not None
)
```

Derives the counter from the underlying state predicate ("this block
is either a null skip-slot or already has a `block_hash`") instead of
using `len(req_blocks)` as a structural proxy. The adopted partial
block has `block_hash = None` (it was cached as partial, never
full-hashed) and is correctly excluded; the eventual `cache_blocks()`
then catches it and inserts the now-full hash.

Patch saved at
[`dev/interlayer/pcache_fix_num_cached_block.patch`](pcache_fix_num_cached_block.patch).

## Validation

Same workload, post-fix, GPU 5,6, util=0.55, win=3600:

| config | clients | pre-fix hit% | post-fix hit% | pre-fix TTFT | post-fix TTFT |
|---|---|---:|---:|---:|---:|
| LRU | 1 | 96.8 % | 96.8 % | 98 ms | 107 ms |
| **pcache** | 1 | 93.6 % (−3.2 pp) | **96.8 %** (tied LRU) | 134 ms (+37 %) | **102 ms** (−5 % vs LRU) |
| LRU | 16 | 96.8 % | 96.8 % | 906 ms | 881 ms |
| **pcache** | 16 | 93.6 % (−3.2 pp) | **96.8 %** (tied LRU) | 1326 ms (+46 %) | **834 ms** (−5 % vs LRU) |

pcache is no longer a regression; it is a marginal TTFT win on this
workload (−5 % vs LRU). Trajectory matches LRU at every turn instead
of diverging at the last few turns.

## Implications

1. **M.7/M.9 results are still valid** — those workloads (single
   request, short prompts) never hit the path where the adopted
   partial block becomes a full block on the same turn. The bug only
   surfaces on long, growing-context workloads where the adoption
   transition to full occurs frequently.
2. **Long-context multi-turn agents** (the production workload that
   originally motivated partial-cache) is now correctly handled. The
   prior pcache behavior would have silently increased TTFT on these
   workloads — likely missed in M.7/M.9 because those benches don't
   exercise the failure regime.
3. **Songyang's W1 collapse** is still not reproduced on Qwen3-8B
   (the workload doesn't pressure the KV pool enough at util=0.55).
   Next step is to re-attempt on hybrid Qwen3.5-35B-A3B where W1 was
   originally observed.

## Files

- Code change: `vllm/v1/core/single_type_kv_cache_manager.py:228-249` (this commit)
- Patch snapshot: `dev/interlayer/pcache_fix_num_cached_block.patch`
- verify/2 driver: `dev/intralayer/verify/2_songyang_w1_regression_repro/driver.py`
- Pre/post run JSONLs: `dev/intralayer/verify/2_songyang_w1_regression_repro/runs/{iso,diag,fix}_*.jsonl`
- This document.
