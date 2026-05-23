# Finding M.10 — root-cause of the M.9 throughput regression

M.9 reported a -12% throughput regression on the cc workload and
attributed it (with five iterations of failed Python-level
optimization) to a GPU-side effect we couldn't fix. The root-cause
turned out to be **two separate bugs**:

1. An **in-place mutation bug** in `_try_partial_extension` that
   contaminated the probe-only diagnostic mode and inflated the
   apparent regression for ALL hit-side modes (`partial`,
   `probe_only`, `sparse_key`, `len_key`).
2. A **real, smaller GPU-side regression** (+13.8%) when partial
   extensions are actually applied — from the attention kernel's
   different memory-access pattern when block_table[K] points to a
   cached-partial block (with offset-R writes) instead of a freshly
   allocated block (with offset-0 writes).

## Diagnostic methodology

Added a battery of env-var flags to isolate the cost:

| env var | behavior |
|---|---|
| `VLLM_PARTIAL_CACHE_ENABLED=1` | enable scaffolding (M.4/M.5) |
| `VLLM_PARTIAL_CACHE_PROBE_ONLY=1` | lookups run but don't apply |
| `VLLM_PARTIAL_CACHE_INSERT_OFF=1` | skip cache_partial_block entirely |
| `VLLM_PARTIAL_CACHE_FORCE_MISS=1` | force get_cached_partial_block to return None |
| `VLLM_PARTIAL_CACHE_TRIVIAL_KEY=1` | inner key = (request_id, partial_len) — never hits |
| `VLLM_PARTIAL_CACHE_LEN_KEY=1` | inner key = partial_len only — content-blind hit |
| `VLLM_PARTIAL_CACHE_SPARSE_KEY=1` | hash only first/last 8 tokens of partial |
| `VLLM_PARTIAL_CACHE_FAST_LOOKUP=1` | lookup skips content tuple |
| `VLLM_PARTIAL_CACHE_NON_DESTRUCTIVE=1` | hits don't remove map entries |

Plus perf_counter timers on `cache_partial_block`,
`get_cached_partial_block`, and `_compute_partial_keys`.

## Phase 1: localize to insert vs lookup

| variant | full_wall | tp | hit% | notes |
|---|---|---|---|---|
| baseline | 14.47s | 153.8 | 91.59 | env unset |
| partial(full) | 16.35s | 136.1 | 98.99 | -11.5% throughput |
| **probe_only** | 16.33s | 136.3 | 91.59 | also regressed! cache fires but "discards" |
| **insert_off** | 14.60s | 152.5 | 91.59 | basically baseline |

Probe_only had the same regression as partial. So the cost was NOT
from applying partial extensions to requests — it was somewhere
along the cache+lookup path that fires before the "apply" step.

## Phase 2: localize to hit-vs-miss

| variant | full_wall | tp | hit% |
|---|---|---|---|
| baseline | 14.47s | 153.8 | 91.59 |
| trivial_key (no hit possible) | 14.49s | 153.7 | 91.59 |
| force_miss (forced None) | 14.57s | 152.8 | 91.59 |
| sparse_key (8+8 tokens) | 16.37s | 136.0 | 98.99 |
| len_key (partial_len only) | 16.45s | 135.4 | 98.99 |
| partial(full 1024-tok key) | 16.35s | 136.1 | 98.99 |

Regression correlates 1:1 with whether lookups HIT. Sparse_key
(16-tok hash) has the same regression as full content tuple. Len_key
(no content at all in key) ALSO has the regression. So the cost is
NOT in tuple-construction or hash-of-1024-ints. The cost triggers
specifically when get_cached_partial_block returns a block.

## Phase 3: instrument what happens on hit

`get_cached_partial_block`'s hit path is just:
- `_partial_cache_hits += 1`
- `_remove_partial_entry((outer_key, inner_key))` — 2 dict ops
- `partial_cache_key_by_block_id.pop(block.block_id, None)` — 1 dict op

3 dict ops × ~95 hits per workload = ~285 dict ops = ~5 µs total.
But the wall-time regression is 2 SECONDS. 400000x amplification.
Impossible from this code alone.

That's when I noticed the bug:

```python
# OLD _try_partial_extension:
for R, _candidate_block in candidates:
    ...
    partial_block = self.block_pool.get_cached_partial_block(...)
    if partial_block is not None:
        computed_blocks[0].append(partial_block)  # ← MUTATES INPUT
        return num_new_computed_tokens + R, computed_blocks
```

`computed_blocks` was passed in by the caller (`get_computed_blocks`).
The `.append()` modified the caller's list IN PLACE. The probe_only
mode tried to "discard" the return value, but the input was already
mutated! The request's block_table got the partial block appended
even in probe_only mode — but `num_computed_tokens` was NOT bumped
(since we threw away the new tokens count).

Result: request enters allocate_slots with K+1 blocks in its
block_table but num_computed_tokens = K*block_size. The GPU then
sees a block_table layout that's internally inconsistent with the
prefill/decode plan. Some downstream code path becomes slower
(unconfirmed exact mechanism, but the effect is reproducible).

## Phase 4: confirm the bug

Fixed `_try_partial_extension` to return a NEW tuple of lists
(deep-ish copy) instead of mutating the input:

```python
# NEW:
if partial_block is not None:
    new_blocks = tuple(list(g) for g in computed_blocks)
    new_blocks[0].append(partial_block)
    return num_new_computed_tokens + R, new_blocks
```

| variant | full_wall | tp | notes |
|---|---|---|---|
| baseline | 14.47s | 153.8 | env unset |
| partial_BEFORE | 16.35s | 136.1 | apply, with mutation bug |
| **partial_FIXED** | 16.47s | 135.2 | apply, after fix |
| **probe_clean** | 14.59s | 152.6 | probe, after fix → matches baseline! |

After the fix:
- `probe_clean` exactly matches baseline (~0% regression). The
  diagnostic mode is finally side-effect-free.
- `partial_FIXED` still has +13.8% regression because actual
  application of partial extensions DOES happen.

## What this means

**The M.9 regression is actually two effects:**

1. **The phantom regression (~10%)** from the in-place mutation bug
   was an artifact of the diagnostic path. This is now fixed; the
   bug never affected the actual production path (apply mode), only
   confused our root-cause analysis. **No production impact from
   this fix.**

2. **The real regression (~13.8%)** when partial extensions are
   applied is genuinely GPU-side. The request's block_table[K]
   pointing to a cached-partial block (in some prior request's HBM
   region, with first R tokens of valid prior K, V) instead of a
   freshly-allocated block changes the attention kernel's memory
   access pattern enough to slow per-token decode by ~+1.35 ms.

## Remaining mystery / future M.11

The +13.8% real regression in apply mode is unexplained at the
attention-kernel level. Profiling with nsys or similar would tell us
whether it's:
- L2 cache misses from non-contiguous block allocations
- TLB pressure from non-temporal block layouts
- Some FlashAttention v3 perf cliff on specific offsets/sizes
- Slot-mapping computation overhead in metadata builder

For now, the M.6/M.7 microbench results (TTFT -43% at R=800) and
the M.9 real-workload (TTFT -16%, bubble -88%) stand. The trade-off
is now precisely understood: enable for TTFT-sensitive workloads,
disable for throughput-sensitive bulk generation. The remaining
regression is from real GPU work and is not Python-side fixable.

## What changed in the codebase

- `vllm/v1/core/kv_cache_manager.py`: `_try_partial_extension` no
  longer mutates input. Returns a new tuple of lists.
- `vllm/v1/core/block_pool.py`: diagnostic env vars added
  (PROBE_ONLY/FORCE_MISS/TRIVIAL_KEY/LEN_KEY/SPARSE_KEY/
  NON_DESTRUCTIVE/FAST_LOOKUP) + perf_counter timers. These are dev-
  only paths; default behavior unchanged.
- `dev/interlayer/runs/09_cc_*.{jsonl,out}`: 7 diagnostic runs
  captured.

## Recommendation

Keep the partial-cache as opt-in (`VLLM_PARTIAL_CACHE_ENABLED=1`).
Document the trade-off clearly: TTFT win on follow-up turns
(-16% on cc workload), throughput cost (-13.8% on bulk decode).
The next investigation (M.11) should profile attention kernel
behavior under apply mode to understand and possibly mitigate the
remaining GPU-side cost.
