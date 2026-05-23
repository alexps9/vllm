# vLLM-side "interlayer" design space: eliminating the partial-block bubble

> The sister sglang project has a multi-pool L2 actuator (cuMemUnmap +
> cuMemMap, 32 MiB pages, two-loop budgeter). vLLM cannot adopt this
> directly because vLLM uses a SINGLE inflated pool. This doc surveys
> mechanisms that fit vLLM's architecture for eliminating the
> equivalent bubble (per-request last-block waste due to block_size
> inflation).

---

## 0. The bubble in numbers

From `dev/README.md` Finding D, measured on Qwen3.5-35B-A3B (Gated
DeltaNet hybrid) replaying 106 real Claude Code sessions:

- block_size inflated **16 → 1056** by `_align_hybrid_block_size` (lcm
  of attention's native 16 and mamba's 1056-token page size — see
  [`vllm/platforms/interface.py:613-674`])
- **42.62%** workload-weighted partial-block waste: nearly half of
  every turn's new content tokens get *trapped* in a partial last
  block at end-of-request and are re-prefilled on the following turn.

The bubble has two root causes:

1. **Per-turn last-block waste.** A request of length L allocates
   `ceil(L / 1056)` blocks. Tokens in `[L:ceil(L/1056)*1056]` (avg
   528 tokens) are physically reserved but logically empty. When the
   request ends, those reserved tokens are abandoned.

2. **Prefix-cache truncation to full-block boundary.** When vLLM
   caches a request's blocks for future prefix-cache use, the code at
   [`vllm/v1/core/single_type_kv_cache_manager.py:299`] computes
   `num_full_blocks = num_tokens // block_size` (integer division).
   Anything in the last partial block is **never cached**. When the
   next turn arrives with the same prefix, the prefix cache only hits
   up to the last full-block boundary, and the partial-block content
   must be re-prefilled.

The second cause is what bites multi-turn workloads, because each
turn extends the previous prefix slightly and the previous turn's
partial-block tail is exactly what the new turn wants to re-use.

---

## 1. What we're actually trying to do

> **Goal**: turn the 42.6% bubble into ≤ 5% real-content waste, at
> the cost of bounded book-keeping overhead in hot paths, without
> touching attention or mamba kernels.

A bubble-killing layer must do ALL of:

1. Make the partial last block of a finishing request **referenceable** by future requests.
2. Survive the same eviction pressure that affects full blocks (LPB-aware).
3. Not break the PagedAttention contract: each physical block belongs to one request's write stream at a time; a sequence's logical block `n` maps to exactly one physical block at exactly one offset 0.

That last constraint kills the simplest fantasies (multiple requests
sharing one block at different offsets, attention reading two blocks
to form one logical block, etc.). It leaves three real options.

---

## 2. Candidate designs

### Candidate A: Partial-block cache with copy-on-write reuse (recommended for first prototype)

**Idea.** When a request finishes with `R` tokens in its last partial
block (where `0 < R < block_size`), compute a partial-block hash for
`R` tokens and insert the block into a second cache map:
`partial_cache: { (parent_full_block_hash, R, hash_of_first_R_tokens) → KVCacheBlock }`.

On the next request's `find_longest_cache_hit`, after finding the
longest full-block prefix hit (call its length `K * block_size`),
also try to extend with a partial-block hit:
1. Compute hash of the next R tokens of the request.
2. Look up `partial_cache[(parent_hash, R, partial_hash)]`.
3. If hit, the new request has `K * block_size + R` cached tokens.

When the new request is admitted with the partial block as a "soft
hit":
- The partial block is **read-only shared** for `[0:R]`.
- The new request needs to write into positions `[K*block_size+R, ...]`,
  which fall inside the partial block at offset `R+1, R+2, ...`.
- That conflicts with the partial block's read-only status, so we do
  **copy-on-write**: allocate a fresh block from the free queue,
  memcpy `[0:R]` from the partial block into it, then write the
  new tokens into offsets `[R+1, ...]` of the COW block.

**Cost per cache hit.** Memcpy of R × `page_size_1_token` bytes per
layer × num_layers. For Qwen3.5-35B on H200, attention KV at R=528:

```
attention: 528 × 1024 B × 64 layers = 35 MB → ~12 µs @ 3 TB/s HBM
mamba    : SSM state has no per-token caching at sub-block;
           mamba MUST re-prefill the partial block's R tokens.
           For R=528: ~500 µs of mamba prefill, but mamba prefill
           is linear-cost so this is much cheaper than attention
           prefill for the same R.
```

**Saving per cache hit.** Attention prefill of R tokens. For R=528 at
~80 TFLOPs sustained on H200: ~460 µs saved.

**Net.** ~450 µs saved per hit. ROI ~ 38× per cache hit. Scales with
hit rate, which on multi-turn cc workload is ~95% per turn (from
existing Finding K data).

**Implementation surface** (estimated 200-300 LOC):
- `vllm/v1/core/kv_cache_utils.py` — add `PartialBlockHash`,
  partial-hash computation helper.
- `vllm/v1/core/block_pool.py` — add `cached_partial_block_map`,
  `cache_partial_block()`, modify `get_cached_block()` to optionally
  return `(block, R)` for partial hits, add COW path for partial reuse.
- `vllm/v1/core/single_type_kv_cache_manager.py` — modify
  `cache_blocks()` to also cache the partial last block; modify
  `find_longest_cache_hit()` (and per-attention-type overrides) to
  attempt partial-extension after full hits.
- `vllm/v1/core/kv_cache_coordinator.py:493-495` — relax the floor at
  `lcm_block_size`; allow returning `cache_hit_length` not on a
  block-multiple boundary.
- `vllm/v1/core/kv_cache_manager.py` — propagate the new
  partial-hit-length through `compute_blocks()` / `get_computed_blocks()`.
- Model runner glue — for mamba groups, recognize partial cache hit
  on attention but treat mamba as full-block cached only (no change
  to mamba kernels).

**Risks / unknowns:**
- The chunked-prefill path may not handle a starting position that is
  not block-aligned; need to verify.
- Eagle/MTP draft heads have their own `find_longest_cache_hit` that
  drops the last block on purpose — partial hit may interact weirdly.
- The COW memcpy needs a Triton/CUDA kernel to be cheap enough; pure
  PyTorch `block[:] = src_block[:R]` would dispatch many small ops.

### Candidate B: Sub-block prefix-cache hashing (`hash_block_size << block_size`)

**Idea.** vLLM already supports `hash_block_size` distinct from
`scheduler_block_size`/`block_size` (see
[`vllm/v1/core/kv_cache_utils.py:575-634`]). Currently the hybrid
hot-path sets `hash_block_size = gcd(group_block_sizes) = 1056`.
What if we set `hash_block_size = 16` so the hash table indexes
prefix matches at 16-token granularity?

**Why it doesn't help here.** Even with finer hashes, the physical
cache is still at 1056-token granularity — `cache_full_blocks` only
records full 1056-token blocks. The hash refinement only matters
when there's *some other* mechanism (P/D, offloading,
`HashListWithBlockSize`) consuming the finer hashes. For local
prefix caching, fine hashes cannot recover the partial tail because
the partial-block content was never cached.

**Useful as a building block for A**, not as a standalone solution.

### Candidate C: Inflate-side reduction (pick a smaller `mamba_block_size`)

**Idea.** Reduce `mamba_page_size` so the lcm-inflate gives a smaller
attention `block_size`. E.g., switch mamba SSM state from fp32 to
bf16 (halves it from 1048 KiB → 528 KiB → attention block_size
inflates to 528 not 1056). Counterfactual already explored in
`dev/counterfactual_block_size.py`.

**Why it's the wrong layer.** This is a model-precision change that
affects mamba's numerical behaviour. Out of scope for "vLLM-side
mechanism", and even at block_size = 528 the bubble is still
~21% — only halves the problem.

### Candidate D: HiMA-style VMM remap of the partial block

**Idea.** This branch already exposes CUDA VMM via
[`vllm/v1/core/hima/actuator/cuda_driver.py`] and
[`vllm/v1/core/hima/actuator/vmm_pool.py`]
(`cuMemUnmap`, `cuMemMap`, `cuMemAddressReserve`). On partial-block
hit, instead of COW memcpy, remap the partial block's physical pages
into the new request's VA range at the right offset.

**Why deferred.** VMM remap operates at 2 MiB page granularity (the
CUDA driver minimum). A partial block is ~1 MB of attention KV per
layer — smaller than a single VMM page. To use VMM remap usefully,
we'd need to either (a) pack many partial blocks into a single VMM
page and remap collectively, or (b) operate at coarser granularity.
Either way, much more invasive than COW for the same end-user effect.

Revisit if COW memcpy turns out to be the bottleneck.

### Candidate E: Tail-pool defragmentation (compact many partial blocks into one)

**Idea.** Periodically (every N steps, or when the partial cache hits
a watermark) compact all live partial blocks into a single "tail-pool"
block that holds the first R₁ tokens of partial-block 1, then R₂
tokens of partial-block 2, etc. Saves physical memory: 10 partial
blocks at R≈500 fit into 5 full blocks instead of 10.

**Why deferred.** Saves block-pool capacity, not prefill compute.
The main bubble cost is wasted compute on re-prefill, not wasted HBM.
And the compaction memcpy is expensive (full block per partial). The
compute-side bubble is the win we want; HBM-side waste is secondary.

---

## 3. Recommendation

**Implement Candidate A first.** It's the only one that converts the
measured 42.6% compute waste directly into cache hits, and the COW
cost is well below the prefill cost it replaces. Estimated 200-300
LOC of Python touching 5-6 files, no kernel changes.

Sequence:
1. Build a microbench in `dev/interlayer/02_partial_cache_micro.py`
   that, **without any source change**, measures the existing
   bubble on a synthetic 2-turn workload (turn 1 fills `K * 1056 + R`
   tokens, turn 2 issues turn 1's prefix + 16 new tokens; measure
   TTFT degradation as a function of R).
2. Prototype the cache-side change behind a flag (`partial_cache_enabled`).
3. Re-run the microbench with the flag on, confirm TTFT collapse on
   high-R cases.
4. Wire to the end-to-end cc workload comparison (extend the
   compare_lru_lpb driver pattern) and confirm Finding K-level
   per-phase improvements.

---

## 4. Open questions for whoever picks this up next

- Does the **chunked-prefill** path tolerate `num_computed_tokens`
  that is not a multiple of `block_size`? If not, partial hits would
  only work in the non-chunked path.
- For Eagle/MTP, `find_longest_cache_hit` deliberately drops the
  last block. Does the partial-extension logic make sense here, or
  should partial hits also be dropped in the speculative-decode
  path?
- What's the right hash function for "first R tokens of a block"?
  Reusing the first-token hash is unsafe (different R's would
  collide). The cleanest answer is `hash(parent_hash, R, sha256_or_xxh3(tokens[0:R]))`.
- The COW write into the destination block needs to land at offset
  R of a freshly allocated block. Does that interact with prefix-
  cache hashing — i.e., when the COW block fills up, does its hash
  match the full-block hash computed earlier? (It should: same
  tokens in the same positions → same hash.)
- Mamba state at offset R: do we need it, or does mamba re-prefill
  the partial extent and we only avoid attention's re-compute? (I
  believe the latter is sufficient and is what the bubble metric
  is actually counting.)
