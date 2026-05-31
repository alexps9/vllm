# interlayer (vLLM) — design

> Cross-pool capacity for vLLM's hybrid models (paged-attention KV +
> recurrent/mamba state). Counterpart to sglang's `dev/interlayer/`, but
> vLLM's bubble has a **different shape**, so the fix is different.
> Kept deliberately small — this is a problem-proof + direction doc, not
> an implementation spec.

## The problem — vLLM's bubble is a *page-size* bubble

Hybrid models have two kinds of state with orthogonal demand:

- **KV** (attention): bytes ∝ total tokens in flight. Natural page small
  (16 tokens).
- **Mamba** (recurrent): one fixed, *indivisible* state per request,
  stored in **fp32**. Natural page large.

**sglang** keeps two physically separate pools split at boot
(`mamba_full_memory_ratio`); its bubble is the *fixed split* — one pool
idle while the other binds. It fixes that by VMM-remapping physical pages
between the two pools at runtime.

**vLLM is different.** It already uses **one fungible block pool** shared
by both (`kv_cache_utils.py:1290-1315` — groups draw different block-ids
from the same free list), so it has **no fixed-split bubble**. But the
price of one shared pool is that **every block must be the same byte
size** = `max(attention_page, mamba_page)`
(`unify_kv_cache_spec_page_size`, `kv_cache_utils.py:1012-1049`). The
mamba page is the larger one, so vLLM **inflates the attention block_size
to match** — on Qwen3.5-35B-A3B that is **1056 tokens** (66× the default
16). Engine says so verbatim: *"Setting attention block size to 1056
tokens to ensure that attention page size is >= mamba page size."*

So vLLM's bubble is **internal fragmentation**: attention KV is allocated
in 1056-token blocks, but real requests (and every request's ragged tail)
fill only a fraction. Blocks are all allocated, yet a large share of the
slots inside them sit empty. Measured on 106 real Claude-Code sessions:
**42.6% workload-weighted waste, p95 >130%** (vs 0.69% at the natural
block_size=16). See [`0_page_bubble/`](0_page_bubble/).

Two costs of the same root: (1) **memory** — wasted KV capacity → fewer
concurrent reqs / less prefix cache; (2) **compute** — coarser
prefix-cache reuse (only at 1056-token boundaries). The earlier pcache
attempt targeted (2) via partial-block caching and **failed on hybrid**
(mamba is block-granular, can't resume mid-block — see
[`0_page_bubble/08_hybrid_architectural_blocker.md`](0_page_bubble/08_hybrid_architectural_blocker.md)).
This effort targets (1).

## Why we can't copy sglang

- vLLM has **no VMM substrate** (no `cuMemCreate/cuMemMap/cuMemUnmap`,
  no growable arenas — confirmed absent in `vllm/v1/`). Pool tensors are
  fixed at boot.
- And we don't *need* sglang's cross-pool remap: vLLM's pool is already
  fungible. The bubble isn't "wrong split", it's "page too coarse".

## The lever (direction — not yet designed)

The fix is to **break the uniform-page constraint** so attention can use
a small page while mamba keeps its big one. Candidate directions (open):

- **Per-group page size**: let the block pool hold groups with different
  `page_size_bytes` instead of forcing one max. Requires reworking the
  single-pool `num_blocks` accounting.
- **Sub-block attention allocator**: keep the 1056 physical page but pack
  multiple short attention sequences / fine-grained spans into one page.
- **Decouple mamba bytes**: shrink the mamba page (e.g. bf16 SSM where
  numerically safe) so the forced attention block_size drops.

Each needs its own verify phase. None committed yet.

## Status / phases

| phase | what | status |
|---|---|---|
| [`0_page_bubble/`](0_page_bubble/) | **prove the bubble exists** (size inflation + realistic-trace waste) | restored from git (commit `438ad0397`, was deleted with pcache); re-validating |
| (future) | solution direction A/B/C above + verify each | not started |
