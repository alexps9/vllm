# sub_block_allocator — RESULTS

> ℹ️ **interlayer CLOSED — not pursued.** This check passed (feasibility was never the blocker); the effort was dropped for *value* reasons. See [`../../POSTMORTEM.md`](../../POSTMORTEM.md).

**PASS (memory-safety, ref-counted).** The two-level allocator stays
memory-safe — including the design's *named net-new structures*
(per-sub-block ref-counting under prefix sharing, the cached-block eviction
lifecycle, append-only ids) — under heavy randomized fuzz with **zero
invariant violations**.

> **Audit correction.** The first model (`fuzz_allocator.py`) was
> **single-owner only** (`occ[slot]=req`) and so never tested ref_cnt>1,
> decrement-to-zero, premature-free of a shared block, leak, or the
> cached-but-unreferenced lifecycle — exactly the design's hard net-new parts
> (`design.md` net-new structures; gate requires "ref-counts exact"). Its
> "memory-safe" was therefore only earned for the slab/packing bookkeeping.
> `fuzz_refcount.py` models the real semantics and is the verification of
> record.

## Faithful model (`fuzz_refcount.py`, mirrors `block_pool.py`)

- sub-block id = `page*K + slot` — **stable / append-only** (reclaim by
  eviction, never relocation; `block_pool.py:48-52`).
- `ref_cnt[id]` = #live requests referencing it; `touch` (prefix-cache hit by
  another req) → `ref_cnt++`; `free` → `ref_cnt--`; at 0 the block becomes
  **cached** (kept for reuse, evictable).
- a page is mamba-usable **iff all K sub-blocks have ref_cnt==0** (cached ones
  on it are evicted as part of the take).
- packing bias: fill partially-used attn pages before opening an all-free one.

## Results (zero violations everywhere)

every-op check (K=33, 20k ops, `check()` after **every** op): 0 violations.

3 seeds × 150k ops, K∈{33,66}, `check()` every 500 ops + drain + leak check:

| K | max ref_cnt during | shared touches | cached evicts | mamba starve | leak | violations |
|---|---:|---:|---:|---:|---|---:|
| 33 | 270–297 | ~37.6k | 65k–67k | ~26.5k | clean | **0** |
| 66 | 248–284 | ~37.6k | 130k–137k | ~26.5k | clean | **0** |

Invariants enforced (raise on breach): `ref_cnt ≥ 0`; `ref_cnt == #live
owners`; no NEW-alloc of a `ref_cnt>0` block; no decrement-below-zero; no
free-of-unowned; mamba page has no live attn refs; mamba pages distinct;
**leak check** (after draining all requests, every sub-block `ref_cnt==0` and
every page returned).

## What this establishes / scope

- ✅ The allocator is **memory-safe under ref-counted prefix sharing**
  (ref_cnt reached ~290), the **cached eviction lifecycle**, decrement-to-zero
  with **no premature free and no leak**, and the **mamba↔attention page-flip**
  — at K∈{33,66}, checked every op on a short trace.
- ⚠️ Still reproduces the **P1 starvation** (mamba_starve ~26k): under heavy
  attention pressure mamba can't always get a whole page. Phase 2 owns
  *safety*; bounding starvation under *realistic* load via cost-model reclaim
  is `cost_reclaim/` (phase 3).
- Scope: a standalone prototype, not the vLLM `BlockPool` integration. It
  proves the design's allocator *can* be safe with the real semantics; the
  integration must preserve these invariants + the append-only constraint.
