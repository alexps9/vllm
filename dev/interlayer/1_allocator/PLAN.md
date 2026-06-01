# 1_allocator — implementation plan

First implementation phase of the vLLM interlayer design
([`../design.md`](../design.md)). Build a **minimal two-level (sub-block)
allocator** so attention allocates at `kernel_block_size` (32) sub-blocks packed
into mamba-sized physical pages (1056), instead of whole 1056-token pages —
eliminating the page-size bubble. Mamba keeps whole pages. Honors vLLM's
append-only block ids; no kernel change (proven: `0_feasibility/`).

Reading order: [`../design.md`](../design.md) → this → `journal/NN_*.md` (work
log) → TaskList.

## Ideal architecture (the target)

One **shared physical page pool** (the existing `BlockPool`, page = mamba size).

- **Mamba** draws **whole pages** from `BlockPool` directly — unchanged.
- **Attention** draws **sub-blocks** from a new **`SubBlockPool`** that sits
  *between* the attention manager and `BlockPool`: it grabs a whole page from
  `BlockPool`, carves it into `R = block_size/ksize` (= 33) sub-blocks, and
  hands those out with **packing bias** (fill a partially-used page before
  opening a fresh one). A page is returned to `BlockPool` (mamba-usable) **only
  when all its sub-blocks are free**.
- **Net-new structures** (all proven in `0_feasibility/`): per-sub-block
  ref-counting, a sub-page-aware free structure, a sub-page prefix cache, and
  the **`IndexedHeap` "cheapest page to vacate"** decision structure
  (`decision_cost` chose eager-delete; sub-µs bounded, no bloat).

Why a *parallel* `SubBlockPool` (not mutating `KVCacheBlock`/`BlockPool` in
place): keeps the whole-page pool + mamba path + the HiMA L1 queue untouched
(flag-off = byte-for-byte stock), and isolates all sub-block bookkeeping behind
one object — the structurally-clean boundary the integration map found.

## The integration seam (file:line, from the code map)

| # | hook | change | risk |
|---|---|---|---|
| 1 | `vllm/v1/core/block_pool.py` `get_new_blocks`/`free_blocks`/`touch` (345-445) | `SubBlockPool` *calls* these for whole pages; not modified | low |
| 2 | `single_type_kv_cache_manager.py` `FullAttentionManager.allocate_new_blocks` (243-270), `get_num_blocks_to_allocate` (89-168) | attention group: route to `SubBlockPool` at ksize granularity (flag-gated) | moderate |
| 3 | **manager↔block_table contract**: `block_table.py:108-109` fans page id `b → b*bpk+[0..bpk)`. With sub-block alloc the manager already holds **kernel-block ids** → attention group must set `bpk=1` (no fan-out) and pass kernel ids straight through | this is the cuda_graph integration residual (#95) — get it exactly right | **moderate-high** |
| 4 | sub-page prefix cache (`cache_full_blocks`/`find_longest_cache_hit`) → hash/align at ksize (#94) | new map keyed hash→sub-block | moderate |
| 5 | mamba-needs-a-page reclaim: when no page fully free, the **policy** (evict cheapest / preempt / defer) | **placeholder** here (e.g. preempt via existing vLLM path); real cost-model policy = `2_cost_reclaim` + L2 (#98) | deferred |

## Stages

**Stage 0 — scaffolding (no behavior change).** Flag
`VLLM_INTERLAYER_SUBBLOCK` (mirror the HiMA config pattern,
`hima/config.py`); new module `vllm/v1/core/interlayer/`. Flag-off ⇒ stock.
*Gate*: full existing KV-cache test suite green, unchanged.

**Stage 1 — `SubBlockPool` data structure, isolated + unit-tested.** Port the
audited prototypes to real classes: per-sub-block ref-count + cached lifecycle +
packing bias (from `sub_block_allocator/fuzz_refcount.py`), the `IndexedHeap`
cheapest-page-to-vacate (from `decision_cost/microbench.py`). NOT wired into the
live path. *Gate*: unit tests re-running the fuzz invariants (0 violations) +
the decision correctness/latency, against the REAL classes.

**Stage 2 — wire attention allocation (flag-gated).** Route the attention
group's alloc/free through `SubBlockPool`; resolve seam #3 (bpk=1, kernel ids
straight through) so the block table + captured CUDA graph get correct scattered
ids. Mamba untouched. *Gate*: flag-on hybrid **smoke run** produces output
numerically-equivalent to stock (per `virtual_split`); flag-off identical.

**Stage 3 — sub-page prefix cache (#94).** Hash/align attention prefix at ksize;
verify zero correctness violations + hit length rounds to ksize not 1056.

**Stage 4 — reclaim placeholder.** Mamba-needs-page with none free → simple
preempt/defer via the existing vLLM mechanism (no cost model yet). The bounded
cost-model policy is `2_cost_reclaim`.

## Verification gate (falsification)

- **No regression**: with the flag OFF, the entire existing v1 KV-cache /
  prefix-cache / hybrid test suite passes unchanged.
- **Correct**: with the flag ON, a hybrid smoke run on the target model is
  **numerically equivalent** to stock (same bar as `virtual_split` — not
  bit-identical; fp reduction-order differences allowed).
- **The win (preview)**: workload-weighted KV waste drops from ~42.6% toward the
  ksize counterfactual (~1%); the full e2e/throughput claim is `3_e2e_win`.
- FALSIFIED if: existing tests break with flag off; output diverges beyond
  fp-equivalence with flag on; or the append-only / cuda-graph invariants break.

## Dependencies & carried items
- `2_cost_reclaim` needs the **L2 cost model** (#98) — not needed for Stages 0-3.
- Carried: decision structure = `IndexedHeap` (#100 real-engine profiling);
  prefix cache verified here (#94); cuda_graph integration residual = seam #3
  (#95 proved the kernel tolerates scatter); L1 lazy-heap fix (#99, separate).

## Status
Stage 1 ✅ done (SubBlockPool module + 15 unit tests; isolated, no live-path
change). Next: Stage 2 (wire attention alloc, flag-gated) — touches the live
allocation path + the block_table contract (seam #3); needs go-ahead.
