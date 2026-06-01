# 00 — kickoff

**State at start:** feasibility gate ([`../../0_feasibility/`](../../0_feasibility/))
closed — page_bubble (42.6% waste), virtual_split (kernel at ksize=32, no kernel
change), sub_block_allocator (two-level allocator memory-safe, audited),
decision_cost (cheapest-page decision incremental + correct → `IndexedHeap`,
audited ×4), cuda_graph (scattered block-table safe under captured graph,
audited). prefix_cache reclassified into this phase (#94).

**Goal of 1_allocator:** the *minimal* real two-level allocator in vLLM —
enough to show the bubble is eliminated with no regression and output stays
numerically-equivalent. Not feature-complete; the cost-model reclaim policy is
deferred to `2_cost_reclaim` (needs L2, #98).

**Code map (from the integration-seam exploration):** `BlockPool` is the single
clean allocator both managers call; append-only block ids are the invariant
(`block_pool.py:52`). Chosen architecture: a **parallel `SubBlockPool`** between
the attention manager and `BlockPool` (carves whole pages into 33 sub-blocks,
packing bias, returns a page only when all sub-blocks free) — keeps mamba + the
whole-page pool + HiMA L1 untouched and flag-off byte-identical. See
[`../PLAN.md`](../PLAN.md) for stages + the 5 seam hooks.

**Sharpest risk identified up front (seam #3):** the manager↔`block_table`
granularity contract. Today the manager hands whole-page ids and
`block_table.py:108-109` fans `b → b*bpk+[0..bpk)`. With sub-block allocation
the manager already holds *kernel-block ids*, so the attention group must use
`bpk=1` and pass kernel ids straight through — and those scattered ids must land
in the **exact persistent `input_block_tables` buffer** the captured CUDA graph
baked (the cuda_graph #95 integration residual). Get this exactly right in
Stage 2.

**Plan for the work:** Stage 0 (flag + module skeleton, no behavior change) →
Stage 1 (`SubBlockPool` data structure, isolated, unit-tested against the
audited fuzz/decision properties) → Stage 2 (wire attention alloc, flag-gated,
smoke-equiv) → Stage 3 (sub-page prefix cache) → Stage 4 (reclaim placeholder).
Stages 0-1 are low blast radius (no live-path change); Stage 2 onward touches
the live allocation path.

**Next:** Stage 0 + Stage 1.
