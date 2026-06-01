# 01 — Stage 1: SubBlockPool data structure (DONE)

Ported the audited `0_feasibility/` prototypes to a real, isolated vLLM module
+ unit tests. **No live-path change** (nothing imports the package yet) ⇒ zero
blast radius; `import vllm` fine, existing `tests/v1/core/test_kv_cache_utils.py`
(57) unaffected.

**Module:** `vllm/v1/core/interlayer/sub_block_pool.py`
- `IndexedHeap` — eager-delete (O(1) peek, O(log n) update/remove, no stale),
  the structure `decision_cost` chose (audited ×4).
- `SubBlockPool(sub_per_page R, page_source)` — sits between the attention
  manager and the shared page pool. Carves a page into R sub-blocks; hands out
  with packing bias; per-sub-block ref-count + cached lifecycle; a page returns
  to the source iff all sub-blocks free (pure-empty → returned immediately;
  cached-but-free → reclaimable, ranked in the heap by `#cached` placeholder
  vacate-cost); `reclaim_page_for_mamba()` vacates the cheapest. Append-only
  ids `pid*R + slot`. `check()` enforces full invariants.
- `page_source` is a `Protocol` (`alloc_page`/`free_page`) → decoupled from
  `BlockPool`; the real wiring is Stage 2.

**Tests:** `tests/v1/core/interlayer/test_sub_block_pool.py` — **15 passed**.
- IndexedHeap vs dict+min reference, every-op invariants, 8 seeds (incl.
  interior remove / remove-min / drain).
- packing bias (4 sub-blocks → 1 page, 5th opens a 2nd); page returns when
  fully free; reclaim picks the cheapest reclaimable page; prefix sharing
  ref>1 → free-to-1-stays-live → free-to-0-caches.
- randomized fuzz (R∈{4,33}) with every-op `check()` + end-of-trace leak check:
  0 violations, sharing exercised (ref≥2), all pages returned, no dangling.

**Cost note:** the reclaim ranking uses `#cached` as a placeholder vacate-cost.
The real recompute-cost (and the preempt-vs-evict policy when nothing is
reclaimable, i.e. `reclaim_page_for_mamba()` returns None) is `2_cost_reclaim`
+ L2 (#98).

**Next — Stage 2 (riskier, touches live path):** add the `VLLM_INTERLAYER_SUBBLOCK`
flag; route the attention group's alloc/free through `SubBlockPool` (page_source
= `BlockPool`); resolve seam #3 (manager now holds kernel-block ids → `bpk=1`,
scattered ids straight into the persistent block-table the CUDA graph baked).
Gate: flag-off byte-identical (full suite), flag-on hybrid smoke numerically
equivalent.
