# decision_cost — feasibility check (phase 0)

**PASS** (audited ×2). The per-step "cheapest page to vacate for mamba"
decision is incremental (O(log P) — flat across 256× pool size; 71–157×
cheaper than the O(P) re-walk), correct (0 violations), and ~2.4× LRU per op,
on a realistic **correlated** workload. **Requires heap compaction** to bound
memory (else 600–900× bloat, inherited from production `LPBPriorityQueue` —
task #99); compaction → ≤8×, and is the design's "steady-state rebalance."

- `microbench.py` — pure-CPU; correlated alloc/free (mirrors
  `../sub_block_allocator/fuzz_refcount.py`), all-page vacate-cost heap.
- `RESULTS.md` — verdict + v1→v3 history (two audits, each changed the result).
- `runs/microbench.out` — captured run.

Full spec + ideal pass bar: [`../../design.md`](../../design.md) verification gate.
