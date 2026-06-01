# decision_cost — feasibility check (phase 0)

**PASS** (audited ×4). The per-step "cheapest page to vacate for mamba"
decision is incremental and correct on a realistic **correlated** workload.
Chosen structure: **`IndexedHeap` (eager-delete)** — O(1) peek (~140 ns, flat
across 256× pool size), **bounded sub-µs tail**, **no bloat** (1.0×), 0
violations (property test 360k ops). Per-op maintenance O(log P), ~1–3 µs
(~4× LRU, mostly hand-rolled-Python-vs-C; µs ≪ a 10–50 ms step). The production
`LPBPriorityQueue` is lazy-delete → real O(P) ~70 ms peek spikes + 600–900×
bloat (task #99); eager-delete is the structural fix.

- `microbench.py` — pure-CPU; correlated alloc/free (mirrors
  `../sub_block_allocator/fuzz_refcount.py`), all-page vacate-cost heap,
  `IndexedHeap` vs lazy `LazyHeap`, + a permanent IndexedHeap property test.
- `RESULTS.md` — verdict + v1→v3 history (two audits, each changed the result).
- `runs/microbench.out` — captured run.

Full spec + ideal pass bar: [`../../design.md`](../../design.md) verification gate.
