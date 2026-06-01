# Implementation plan — vLLM interlayer (page-size bubble)

Roadmap for landing the design in [`design.md`](design.md). Tasks are grouped
into phases by ship-gate dependency. Each has a **falsification criterion**.
Reading order: this file → `design.md` → per-phase folder README / TaskList.

**Methodology note (earned the hard way).** Every feasibility check below is
adversarially **audited** before it counts. Three audits in a row caught a
first-cut test verifying the easy/wrong thing (phase virtual_split tested
determinism not block-size invariance; sub_block_allocator tested single-owner
not ref-counting; the cost-reclaim *sim* had two bugs and tested the wrong axis
entirely). Lesson: **a simulation that re-derives the policy logic is
bug-prone and only proves the model, not the integration.** So properties that
depend on the real allocator + cost model + workload dynamics are verified on
a **minimal real implementation**, not a paper sim.

---

## Phase 0 — Feasibility gate (pre-implementation, in [`0_feasibility/`](0_feasibility/))

Cheap, self-contained checks that the design is even possible. Must pass
before writing any vLLM integration. Each is a subfolder; each gets audited.

| check | status | property | falsification |
|---|---|---|---|
| [`page_bubble`](0_feasibility/page_bubble/) | ✅ done | the bubble exists | waste at 1056 ≈ waste at 16 (it's 42.6% vs 0.69%) |
| [`virtual_split`](0_feasibility/virtual_split/) | ✅ done | kernel runs at `ksize=32 ≪ 1056`, valid attention at sub-page granularity, no kernel change | kernel pinned to 1056 / wrong output at 16-vs-32 (it's numerically equivalent; bit-identical was retired as fp-impossible) |
| [`sub_block_allocator`](0_feasibility/sub_block_allocator/) | ✅ done | two-level allocator is memory-safe **with ref-counting + cached lifecycle + sharing** | any invariant violation / leak (0 over 3 seeds × 150k ops + adversarial; ref_cnt→290) |
| [`decision_cost`](0_feasibility/decision_cost/) | ✅ done (audited ×4) | "cheapest page to vacate" incremental & correct; **structure chosen = `IndexedHeap` (eager-delete)**: O(1) query, bounded ~7µs tail, no bloat, ~4× LRU per-op (µs ≪ ms step → fine) | query O(P)/unbounded tail (lazy heap: 50ms — that's why eager-delete) / wrong page (0 viol). Lazy `LPBPriorityQueue` carries the defect → task #99 |
| [`cuda_graph`](0_feasibility/cuda_graph/) | ✅ done (GPU probe) | scattered sub-block block-table safe under captured-graph replay (prefill+decode: 0 faults, no recapture, bit-identical to contiguous; control differs) | replay fault / recapture / scattered ≠ contiguous ref — none |
| ~~prefix_cache~~ | → impl (#94) | reclassified to `1_allocator` | — |

**Gate**: page_bubble / virtual_split / sub_block_allocator / decision_cost ✅
done (audited). **`prefix_cache` reclassified → `1_allocator` (#94)**: its only
paper-checkable part ("hit rounds to ksize") is trivially true by a parameter
(hash chunk size); the real risk (collision / shared-sub-block ref-count /
eviction) is the cache machinery × the allocator — model≠integration, needs the
real impl. **`cuda_graph` (#95) ✅ done**: a minimal-GPU probe on the real
`flash_attn_varlen_func` confirmed the code-read — a scattered sub-block
block-table replays under a captured graph with **0 faults, no recapture,
bit-identical to the contiguous reference** (prefill + decode), and a control
(block-table → different KV) differs, proving the graph re-reads the live table.
Scattered ids are "just different data": **no kernel change, no eager-mode
fallback**. The feasibility gate is now **CLOSED**.

**Audit note (decision_cost):** took FOUR audits. v1 timer-inflated + dismissed
bloat; v2 unrealistic independent-slot workload + reclaimable-only heap; v3
(correlated + all-page vacate-cost) revealed the lazy-delete heap's unbounded
O(P) ~50 ms peek spikes hidden by the mean; v4 chose an eager-delete
`IndexedHeap` (O(1) peek, sub-µs tail, no bloat) — audit-4 confirmed the choice
(capping the lazy heap's trim instead gives 31% wrong answers) and sharpened
the numbers (maintenance is O(log P) not flat; 4× is mostly Python-vs-C). The
recurring lesson: first-cut tests verify the easy case / report the metric that
hides the bad behavior; audits force the realistic regime.

---

## Implementation phases (top-level `1_xxx`, `2_xxx`, … — created when the gate closes)

These verify properties that **cannot** be faithfully checked on paper — they
need a minimal real vLLM implementation. (This is why old "phase 3 / phase 7"
moved here out of `0_feasibility`.)

### 1_allocator — minimal two-level allocator in vLLM
Integrate the audited allocator design into `BlockPool` / `KVCacheManager`:
attention allocates at `kernel_block_size` sub-blocks (packing bias), mamba
keeps whole pages, honoring vLLM's append-only block-id constraint. Minimal,
not feature-complete.
- *Falsification*: existing vLLM KV-cache tests / a hybrid smoke run break;
  or attention output diverges beyond fp-equivalence (per `virtual_split`).

### 2_cost_reclaim — cost-model page reclaim (the make-or-break; was "phase 3")
On the Phase-1 implementation: when mamba needs a whole page and none is free,
evict the cheapest attention (cost-model) / preempt / defer. Verify **mamba
starvation, recompute amplification, and attention starvation stay bounded**
under KV-bound real + adversarial load, and ≥ stock.
- *Depends on*: **the L2 cost model** (removed 2026-05, redesigning) — at
  minimum a simple recompute-cost estimator. interlayer ⇄ L2 coupling.
- *Falsification*: mamba hard-starves unboundedly / recompute amplification or
  tail unbounded / attention starves / worse than stock on the target
  (agent / short-ragged) load.
- *Note*: the earlier paper-sim attempt was deleted — it had a partial-alloc
  leak, a recompute mis-charge, and tested attention sizing instead of mamba
  starvation (audit `abd3416634cab13b6`). Verify on real code, not a sim.

### 3_e2e_win — bubble eliminated, no regression (was "phase 7")
KV-bound agent load, n=3, fix vs stock on the real engine.
- *Falsification*: waste doesn't drop toward the ksize counterfactual (~1%) /
  throughput regresses / no real gain under KV pressure.

---

## Dependencies & ordering
- Phase 0 gate → Phase 1 (allocator) → Phase 2 (cost_reclaim) → Phase 3 (e2e).
- **L2 cost model is a hard prerequisite for Phase 2** — the two efforts are
  coupled; an accurate cost model is the real make-or-break (see `design.md`).
- Host note: GPU runs must be **isolated** (recurring CUDA-init wedge);
  `ninja` + `.venv/bin` on PATH for the GDN JIT.

## Status snapshot
**Feasibility gate CLOSED.** page_bubble / virtual_split / sub_block_allocator /
decision_cost (audited ×4) / cuda_graph (GPU probe) all ✅; prefix_cache
reclassified → `1_allocator` (#94). Next: build **`1_allocator`** (#97), which
pulls in the **L2 cost model** (#98) for `2_cost_reclaim`. Carried follow-ups:
decision-structure = eager-delete `IndexedHeap`; real-engine maintenance
profiling (#100); L1 lazy-heap fix (#99); prefix-cache verified on impl (#94).
No implementation started yet.
