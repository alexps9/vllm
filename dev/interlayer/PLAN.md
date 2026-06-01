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
| [`decision_cost`](0_feasibility/decision_cost/) | ⬜ todo (#93) | per-step "cheapest page to free" decision is cheap (incremental ≤~3× LRU) | per-op cost > 3× LRU; steady-state not async-able |
| [`prefix_cache`](0_feasibility/prefix_cache/) | ⬜ todo (#94) ⚠ | mixed-granularity prefix cache correct + finer reuse | correctness violation / hit doesn't round to ksize. **⚠ may need minimal impl — revisit (see below).** |
| [`cuda_graph`](0_feasibility/cuda_graph/) | ⬜ todo (#95) ⚠ | sub-block block-table safe under captured-graph replay | replay fault / recapture. **⚠ may need minimal impl — revisit.** |

**Gate**: all six pass (audited). `decision_cost` is purely self-contained.
`prefix_cache` and `cuda_graph` test properties of the *real* block-table /
prefix-cache changes, so they likely need the Phase-1 allocator to exist —
flagged for reclassification to implementation if a faithful pre-impl check
isn't possible.

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
3/6 feasibility checks done (all audited). decision_cost / prefix_cache /
cuda_graph remain. No implementation started — gate not closed.
