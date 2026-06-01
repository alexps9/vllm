# decision_cost — RESULTS

**PASS, with one required follow-up (heap compaction).** On a **realistic
correlated workload**, the per-step "cheapest page to vacate for mamba"
decision is **incremental** (structurally O(log P) — flat across a 256× pool-
size range; **71–157× cheaper** than the O(P) re-walk it replaces),
**correct** (0 violations, never hands out a mamba page), and costs
**~2.4× LRU per op**. The lazy-delete heap **requires compaction** to bound
memory (600–900× bloat without it — a defect inherited from production
`LPBPriorityQueue`, task #99); compaction bounds it to ≤8× and is exactly the
design's "steady-state rebalance off the hot path."

Reproduce: `.venv/bin/python microbench.py` (pure-CPU, no GPU/vLLM import).
Raw: [`runs/microbench.out`](runs/microbench.out).

## History — this is v3, after TWO adversarial audits

Each audit caught the campaign's recurring failure mode (testing the wrong
thing) and changed the result. v1/v2 numbers are retired.

| version | what it tested | audit verdict |
|---|---|---|
| v1 | page events injected for free; per-op `perf_counter_ns` timing | timing-inflated (2.5–3.2×); bloat dismissed untested |
| v2 | independent random sub-block slots; heap of **reclaimable-only** pages | **didn't match reality**: independent slots → 99.6% partial pages (real pages are bimodal); reclaimable pages are rare under pressure, so a reclaimable-only heap is ~empty and the decision is trivial |
| **v3** | **correlated** whole-sequence alloc/free (bimodal pages); heap ranks **all** pages by vacate-cost | this doc |

The v2→v3 fix is the important one: under KV pressure almost no page is fully
free (measured below), so "cheapest *reclaimable* page" is a trivial empty
query. The real decision (design.md: "free the cheapest page — evict the
attention whose prefixes are cheapest to recompute") ranks **every** attention
page by **vacate-cost** (0 = free, Σ recompute = cached, + a large preempt
penalty = live). That heap holds ~all P pages under pressure, so
incremental-vs-rewalk genuinely matters — this is the regime v3 measures.

## What was tested (v3)

Workload reuses the **correlated** lifecycle of `sub_block_allocator/
fuzz_refcount.py` (whole-sequence alloc of 1..3K blocks with packing bias,
shared-prefix touch with ref_cnt>1, whole-sequence free), with incremental
slab bookkeeping. It emits a trace of per-page vacate-cost changes; the two
decision structures **replay** it under one timer (batch timing):
- **incremental** = lazy-delete min-heap of `(vacate_cost, page)` over all
  in-pool pages (exact pattern of vLLM's audited `LPBPriorityQueue`);
- **LRU baseline** = recency `OrderedDict` (move-to-back on access, vacate =
  front) — the cost-blind alternative.

## Results

### Workload realism — pages are bimodal; reclaimable pages are RARE  ✅
Occupancy sampled across the run (200k ops):

| P | free pages | full pages | partial pages | mean reclaimable |
|---:|---:|---:|---:|---:|
| 1,000 | 0.9% | 99.1% | ~0% | 9 |
| 4,000 | 3.2% | 96.8% | ~0% | 128 |

**Bimodal** (free + full ≫ partial), confirming correlated alloc/free — *not*
the independent-slot model's 99.6% partial pages. Under pressure reclaimable
(fully-free) pages are rare, which is exactly why the decision must rank **all**
pages by vacate-cost, not just the empty reclaimable set.

### (1) Query is incremental — O(log P), not the O(P) re-walk  ✅ (load-bearing)
**Structural complexity** (fixed update-rate, vary P — isolates the heap's big-O):

| P | 1,000 | 4,000 | 16,000 | 64,000 | 256,000 |
|---|---|---|---|---|---|
| peek ns | 244 | 241 | 240 | 241 | 239 |

**Flat across 256× P** ⇒ peek is O(log P), decisively **not** O(P).

**In-workload** query vs the O(P) ground-truth scan:

| P | incremental query | naive O(P) scan | naive / inc |
|---:|---:|---:|---:|
| 1,000  | 0.55 µs | 39 µs  | 71× |
| 4,000  | 1.20 µs | 157 µs | 131× |
| 16,000 | 4.68 µs | 734 µs | **157×** |

The in-workload incremental query rises with P here only because this
workload's *update volume* scales with P (larger pool = less pressure = more
churn) plus a one-time drain of superseded init entries — **not** a structural
O(P) (the fixed-churn table above proves that). Either way it stays **71–157×
cheaper** than the re-walk, which hits **0.73 ms** at P=16k. Compaction makes
the in-workload query marginally faster (e.g. 1.20 vs 1.48 µs at P=4k).

### (2) Per-op maintenance ~2.4× LRU (batch-timed)  ✅
All-page vacate-cost heap vs the recency `OrderedDict`, 3 seeds, P=4000:

| seed | inc ns/ev | LRU ns/ev | ratio |
|---:|---:|---:|---:|
| 0 | 595 | 255 | 2.33× |
| 1 | 650 | 287 | 2.27× |
| 2 | 681 | 268 | 2.54× |

**~2.4× LRU** — within the ≤3× ideal bar. Higher than v2's 1.3× because the
heap now ranks *all* P pages (the realistic, non-trivial regime), re-keying on
every sub-block transition that changes a page's vacate-cost.

### (3) Correctness — **zero** violations  ✅
Across all P, at every query (≈21k/run): incremental cheapest() returned a page
of the **true minimum vacate-cost** AND **never returned a mamba-owned page**:
**0** violations.

### (4) Heap memory — **requires compaction** (real defect, fixable)  ⚠→fixed
Direct `LazyHeap` stress (5M updates, 5000 live keys, occasional pops):

| mode | logical | physical (peak) | steady bloat | ceiling |
|---|---:|---:|---:|---:|
| **no compaction** (== production `LPBPriorityQueue` today) | 5,000 | 4.5 M | 600× | 900× |
| compaction (rebuild when physical > 8× logical) | 5,000 | 38 k | **~5×** | ≤8× |

A re-keyed page leaves a **stale interior leaf** reclaimed only if it bubbles to
the top; under update-heavy/pop-light load the physical heap grows ~linearly
with updates. **Inherited from the production heap** (`lpb_queue.py`, no
compaction — task #99). The fix is trivial (heapify live entries past a
threshold) and bounds it to ≤8×. **Must land with the implementation**
(`1_allocator`); L1's `LPBPriorityQueue` should get the same (task #99). This
compaction *is* the design's "steady-state rebalance off the hot path" —
amortized, off the per-decision path.

## Corrections to design.md (recorded)

- **"amortized O(1)"** → measured **O(log P)** (flat across 256× P at fixed
  churn); decisively not O(P).
- **"≤3× LRU per-op"** holds (~2.4× on the realistic all-page heap).
- **"steady-state rebalance off the hot path"** = the **compaction** in (4),
  now concretely identified and measured, not just asserted.
- **New, important:** under realistic KV pressure reclaimable (fully-free)
  pages are rare, so the decision ranges over **all pages by vacate-cost**
  (free → cached → live), which the asymmetry (live-page preempt penalty ≫
  cached recompute) orders correctly — matching the design's self-correcting
  cost argument. This is now the tested regime.

## Scope / caveats

- Standalone CPU model of the *decision structure*, not the vLLM scheduler
  integration. Proves it is incremental, correct, cheap, and (with compaction)
  memory-bounded under a realistic correlated workload. Integration must
  preserve these — part of impl phase `1_allocator`.
- The vacate-cost of a *live*-holding page uses a fixed preempt penalty; the
  real preempt/recompute cost comes from the L2 cost model (interlayer ⇄ L2
  coupling). decision_cost tests the *structure's* speed; the cost *values*
  and the reclaim *policy outcome* are `2_cost_reclaim`.
