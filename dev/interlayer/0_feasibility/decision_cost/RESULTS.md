# decision_cost — RESULTS

> ℹ️ **interlayer CLOSED — not pursued.** This check passed (feasibility was never the blocker); the effort was dropped for *value* reasons. See [`../../POSTMORTEM.md`](../../POSTMORTEM.md).

**PASS (audited ×4). Chosen structure: `IndexedHeap` (eager-delete).** On a
realistic correlated workload, the per-step "cheapest page to vacate for mamba"
decision is **incremental** (query **O(1) peek, ~140 ns**, flat across 256× pool
size), **correct** (property test 360k ops × 30 seeds + workload, 0 violations,
never returns a mamba page), with a **bounded sub-µs worst case** (the lazy-delete
heap instead has **real O(P) spikes growing to ~70 ms**) and **no memory bloat**
(1.0× vs lazy 600–900×). Per-op maintenance is **O(log P), ~1–3 µs (~4× LRU)** —
over the literal ≤3× bar, but the 4× is *mostly* hand-rolled-Python-vs-C constant
(algorithmic gap only ~1.3–1.7×), and µs ≪ a 10–50 ms step makes it
**operationally irrelevant** (see "decision criterion"). The production
`LPBPriorityQueue` is lazy-delete and carries the spike + bloat defect — task #99.

A steelman (audit-4): cheaply capping the lazy heap's trim to bound its tail
produces **31% wrong answers** — you cannot get bounded-tail AND correctness from
the lazy heap, so the eager-delete per-op cost is warranted.

Reproduce: `.venv/bin/python microbench.py` (pure-CPU, no GPU/vLLM import).
Raw: [`runs/microbench.out`](runs/microbench.out).

## History — v4, after FOUR adversarial audits (each changed or sharpened the result)

| version | tested | audit found |
|---|---|---|
| v1 | page-events, per-op timing | timer-inflated ratio; bloat dismissed untested |
| v2 | independent random slots; reclaimable-only heap | unrealistic dynamics (real pages bimodal); reclaimable heap trivially empty under pressure |
| v3 | correlated workload; all-page vacate-cost lazy heap | **lazy-delete heap has unbounded O(P) peek spikes (~50 ms), hidden by reporting the mean** |
| **v4** | + `IndexedHeap` (eager delete); p50/p99/MAX | audit-4: decision JUSTIFIED; fixed reporting — maintenance is O(log P) not flat (F1); 140 ns is ~½ timer floor & idx MAX is jitter not peek (F2); 4× is mostly Python-vs-C, algorithmic gap ~1.3–1.7× (F3); added permanent property test (F4/F5) |

That this took 4 versions / 4 audits is itself the finding: a subtle structure
under churn has real tradeoffs that first-cut tests miss. The audit gate is
load-bearing here, not decorative.

## The decision criterion (corrected)

The design's "≤3× LRU per-op" was a **proxy** for "each decision is cheap enough
not to perturb the scheduler." The faithful criterion is:

> **absolute per-decision cost ≪ scheduler-step budget, AND bounded worst case.**

This is pure-CPU metadata work (design.md: "metadata only — no syscall to hide …
the lever is cheap incremental decisions, not stream overlap"). It is **not
overlapped and does not need to be** — it just has to be small against a forward
pass:
- query (gates admission): **140 ns** — negligible.
- maintenance (re-key a page's cost on each sub-block alloc/free): **~1.2 µs**
  (~0.9 µs over LRU). At an estimated tens–hundreds of such events per scheduler
  step → tens–hundreds of µs/step, vs a **~10–50 ms** forward pass = ~0.1–1%.
  Negligible, no waiting, no overlap required.
- the thing that WOULD stall a step is a **50 ms** single-peek spike (> a whole
  forward pass) — which is exactly the lazy-delete failure mode the chosen
  structure removes.

So `IndexedHeap` **passes the real criterion** (µs ≪ ms, bounded ~7 µs tail);
`LazyHeap` **fails it** (50 ms tail). The 4× per-op is the wrong thing to
optimize — magnitudes decide it. (Part of the 4× is hand-rolled Python vs C
`heapq`; irrelevant given the absolute size.)

## Results (canonical run, P up to 16k = realistic single-engine pages)

### Workload realism — bimodal pages; reclaimable pages rare
| P | free | full | partial | mean reclaimable |
|---:|---:|---:|---:|---:|
| 1,000 | 0.9% | 99.1% | ~0% | 9 |
| 4,000 | 3.2% | 96.8% | ~0% | 128 |

Correlated alloc/free ⇒ pages are bimodal (not the independent-slot model's
99.6% partial). Under pressure fully-free pages are rare ⇒ the decision must
rank **all** pages by vacate-cost (free → cached → live-preempt-penalty), which
the cost asymmetry orders correctly.

### (0) Correctness — property test (the backbone, audit-4 F5)
The hand-rolled IndexedHeap is checked vs a `dict`+`min()` reference with FULL
invariants (peek==reference-min; pos-map bijective with the heap array; heap
order parent≤children; contents match) **after every op**, over **360k ops × 30
seeds** with small keyspaces (heavy ties), **54k interior removes**, remove-min,
remove-then-readd, and drain-to-empty: **0 violations**. The correlated workload
itself drives interior removes (the hard sift path) with 0 violations too.

### (1) Query latency vs P — LazyHeap vs IndexedHeap vs naive O(P)
p50 / p99 / **MAX** (the *mean* hides the lazy O(P) spikes):

| P | lazy p50/p99/MAX | idx p50/p99/MAX | naive O(P) mean |
|---:|---|---|---:|
| 1,000  | 260 ns / 4.2 µs / **1.6 ms** | 140 ns / 280 ns / 4.8 µs* | 41 µs |
| 4,000  | 310 ns / 7.7 µs / **5.7 ms** | 160 ns / 400 ns / 11 µs* | 172 µs |
| 16,000 | 560 ns / 14 µs / **70.5 ms** | 260 ns / 570 ns / 51 µs* | 751 µs |

LazyHeap MAX is a **real O(P) trim spike** — scales 1.6→70 ms with P. IndexedHeap
p50/p99 are tight (140–570 ns); its **MAX\* is scheduler JITTER, not peek cost**
(audit-4 F2): batched worst per-call is ~125 ns and the bare `perf_counter_ns()`
floor on this box is ~56 ns, so the real peek is sub-µs (~68 ns net). Fixed-churn
probe: peek p50 **flat ~140 ns across 256× P** ⇒ O(1).

### (2) Per-op maintenance — IndexedHeap, and how much is algorithm vs Python
Batch-timed, P=4000, 3 seeds: inc ~1.2–1.3 µs/event, LRU ~0.27–0.34 µs/event →
**3.74 / 4.04 / 4.37× LRU** (load-dependent; ~2.9× at lighter load).

Bare-op decomposition (audit-4 F3 — *how much of the 4× is algorithm vs the
hand-rolled-Python-sift-vs-C constant*): IndexedHeap.update 0.78 µs, LazyHeap
`heapq.heappush` 0.62 µs, OrderedDict.move 0.19 µs ⇒ **idx/lazy = 1.3–1.7×
(algorithmic)**, idx/OrderedDict = **4.1× (mostly Python-sift vs C primitive)**.
So the 4× is *predominantly* implementation constant, not algorithm — a
C-backed indexed heap would likely beat the lazy heap on per-op too.

Maintenance is **O(log P)**, not flat (audit-4 F1): ~1.0 µs @1k → ~1.1 µs @16k →
~2.9 µs @64k → ~3.2 µs @256k. Still **≪ a 10–50 ms step** at every size.

### (3) Correctness in the workload — **0** violations
Across all P and ~21k queries/run: IndexedHeap (and LazyHeap) returned the true
min-vacate-cost page and **never a mamba-owned page**. 0 violations. (mamba-take
+ near-empty-pool paths are lightly hit in the workload — audit-4 F4 — but
hammered by the (0) property test.)

### (4) Memory — IndexedHeap has no bloat by construction
5M updates, 5000 keys: lazy no-compaction **600–900×**; lazy compaction(8) ~5×
steady / ≤8× ceiling (but **does not** bound the peek tail); **IndexedHeap 1.0×**
(physical == logical, no stale entries ever).

## Corrections to design.md (recorded)

- "amortized O(1)" → query is O(1) peek with the IndexedHeap (was O(log P) +
  unbounded trim with the lazy heap).
- "≤3× LRU per-op" → superseded by **absolute µs ≪ step budget + bounded tail**;
  IndexedHeap is ~4× LRU but passes the real criterion.
- "steady-state rebalance off the hot path" → with eager delete there is **no**
  rebalance: the structure is always compact. (The lazy heap would need
  compaction + still couldn't bound the tail.)
- Under realistic pressure fully-free pages are rare, so the decision ranks
  **all** pages by vacate-cost — the tested regime.

## Scope — what remains for `1_allocator` (narrow)

The **structure is decided: IndexedHeap.** Open items needing the real engine
(not another microbench):
1. **Profile whether the maintenance lands on the critical path (task #100).**
   The µs number only *counts* if it increases end-to-end step wall-clock —
   e.g. a step that was ~10 ms now waits ~40 ms. If this CPU work is already
   hidden inside the GPU-forward **overlap window** and stays within it after
   adding the heap maintenance, the absolute µs are **zero-impact** regardless
   of their size. Measure step latency / TTFT / TPOT / throughput vs stock
   under KV-bound load — *did it push out of the overlap window?* — not the
   raw per-op µs. Only optimize (C-level heap, batch re-keys) if it did.
2. The *outcome* that the cheapest page is often a fully-live page (needs
   preempt/defer) is the **cost_reclaim policy** (`2_cost_reclaim`) + L2 cost
   model — decision_cost only proves the *structure* finds the min cheaply.
3. Apply eager-delete (or at least compaction) to L1's `LPBPriorityQueue`
   (task #99) — same latent spike/bloat there.
