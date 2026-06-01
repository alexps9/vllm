# interlayer (vLLM) — design

> Eliminate the **page-size bubble** in vLLM hybrid models (attention KV +
> mamba recurrent state). This doc states only the *target ideal
> architecture* and the verification gate it must pass before any
> implementation. Counterpart to sglang's `dev/interlayer/`, but vLLM's
> bubble — and its fix — are different.

## The problem

A hybrid model has two state types with opposite natural shapes:

- **Mamba state** — one **big, indivisible, contiguous** blob per request
  (fp32 SSM). The recurrent kernel reads/writes it whole → must be contiguous.
- **Attention KV** — **many small** blocks (tokens-grained), one per span.

vLLM serves both from **one fungible block pool**, which requires every block
to be the same byte size = `max(attention_page, mamba_page)`. The mamba page
is larger, so vLLM inflates the **allocation** block_size to match it — on
Qwen3.5-35B-A3B that is **1056 tokens**. Attention KV is then *allocated* in
1056-token blocks that real requests fill only fractionally → blocks all
allocated, most of their slots empty. Measured: **42.6% workload-weighted KV
waste on 106 real CC sessions** (vs 0.69% at the natural granularity); worst
on small-increment multi-turn agent traffic. Proof: [`0_page_bubble/`](0_page_bubble/).

## The enabling fact (and its precise limit)

The bubble is **purely an allocator-granularity problem; the kernel is
already fine-grained.** vLLM has **virtual block splitting**
(`prepare_kernel_block_sizes` / `select_common_block_size`; `gpu_model_runner`
`num_blocks_per_kv_block = block_size // kernel_block_size`): the attention
kernel runs at a small backend-native `kernel_block_size` (a factor of 1056,
e.g. 16/32). So the fix needs **no kernel change** — that part is genuinely
free.

**But virtual splitting is NOT a sub-block allocator.** It is a fixed
arithmetic fan-out of a *whole-page* block id: `block_table.py:196-201` maps
manager block `N` → kernel blocks `N*ratio + [0..ratio-1]`, always contiguous,
always derived from `N`. The atomic unit of **allocation, freeing,
ref-counting, and prefix-caching is still the whole 1056 page**
(`FullAttentionManager` allocates `cdiv(tokens, 1056)` whole blocks;
`KVCacheBlock` has one `ref_cnt`; `FreeKVCacheBlockQueue` is a whole-block
list; prefix cache maps `hash → one whole KVCacheBlock`, `block_pool.py`). So
attention-at-sub-page-granularity is **net-new structure**, not something the
existing splitting hands us. The kernel is free; the allocation layer is
greenfield.

## Ideal architecture

**Decouple attention's allocation granularity from the mamba page, over one
shared physical pool, leveraging the existing virtual block splitting.**

- **One physical pool**, page = the mamba state size (the indivisible unit).
- **Mamba** allocates **whole pages** — 1 page = 1 state. Contiguous; mamba
  spec and kernel unchanged.
- **Attention** allocates at its natural **`kernel_block_size`** (sub-page)
  granularity, packing `1056/ksize` sub-blocks into each physical page. The
  attention kernel already consumes KV at this granularity, so **no kernel
  change**. The attention block table and prefix cache move to sub-block
  granularity — which also removes the coarse 1056-token prefix-cache
  rounding.
- A physical page becomes **mamba-usable again only when all its attention
  sub-blocks are free**. The effective KV↔mamba split stays fully dynamic
  (one pool), with no fixed boot ratio.

**Net-new structures this requires** (none of it reuses existing machinery —
see "enabling fact"): a two-level allocator (physical page ↔ its sub-blocks),
**per-sub-block ref-counting** (today `KVCacheBlock.ref_cnt` is per whole
page), a **sub-page-aware free queue**, and **sub-page prefix-cache entries**
(today the cache maps a hash to one whole block). It extends vLLM's
whole-page / append-only block model, so phases 2 and 4 are real engineering,
not config.

Properties this preserves: **no VMM / page-remapping** (vLLM has none), **no
fixed split**, **no change to model numerics** (SSM stays fp32), **no kernel
change**, mamba contiguity intact. The bubble drops from rounding-to-1056 to
rounding-to-`kernel_block_size` (~`ksize/2` tokens per sequence, e.g. ~8–16
instead of ~528).

## Target regime: prefix-caching (`align`) — resolved

We target the production agent setting: **prefix caching on**, which forces
`mamba_cache_mode="align"`. Confirmed in code + live:

- The **attention 1056 bubble is identical in `align`** (attention still
  allocates `cdiv(tokens,1056)` mostly-empty blocks; measured live).
- In `align`, mamba does **not** hoard `cdiv(L,1056)` snapshots — it keeps a
  **rolling 1–2 live state pages** per request (state copied forward at each
  block boundary, the old block freed → `null_block`;
  `single_type_kv_cache_manager.py:916-933, 988-1001`), plus **cached
  snapshot pages** shared via prefix cache. Each mamba block is **one whole
  physical page**.
- Therefore mamba **never needs multi-page contiguous runs** — in *either*
  mode it needs a **whole single page** at a time (1/req in `none`; rolling
  few + snapshots in `align`). So the hard property (below) is "≥1 fully-free
  page when mamba needs one," not "N contiguous pages."

Net: the fix applies **unchanged** in `align`, and by freeing whole pages it
*helps* mamba claim its rolling/snapshot pages. (Replace "indivisible big
page" everywhere with "whole single page".)

## Mamba page availability — a cost decision

When mamba needs a whole page and none is fully free, this is **not a "ran
out" failure — it is a cost decision**, the same one L1 (recompute cost) and
L2 (admitter) already model: free the **cheapest** page — evict the attention
sub-blocks whose prefixes are cheapest to recompute — or, if everything is too
expensive, preempt/defer via vLLM's existing mechanism (preempting a running
request frees both its attention and its mamba, so no deadlock). The
**asymmetry makes this self-correcting**: mamba state is expensive to lose
(recompute = the whole sequence), attention prefixes are cheap (per-prefix),
so the cost model naturally protects mamba and evicts attention — with no
special rule. (`block_pool.py:48-52` keeps block ids immutable / append-only,
so relocation/compaction is not an option — reclaim is by eviction, not
movement. Packing bias — fill partially-used attention pages before opening
fresh ones — keeps fully-free pages plentiful and makes each eviction a clean
coherent page; it is an optimization for the cost decision.)

This makes **interlayer a consumer of the same cost model as L2** (the
admitter / recompute-cost layer removed in 2026-05 and slated for redesign):
**an accurate cost model is a prerequisite — the two efforts are coupled.**

The make-or-break is therefore **performance, not correctness**: under
KV-bound real + adversarial load, does cost-model-driven page reclaim keep
**recompute amplification, tail latency, and attention-side starvation
bounded**?

## Cost-model decision-layer performance

The per-step decision (cost-rank the cheapest page to free) sits on the
scheduler hot path because it gates admission. The async/overlap lever here is
**different from sglang's**: sglang ran the actuator on a worker thread to hide
`cuMem` *syscall* latency from the decode stream. vLLM's page reclaim is
**metadata only (free block ids) — no syscall to hide**, and the recompute
price is paid later on the normal prefill path (a throughput cost the model
already counts, not a stall). So the lever is **cheap, incremental decisions,
not stream overlap**: maintain the "cheapest page to free" incrementally (like
L1's LPB heap) rather than re-walking structures each time; only steady-state
rebalance may run async. This per-decision cost must be bounded (echoing
verify/6's ≤~3× LRU target).

## Verification gate

Numbered subdirs, sglang-style. Each phase has a **falsifiable pass bar set
at the ideal level** (strictly-better-or-equal; zero violations; waste → the
counterfactual floor). Bars tagged *(calibrate)* are first-run-tunable; the
rest are hard lines. **Implementation starts only if all of 1–7 pass.**

**0 · page_bubble** — ✅ done. Bubble = 42.6% workload-weighted KV waste
(106 real CC sessions).

**1 · virtual_split** — the attention kernel is byte-exact at
`kernel_block_size` ≪ page.
- *Test*: Qwen3.5-35B-A3B (align); run attention over the sub-divided
  1056-page layout vs the reference full-page path, identical inputs.
- *Pass (ideal)*: outputs **bit-identical** (atol = rtol = 0).

**2 · sub_block_allocator** — the two-level allocator is memory-safe.
- *Test*: CPU fuzz — ≥10⁶ randomized **and adversarial (max-scatter)**
  alloc/free/page-flip ops with invariant assertions (no two live ids alias
  bytes; ref-counts exact; a page is mamba-usable iff all its sub-blocks free).
- *Pass (ideal)*: **zero** invariant violations; every fully-freed page
  returned to the pool.

**3 · cost_reclaim** — *the make-or-break (performance)*. Cost-model-driven
page reclaim stays bounded.
- *Test*: KV-bound real load (W2, high conc/util) **+** adversarial
  (max attention sub-block scatter × mamba-heavy interleaving), vs stock vLLM.
- *Pass (ideal)*: mamba page starvation = **0** (never stalls); recompute
  amplification **≤ 0%** vs stock *(calibrate)*; **p99 TTFT ≤ stock**
  *(calibrate)*; no sustained attention starvation (every request progresses
  within bounded steps). I.e. **strictly ≥ stock on every axis.**

**4 · decision_cost** — the per-step decision is cheap.
- *Test*: microbench the incremental "cheapest page to free" structure under
  realistic alloc/free churn (verify/6-style), vs the LRU free-queue.
- *Pass (ideal)*: **≤ 3× LRU** per-op (verify/6 precedent), amortized O(1);
  steady-state rebalance off the hot path.

**5 · prefix_cache** — mixed-granularity cache is correct and finer.
- *Test*: hybrid requests sharing prefixes at sub-page boundaries.
- *Pass (ideal)*: **zero** correctness violations (cached bytes = recompute,
  no cross-request contamination); hit length rounds to `kernel_block_size`,
  not 1056 — the 1056 rounding is **eliminated**.

**6 · cuda_graph** — the sub-block block-table is safe under capture/replay.
- *Test*: capture a CUDA graph with the sub-block layout, replay across
  alloc/free/page-flip events.
- *Pass (ideal)*: **zero** replay faults; **no recapture** needed.

**7 · the_win** — the bubble is eliminated with no regression.
- *Test*: KV-bound agent load, n=3, fix vs stock.
- *Pass (ideal)*: waste → the `kernel_block_size` counterfactual floor (~1%
  at ksize=32, vs 42.6%); **throughput ≥ stock** with a real gain under KV
  pressure (target throughput / hit-rate uplift) *(calibrate)*.

interlayer depends on the L2 cost model (removed / redesigning) — coupled.
