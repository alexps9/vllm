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
on small-increment multi-turn agent traffic. Proof: [`0_feasibility/page_bubble/`](0_feasibility/page_bubble/).

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

Two tiers (full roadmap + status in [`PLAN.md`](PLAN.md)):

- **Feasibility gate** — cheap, self-contained pre-implementation checks in
  [`0_feasibility/`](0_feasibility/). Each has a falsifiable ideal pass bar and
  is **adversarially audited** before it counts. Must all pass before any vLLM
  integration.
- **Implementation phases** — `cost_reclaim` (the make-or-break) and the
  `e2e_win` are **not** paper-checkable: they depend on the real allocator +
  cost model + workload dynamics. They moved to the top-level `1_…`/`2_…`
  implementation phases (see PLAN.md) and are verified on a **minimal real
  implementation** — a paper sim that re-derives the policy is bug-prone and
  only proves the model (the deleted cost-reclaim sim had 2 bugs + tested the
  wrong axis).

**0 · page_bubble** — ✅ done. Bubble = 42.6% workload-weighted KV waste
(106 real CC sessions).

**1 · virtual_split** — ✅ **done.** The attention kernel runs at
`kernel_block_size = 32 ≪ 1056` (splitting active; it *cannot* use 1056 here —
fp32-SSM forces flash-attn to `[16,32,64]`), and computes **valid attention at
sub-page granularity with no kernel change**.
- *Test*: force `kernel_block_size = 16 vs 32` (both legal factors of 1056),
  greedy, identical prompts; compare.
- *Result*: numerically equivalent (logit Δ ~1e-3); 3/4 prompts bit-identical
  tokens, 1/4 flips at token 58/128 — benign fp reduction-order, not a bug.
- *Pass criterion (corrected)*: **valid attention at the target granularity,
  numerically equivalent** — NOT bit-identical. "Bit-identical (atol=rtol=0)"
  was misconceived: block size inherently changes fp reduction order, so the
  fix is **numerically-equivalent-but-not-bit-identical** vs the 1056 baseline
  — the same class of variation stock vLLM already has across `block_size`.
  **Accepted property of the fix** (see `0_feasibility/virtual_split/RESULTS.md`).

**2 · sub_block_allocator** — the two-level allocator is memory-safe.
- *Test*: CPU fuzz — ≥10⁶ randomized **and adversarial (max-scatter)**
  alloc/free/page-flip ops with invariant assertions (no two live ids alias
  bytes; ref-counts exact; a page is mamba-usable iff all its sub-blocks free).
- *Pass (ideal)*: **zero** invariant violations; every fully-freed page
  returned to the pool.

**decision_cost** — ✅ **done** (audited ×2). The per-step decision is cheap.
- *Test*: microbench the incremental "cheapest page **to vacate**" heap on a
  **correlated** workload (mirrors `sub_block_allocator/fuzz_refcount.py`),
  ranking **all** attention pages by vacate-cost, vs the O(P) re-walk + an LRU
  recency baseline.
- *Result*: chosen structure = **`IndexedHeap` (eager-delete)**: query **140 ns**
  (O(1) peek), structurally **O(log P)** (flat across 256× pool size),
  **bounded ~7 µs tail** (vs the lazy-delete heap's **~50 ms** O(P) spikes), **no
  bloat**, **0** correctness violations (never returns a mamba page). Per-op
  maintenance ~1.2 µs (~4× LRU).
- *Pass criterion (corrected)*: the literal "≤3× LRU per-op" was a proxy and is
  superseded by the faithful test — **absolute per-decision cost ≪ scheduler-step
  budget AND bounded worst case.** This is pure-CPU metadata (no syscall to
  overlap); ~1.2 µs maintenance × tens–hundreds events/step = tens–hundreds µs
  vs a ~10–50 ms forward pass (~0.1–1%, negligible). IndexedHeap passes (µs ≪ ms,
  bounded tail); the lazy-delete heap **fails** (50 ms single-peek spike >
  a whole step). The ~4× per-op is irrelevant at this magnitude.
- *Note*: with eager delete there is **no** "steady-state rebalance" needed —
  the heap is always compact (the lazy heap would need compaction AND still
  couldn't bound the tail; that's task #99 for L1's `LPBPriorityQueue`).
  Remaining for `1_allocator`: confirm per-step CPU budget on real HW; the
  cheapest page often being fully-live (→ preempt) is `2_cost_reclaim` policy.
  See `0_feasibility/decision_cost/RESULTS.md`.

**prefix_cache** — **reclassified → `1_allocator` (#94).** No faithful pre-impl
check: "hit rounds to `kernel_block_size` not 1056" is trivially true by setting
the hash chunk/alignment to `ksize` (`kv_cache_utils.py:645-688`,
`single_type_kv_cache_manager.py:483-528`); the real risk (collision handling,
ref-counting of shared sub-blocks on a co-owned page, eviction lifecycle) is the
cache machinery × the two-level allocator — verify on the real impl. *Pass
(ideal, on impl)*: zero correctness violations; hit length rounds to `ksize`.

**cuda_graph** — ✅ **done** (GPU probe). The **scattered** sub-block
block-table is safe under capture/replay. Code read: the block-table is a
per-step-written *persistent input* tensor (same address across replays,
`block_table.py:140-145`), read data-driven by the kernel (no fixed `N*ratio`
contiguity assumption, `block_table.py:226-288`).
- *Test*: scattered block-table (non-contiguous ids) vs a contiguous reference
  holding the same logical KV; real `flash_attn_varlen_func` under CUDA-graph
  **capture once → replay with changed block-table values**; + a control where
  the block-table points at *different* KV.
- *Result*: **0 faults, no recapture, bit-identical** to the contiguous ref
  across 5 scatterings × {prefill, decode}; control differs (graph re-reads the
  live table). ⇒ scattered ids are "just different data": **no kernel change,
  no eager-mode fallback**. See `0_feasibility/cuda_graph/RESULTS.md`.

**Implementation-phase checks** (top-level `1_…`/`2_…`, on a minimal real
impl — see [`PLAN.md`](PLAN.md)): **cost_reclaim** (mamba starvation /
recompute amplification / tail bounded, ≥ stock) and **e2e_win** (waste →
~1% counterfactual, no throughput regression). Both depend on the **L2 cost
model** (removed / redesigning) — interlayer ⇄ L2 are coupled.
