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

Each phase is a property that must **strictly pass** before implementation
begins. (Numbered subdirs, sglang-style.)

| phase | property to prove | how |
|---|---|---|
| [`0_page_bubble/`](0_page_bubble/) | the bubble exists (42.6%) | ✅ done |
| `1_virtual_split` | attention kernel is byte-exact at `kernel_block_size` ≪ page (pin the live value) | GPU correctness probe |
| `2_sub_block_allocator` | two-level allocator: attention sub-block alloc/free + page mamba↔attention flip, **no byte overlap, no use-after-free**, under concurrent alloc/free | CPU unit tests + invariants |
| `3_cost_reclaim` | **the make-or-break (performance)** — cost-model-driven page reclaim keeps **recompute amplification, tail latency, and attention starvation bounded** under KV-bound real + adversarial load (depends on an accurate cost model) | simulation + e2e stress |
| `4_decision_cost` | per-step decision is cheap — incremental "cheapest page to free" structure, **bounded hot-path cost** (≤~3× LRU, verify/6-style); steady-state rebalance async | microbench |
| `5_prefix_cache` | mixed-granularity prefix cache is correct and reuses finer (≈`ksize` vs 1056) | unit + e2e hit comparison |
| `6_cuda_graph` | sub-block block-table shape change is safe under captured-graph replay | captured-graph replay |
| `7_the_win` | bubble waste drops to the `kernel_block_size` counterfactual with **no throughput regression** | n=3 e2e |

Implementation starts only if **all** of 1–7 pass.
