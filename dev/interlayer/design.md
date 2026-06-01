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

## The one hard property

**Mamba whole-page availability.** A page held even partially by live
attention sub-blocks cannot serve a (whole-page) mamba allocation. Under
attention-heavy or adversarial interleaving, scattered attention sub-blocks
could starve mamba of whole pages while sub-block space is plentiful.

The obvious mitigation — **compaction (relocate attention sub-blocks to
consolidate free pages) — is BLOCKED**: vLLM deliberately keeps block ids
immutable so block tables stay append-only (`block_pool.py:48-52`); moving a
sub-block would change its id and break that invariant across the scheduler.
So the bound must come from **placement, not relocation**: a packing bias
(fill partially-used attention pages before opening fresh ones, keeping whole
pages free for mamba) and/or a soft mamba reservation floor. Proving such a
placement policy bounds mamba starvation under realistic + adversarial load is
the make-or-break property.

## Verification gate

Each phase is a property that must **strictly pass** before implementation
begins. (Numbered subdirs, sglang-style.)

| phase | property to prove | how |
|---|---|---|
| [`0_page_bubble/`](0_page_bubble/) | the bubble exists (42.6%) | ✅ done |
| `1_virtual_split` | attention kernel is byte-exact at `kernel_block_size` ≪ page (pin the live value) | GPU correctness probe |
| `2_sub_block_allocator` | two-level allocator: attention sub-block alloc/free + page mamba↔attention flip, **no byte overlap, no use-after-free**, under concurrent alloc/free | CPU unit tests + invariants |
| `3_mamba_availability` | **the hard one** — attention sub-block scatter does not starve mamba of whole pages beyond an acceptable bound; compaction/reservation holds under realistic + adversarial load | simulation + e2e stress |
| `4_prefix_cache` | mixed-granularity prefix cache is correct and reuses finer (32 vs 1056) | unit + e2e hit comparison |
| `5_cuda_graph` | sub-block block-table shape change is safe under captured-graph replay | captured-graph replay |
| `6_the_win` | bubble waste drops to the `kernel_block_size` counterfactual with **no throughput regression** | n=3 e2e |

Implementation starts only if **all** of 1–6 pass.
