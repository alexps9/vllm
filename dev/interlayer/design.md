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

## The enabling fact

The bubble is **purely an allocator-granularity problem; the kernel is
already fine-grained.** vLLM has **virtual block splitting**
(`gpu_model_runner.py` `num_blocks_per_kv_block = block_size //
kernel_block_size`; `prepare_kernel_block_sizes` / `select_common_block_size`):
the attention kernel already runs at a small backend-native `kernel_block_size`
(a factor of 1056, e.g. 16/32), viewing each 1056 page as `1056/ksize` kernel
blocks. Only the **allocator / manager / prefix-cache** operate at 1056
(`FullAttentionManager` allocates `cdiv(tokens, 1056)` whole pages). So the fix
needs **no kernel change** — only the allocation layer.

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

Properties this preserves: **no VMM / page-remapping** (vLLM has none), **no
fixed split**, **no change to model numerics** (SSM stays fp32), **no kernel
change**, mamba contiguity intact. The bubble drops from rounding-to-1056 to
rounding-to-`kernel_block_size` (~`ksize/2` tokens per sequence, e.g. ~8–16
instead of ~528).

## The one hard property

**Mamba whole-page availability.** A page held even partially by live
attention sub-blocks cannot serve a (whole-page) mamba allocation. Under
attention-heavy or adversarial interleaving, scattered attention sub-blocks
could starve mamba of whole pages while sub-block space is plentiful. The
architecture must **bound this** — via a compaction step (relocate attention
sub-blocks to consolidate free pages) and/or a soft reservation policy — and
prove the bound holds under realistic and adversarial load. This is the
make-or-break property.

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
