# virtual_split — RESULTS

**PASS.** The attention kernel already runs at `kernel_block_size = 32`, far
below the 1056 *allocation* block — virtual block splitting is active, so the
1056 bubble is purely an allocator-granularity problem and the sub-page fix
needs **no kernel change**.

## Live probe (Qwen3.5-35B-A3B, align, TP=1, in-process)

`runs/probe_result.json`:

| field | value |
|---|---|
| manager block_size (allocation) | **1056** |
| kernel_block_size (kernel) | **32** |
| split ratio (1056/32) | 33 |
| attention backend | FlashAttention |
| splitting_active | **true** |
| deterministic greedy | true |

## Why 32 (and why the kernel *cannot* use 1056)

`select_common_block_size(1056, [FlashAttn])`
(`vllm/v1/worker/utils.py`): for a **hybrid model with fp32 SSM state**,
flash-attn's `get_supported_kernel_block_sizes` returns `[16, 32, 64]` (not
`MultipleOf(16)`) — the fp32-SSM NaN-propagation branch
(`flash_attn.py:77-94`, ref flash-attention#1974). 1056 ∉ {16,32,64};
1056 % 64 ≠ 0; **1056 % 32 = 0 → kernel_block_size = 32.**

So the very fp32-SSM state that *inflates* the allocation block to 1056 *also*
forbids the kernel from using blocks ≥128 — the attention kernel **cannot**
operate at the 1056 page even if asked. Sub-page (≤64-token) attention blocks
aren't merely kernel-compatible; they are the only thing the kernel will run.

## Implications for the design

- **"No kernel change" confirmed**: the kernel consumes KV at 32-token blocks
  today; an allocator that hands attention 32-token sub-blocks feeds the
  kernel exactly what it already uses.
- **Natural attention granularity floor = 32** → the bubble fix targets the
  block_size=32 counterfactual: **~1.3% waste** (vs 42.6% at 1056).
- Correctness sanity: greedy generation is deterministic and coherent at this
  layout. A bit-identical-vs-1056 comparison is **not applicable** — the
  kernel can't legally use 1056 here.

## Caveats / scope

This confirms the *kernel-transparency premise* (kernel runs fine-grained,
accepts sub-page blocks). It does **not** yet exercise sub-page blocks placed
at *arbitrary* physical offsets by a two-level allocator — that is gated on
`sub_block_allocator/` (phase 2) producing such layouts, with end-to-end
correctness ultimately in `the_win/`.

Boot note: the GDN kernel JIT-compiles via `ninja` (install `ninja`; put
`.venv/bin` on PATH). Run in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so
the capture fires.
