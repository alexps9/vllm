# cuda_graph — RESULTS

> ℹ️ **interlayer CLOSED — not pursued.** This check passed (feasibility was never the blocker); the effort was dropped for *value* reasons. See [`../../POSTMORTEM.md`](../../POSTMORTEM.md).

**PASS.** A **scattered** (arbitrary, non-contiguous) sub-block block-table is
safe under CUDA-graph capture/replay on the real flash-attn kernel: **zero
replay faults, no recapture**, and scattered output is **bit-identical** to the
contiguous reference. Confirmed on the same kernel `virtual_split` established
runs at `kernel_block_size=32`, for both **prefill** and **decode**.

Reproduce: `CUDA_VISIBLE_DEVICES=<free gpu> .venv/bin/python probe.py`.
Raw: [`runs/probe.out`](runs/probe.out).

## The risk

The two-level allocator makes an attention sequence's sub-blocks
**non-contiguous** (arbitrary physical ids), unlike the existing virtual-block-
splitting which fans a manager block to a *contiguous* run `N*ratio+[0..ratio)`.
Fear: a captured CUDA graph could bake in a contiguity assumption and
fault / mis-read when the block-table holds scattered ids — which would force
eager mode (a real perf loss), a design-level showstopper.

Code read (recorded in `design.md`) said no: the block-table is a per-step-
written **persistent input** tensor (same address across replays,
`block_table.py:140-145`), read **data-driven** by the slot kernel (no `N*ratio`
assumption, `block_table.py:226-288`). This probe confirms it on the real GPU.

## Method (`probe.py`)

GQA (8 q-heads / 2 kv-heads / head_dim 128), `ksize=32`, paged KV cache of 2048
physical blocks. For each case, several **disjoint** physical block-sets are
pre-filled with the **same logical KV** per sequence (1 contiguous + 5 random
scatterings). The kernel is called with the **production CUDA-graph config** the
vLLM backend passes (`flash_attn.py:796-818`): **FA3** (`get_flash_attn_version`),
**`num_splits=32`** (`flash_attn_max_num_splits_for_cuda_graph`), and the FA3
**`scheduler_metadata`** AOT schedule.

1. Eager reference = `flash_attn_varlen_func` with the contiguous block-table.
2. **Capture a CUDA graph ONCE** (counter-verified) wrapping the kernel, reading
   a *persistent* block-table tensor.
3. **Replay** while overwriting that tensor with each scattering (no recapture).
4. **Control:** one replay with a row pointing at **different** KV — its output
   **must differ**, else "match" would be trivially true (rules out "the kernel
   ignores the block-table").

Cases: **prefill** (1×200), **decode** (1×1 over 200), **mixed batch**
(prefill + 2 decodes, **per-row** scatter), **long-context decode** (1 over 4096
→ split-KV genuinely partitions the sequence across many scattered blocks).

## Results

| case | scatter replays vs ref | control diff (≠ref) | fault | captures |
|---|---|---|---|---|
| prefill (1×200) | 5/5 **0.0** | 4.09 ✅ | none | 1 |
| decode (1×200) | 5/5 **0.0** | 0.57 ✅ | none | 1 |
| mixed batch (prefill+2 decode, per-row scatter) | 5/5 **0.0** | 4.31 ✅ | none | 1 |
| long decode (1×4096, split-KV) | 5/5 **0.0** | 0.14 ✅ | none | 1 |

- **Bit-identical** (max abs diff 0.0) across every scattering and every case:
  same logical KV in scattered physical blocks ⇒ same key order ⇒ same math.
  Stronger than `virtual_split`'s "numerically equivalent" (only the *physical
  address* changes, not reduction order).
- The **control differs** in all cases ⇒ the captured graph genuinely
  **re-reads the live block-table** each replay; the 0.0 matches are not the
  kernel ignoring it.
- **0 faults, captured exactly once** (measured counter) across 7 replays/case;
  multi-sequence **per-row** scatter and split-KV (`num_splits=32`) both hold.

## Conclusion

The captured graph treats the block-table as live input data; **scattered
sub-block ids are just different data**. No contiguity assumption is baked
anywhere downstream of the block-table read. The design's sub-block allocator
needs **no kernel change and no eager-mode fallback** for CUDA graphs. The
design-level showstopper risk is retired.

## Scope / caveats

- Real GPU, real `flash_attn_varlen_func` under the **prod CUDA-graph config**
  (FA3 + `num_splits=32` + `scheduler_metadata`), but a **standalone call**, not
  the full engine. It proves the *kernel + captured graph* tolerate scattered /
  per-row block-tables; the engine plumbing is exercised in `1_allocator`.
- **Single residual (integration-only, → `1_allocator`):** the kernel tolerates
  the scatter (proven here), but whether the two-level allocator writes the
  scattered kernel-block ids into the **exact persistent `input_block_tables`
  buffer the captured graph baked** (`block_table.py:140-145` /
  `get_dummy_block_tables`) — rather than allocating a fresh tensor — is an
  integration property only the real impl can confirm.

*History:* an audit (`a49bf77f`) found v1 of this probe omitted the prod args
(`num_splits` / `scheduler_metadata` / `fa_version`) and ran single-sequence; it
verified the conclusion was unchanged with them, and those + multi-seq + split-KV
are now folded into the probe above (all bit-identical).
