# cuda_graph — RESULTS

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

One GQA sequence (8 q-heads / 2 kv-heads / head_dim 128), `ksize=32`, paged KV
cache of 512 physical blocks. Several **disjoint** physical block-sets are
pre-filled with the **same logical KV** (1 contiguous + 5 random scatterings).

1. Eager reference = `flash_attn_varlen_func` with the contiguous block-table.
2. **Capture a CUDA graph ONCE** wrapping the kernel, reading a *persistent*
   block-table tensor.
3. **Replay** while overwriting that tensor with each scattering (no recapture).
4. **Control:** one replay with the block-table pointing at **different** KV —
   its output **must differ** from the reference, else "match" would be
   trivially true (this rules out "the kernel ignores the block-table").

Run for **prefill** (S_q=S_kv=200, causal) and **decode** (S_q=1, S_kv=200).

## Results

| mode | scatter replays vs ref | control diff (≠ref) | replay fault | recapture |
|---|---|---|---|---|
| prefill | 5/5 **0.0** (bit-identical) | 4.09 ✅ differs | none | none (captured once) |
| decode  | 5/5 **0.0** (bit-identical) | 0.57 ✅ differs | none | none (captured once) |

- **Bit-identical** (max abs diff 0.0) across every scattering: same logical KV
  in scattered physical blocks ⇒ same key order ⇒ same math. Stronger than
  `virtual_split`'s "numerically equivalent" because here only the *physical
  address* changes, not the reduction order.
- The **control differs** (4.09 prefill / 0.57 decode) ⇒ the captured graph
  genuinely **re-reads the live block-table** each replay; the 0.0 matches are
  not an artifact of the kernel ignoring it.
- **0 faults, captured once** across 7 replays/mode.

## Conclusion

The captured graph treats the block-table as live input data; **scattered
sub-block ids are just different data**. No contiguity assumption is baked
anywhere downstream of the block-table read. The design's sub-block allocator
needs **no kernel change and no eager-mode fallback** for CUDA graphs. The
design-level showstopper risk is retired.

## Scope / caveats

- Real GPU, real `flash_attn_varlen_func` (the prod kernel), but a **standalone
  call**, not the full engine + allocator. It proves the *kernel + captured
  graph* tolerate scattered block-tables; the engine integration (building such
  block-tables from the two-level allocator, persistent-buffer plumbing) is
  exercised in `1_allocator`.
- Single sequence. The block-table is read per-row (`block_table[req]`), so
  multi-sequence batches use the identical read path; not separately swept here.
- `fa_version` left at the library default (the version `virtual_split` runs).
