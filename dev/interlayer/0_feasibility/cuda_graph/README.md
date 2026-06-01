# cuda_graph — feasibility check (phase 0)

**PASS.** A scattered (non-contiguous) sub-block block-table replays correctly
under a captured CUDA graph on the real flash-attn kernel: zero faults, no
recapture, bit-identical to the contiguous reference, for prefill and decode.
A control (block-table → different KV) differs, proving the graph re-reads the
live table. So scattered sub-block ids are "just different data" — no kernel
change, no eager-mode fallback. Showstopper risk retired.

- `probe.py` — real GPU, real `flash_attn_varlen_func`, capture-once /
  replay-with-changed-block-table + control.
- `RESULTS.md` — verdict + method. `runs/probe.out` — captured run.

Full spec: [`../../design.md`](../../design.md) verification gate.
