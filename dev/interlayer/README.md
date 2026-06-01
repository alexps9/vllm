# dev/interlayer (vLLM)

vLLM-side **interlayer** work: the cross-pool / page-size **bubble** in
hybrid models (attention KV + mamba recurrent state). Counterpart to the
sglang repo's `dev/interlayer/`, but vLLM's bubble is a *page-size*
(internal-fragmentation) bubble, not sglang's fixed-split bubble — see
[`design.md`](design.md).

[`design.md`](design.md) states the **ideal architecture** (decouple
attention's allocation granularity from the mamba page over one shared pool,
using vLLM's existing virtual block splitting — no VMM, no fixed split, no
kernel change, no model-numerics change) and the **verification gate** that
must pass before any implementation.

**We are pre-implementation.** All of this is *verification* — proving the
design is feasible before any code is written. It therefore all lives under
**`0_feasibility/`** as subfolders (one per check). Top-level numbered dirs
(`1_…`, `2_…`) are **reserved for implementation components** (sglang-style),
and stay empty until the gate below passes.

`0_feasibility/` subfolders — each a property that must strictly pass:

| check | property | status |
|---|---|---|
| [`page_bubble/`](0_feasibility/page_bubble/) | the bubble exists (42.6% on 106 CC sessions) | ✅ done |
| [`virtual_split/`](0_feasibility/virtual_split/) | attention kernel byte-exact at `kernel_block_size` ≪ page | in progress |
| [`sub_block_allocator/`](0_feasibility/sub_block_allocator/) | two-level allocator: no overlap / no UAF on page flip | planned |
| [`cost_reclaim/`](0_feasibility/cost_reclaim/) | **make-or-break (perf)** — cost-model page reclaim keeps recompute amplification / tail latency / attention starvation bounded | planned |
| [`decision_cost/`](0_feasibility/decision_cost/) | per-step decision cheap (incremental, ≤~3× LRU); steady-state async | planned |
| [`prefix_cache/`](0_feasibility/prefix_cache/) | mixed-granularity prefix cache correct + finer reuse | planned |
| [`cuda_graph/`](0_feasibility/cuda_graph/) | sub-block block-table safe under captured graph | planned |
| [`the_win/`](0_feasibility/the_win/) | waste → `kernel_block_size` counterfactual, no throughput regression | planned |

Implementation starts only if **all** of 1–7 pass. **interlayer depends on
the L2 cost model** (removed/redesigning) — the two efforts are coupled.
