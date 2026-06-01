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

Numbered phase subdirs (sglang-style); each is a property that must strictly
pass:

| phase | property | status |
|---|---|---|
| [`0_page_bubble/`](0_page_bubble/) | the bubble exists (42.6% on 106 CC sessions) | ✅ done |
| `1_virtual_split` | attention kernel byte-exact at `kernel_block_size` ≪ page | planned |
| `2_sub_block_allocator` | two-level allocator: no overlap / no UAF on page flip | planned |
| `3_cost_reclaim` | **make-or-break (perf)** — cost-model page reclaim keeps recompute amplification / tail latency / attention starvation bounded | planned |
| `4_decision_cost` | per-step decision cheap (incremental, ≤~3× LRU); steady-state async | planned |
| `5_prefix_cache` | mixed-granularity prefix cache correct + finer reuse | planned |
| `6_cuda_graph` | sub-block block-table safe under captured graph | planned |
| `7_the_win` | waste → `kernel_block_size` counterfactual, no throughput regression | planned |

Implementation starts only if **all** of 1–7 pass. **interlayer depends on
the L2 cost model** (removed/redesigning) — the two efforts are coupled.
