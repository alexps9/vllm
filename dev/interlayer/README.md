# dev/interlayer (vLLM)

vLLM-side **interlayer** work: the cross-pool / page-size **bubble** in
hybrid models (attention KV + mamba recurrent state). Counterpart to the
sglang repo's `dev/interlayer/`, but vLLM's bubble is a *page-size*
(internal-fragmentation) bubble, not sglang's fixed-split bubble — see
[`design.md`](design.md).

[`design.md`](design.md) = the **ideal architecture**. [`PLAN.md`](PLAN.md) =
the **roadmap + live status** (feasibility gate → implementation phases),
each task with a falsification criterion. Start there.

**We are pre-implementation.** The cheap, self-contained checks live under
**`0_feasibility/`** (each adversarially audited before it counts). Properties
that need the real allocator + cost model + workload (`cost_reclaim`,
`e2e_win`) are **not** paper-checkable — they moved to the top-level
implementation phases (`1_…`/`2_…`), verified on a minimal real impl. See
PLAN.md.

`0_feasibility/` checks:

| check | property | status |
|---|---|---|
| [`page_bubble/`](0_feasibility/page_bubble/) | the bubble exists (42.6% on 106 CC sessions) | ✅ done |
| [`virtual_split/`](0_feasibility/virtual_split/) | kernel runs at `ksize=32 ≪ 1056`, valid at sub-page granularity, no kernel change (numerically-equiv) | ✅ done |
| [`sub_block_allocator/`](0_feasibility/sub_block_allocator/) | two-level allocator memory-safe **with ref-counting + cached lifecycle** (0 violations) | ✅ done |
| [`decision_cost/`](0_feasibility/decision_cost/) | per-step decision cheap (incremental, ≤~3× LRU) | ⬜ todo |
| [`prefix_cache/`](0_feasibility/prefix_cache/) ⚠ | mixed-granularity prefix cache correct + finer reuse | ⬜ todo (may need impl) |
| [`cuda_graph/`](0_feasibility/cuda_graph/) ⚠ | sub-block block-table safe under captured graph | ⬜ todo (may need impl) |

**Implementation phases** (PLAN.md): `1_allocator` → `2_cost_reclaim`
(make-or-break) → `3_e2e_win`. **Depend on the L2 cost model** (redesigning) —
interlayer ⇄ L2 coupled.
