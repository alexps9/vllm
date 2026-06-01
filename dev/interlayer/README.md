# dev/interlayer (vLLM) — CLOSED

**This effort is closed — not pursued.** It explored eliminating the hybrid
page-size bubble (attention KV + mamba state) via a two-level sub-block
allocator. Investigation (incl. live measurement) showed it isn't worth
building on vLLM. **Read [`POSTMORTEM.md`](POSTMORTEM.md) — it is the record.**

One-line why: the memory bubble is small (~0.5% at 100k, ~5% at short
contexts); the bigger harm (the "42.6%" = prefix-cache recompute tail) is
mamba-bound and unfixable by this approach (`0_feasibility/page_bubble/
08_hybrid_architectural_blocker.md`); the cross-type cost decision collapses to
L1; and the real gap vs sglang (radix / exact-turn-end caching) is a vLLM-core
re-architecture, outside the HiMA framework and not symmetric with the sglang
work. The cost-model lever pays off on sglang's separate-pool architecture, not
on vLLM's unified pool (where L2 was measured neutral and removed).

`0_feasibility/` is **retained as the measurement evidence** behind the
post-mortem (feasibility was established; the effort was dropped for value, not
feasibility, reasons). The forward-looking docs and the prototype code were
removed (git-recoverable).
