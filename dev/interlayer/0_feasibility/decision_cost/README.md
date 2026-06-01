# decision_cost — feasibility check (phase 0)

per-step 'cheapest page to free' decision is cheap: <=3x LRU per-op (verify/6-style), amortized O(1); steady-state rebalance async.

Full spec + ideal pass bar: [`../../design.md`](../../design.md) verification gate.
