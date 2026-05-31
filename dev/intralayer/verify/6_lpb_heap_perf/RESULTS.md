# verify/6 — LPB heap performance: RESULTS

**Verdict: all targets met.** LPB free-queue rotate cost driven from
5065 → 913 ns/op (16.2× → **2.8× LRU**, T1 ≤3× met) via the indexed-heap
rewrite + tiered cold-FIFO/hot-heap split. e2e L1-only PhaseH = **−8.8%**
vs LRU on fresh same-env n=3 (T2/T3 met). Full status board + per-stage
journals in [`README.md`](README.md) and [`journal/`](journal/).
