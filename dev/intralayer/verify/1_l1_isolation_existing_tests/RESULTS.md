# verify/1 — L1 (LPB) isolation: RESULTS

**Verdict: L1 wins on both target models, fresh same-env n=3.**

| metric | 35B Path A (util=0.9) | 122B Path B (util=0.9) |
|---|---:|---:|
| PhaseH TTFT vs LRU | **−8.8%** (l1_only) / −10.7% (full) | **−12.2%** (l1_only) |
| PhaseH hit% vs LRU | +2.96 pp (88.87 vs 85.91) | +2.96 pp |
| throughput | tied | tied |

Plus (util=0.35, n=3): L1 −5% PhaseH TTFT — wins under genuine KV pressure too.

**Anchor survival window (pressure curve):** L1 holds the warmed anchor to
cold-burst K=20 vs LRU's K=10 — a **2× wider survival window**.

Detail: [`pathB_fresh_result.md`](pathB_fresh_result.md),
[`pressure_curve_result.md`](pressure_curve_result.md). Repro: [`run.sh`](run.sh).

> Note: [`why_l1_lost.md`](why_l1_lost.md) investigated an apparent L1 **+20%
> loss** — that was a **stale-baseline phantom** (current code vs an LRU
> archive from a lighter-load epoch). Fresh same-env n=3 (above) debunked it.
