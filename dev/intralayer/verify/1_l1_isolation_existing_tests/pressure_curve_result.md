# verify/1 pressure curve (fresh) — L1 doubles the anchor survival window (2026-05-30)

`e2e_l1_pressure_curve.py` sweeps cold-burst pressure K (number of cold
sessions churned between anchor warmup and the survival probe) and reports
whether the warmed anchor (4224 cached tokens = ~4.5 blocks) is still cached
afterward. Fresh same-env, l1_only vs lru, util=0.35, TP=2, GPUs 5,6.

## Result — anchor cached vs cold-burst K

| K | lru anchor | l1_only anchor |
|---:|---:|---:|
| 0  | 4224 | 4224 |
| 5  | 4224 | 4224 |
| 10 | **0** (evicted) | 4224 |
| 15 | 0 | 4224 |
| 20 | 0 | **0** (evicted) |
| 25 | 0 | 0 |
| 30 | 0 | 0 |

**LRU's anchor breaks at K=10; L1 (LPB)'s at K=20 — a 2× wider survival
window.** Below K=10 both protect it (no pressure); above K=20 neither can
(pool fully churned). In the K=10–15 band only LPB keeps the anchor — that
is exactly the regime the discrete Phase A→H swarm measures as the
−8.8…−10.7 % TTFT win.

## Interpretation

This is the continuous-pressure view that complements verify/1's discrete
Phase A→H points: LPB's anchor protection isn't binary-everywhere — it has a
cliff, but the cliff sits at **2× the cold-burst pressure** of LRU's. So LPB
buys roughly a doubling of how much churn a high-value prefix can survive.
Consistent with the verify/5 window-sensitivity cliff (K≈20–25).

Note: this is the same mechanism as verify/7's finding that decoy *scale*
doesn't bite at util=0.9 — there the pool was large enough that even K-large
decoys never crossed the cliff. Here at util=0.35 (smaller pool) the cliff
is reachable, and L1 pushes it out 2×.
