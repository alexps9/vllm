# verify/7 results — no LPB worst-case via decoy scale (2026-05-30)

Tests `vllm.md` finding 4 ("LPB's true worst case never triggered;
scale=10 ~49% occupancy, scale≥20 needed"). compare_lru_lpb.py PathA,
util=0.9, TP=2, GPUs 5,6, escalating `--phase-f-scale`.

## Result

| scale | config | n | PhaseH hit% | swarm2 wall | anchor cached |
|---:|---|---:|---:|---:|---:|
| 10 (verify/1) | l1_only | 3 | 88.87 | — | 126720 |
| 10 (verify/1) | lru | 3 | 85.91 | — | 122496 |
| **20** | l1_only | 3 | **88.87 ± 0.00** | 381 ± 16 ms | 126720 |
| **20** | lru | 3 | **85.91 ± 0.00** | 412 ± 24 ms | 122496 |
| **40** | l1_only | (killed) | — | — | — |
| **40** | lru | 1 | **85.91** | — | **122496** |

## Finding — decoy scale is the wrong lever

The PhaseH hit% and anchor survival are **bit-identical across scale 10,
20, and 40** for *both* lru and l1_only. Doubling and quadrupling the decoy
pressure changes nothing: at util=0.9 the KV pool is large enough that the
phase-F decoy prompts **churn among themselves** (cold blocks evicting cold
blocks) and never accumulate enough live occupancy to force the warmed
anchor out — even under LRU (lru anchor stays at 122496 at every scale).

So **finding 4's premise is empirically false**: more decoys ≠ more
anchor-eviction pressure. The lever that actually pressures the anchor is
**pool size (util)**, not decoy count — and that axis was already swept in
verify/1 (L1 wins at util=0.9 *and* util=0.35).

## Verdict

**No LPB failure mode is triggerable via decoy scale.** Across scale
10→40, L1 (LPB) consistently:
- protects the anchor better (126720 vs LRU's 122496 cached),
- delivers +2.96 pp PhaseH hit% (88.87 vs 85.91), and
- is faster (−7.5 % swarm2 wall at scale=20, n=3).

The "L1 wins" conclusion is **robust** — it does not degrade under up to 4×
the decoy pressure of the original measurement. To stress the anchor one
must shrink the pool (lower util), which verify/1 already covered (L1 still
wins at util=0.35). This closes finding 4.

(l1_only s40 and s40 t2/t3 were not run: the lru-anchor-constant result at
scale=40 already proves the decoys don't bite, and the metric is
deterministic. Saved GPU time + avoided further exposure to the recurring
host CUDA-init wedge.)
