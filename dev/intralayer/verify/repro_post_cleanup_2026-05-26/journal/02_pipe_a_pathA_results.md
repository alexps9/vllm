# 02 — Pipe A PathA n=3 vs Phase 11h archive (2026-05-27)

## Setup

- `compare_lru_lpb.py --util 0.9 --tp 2 --phase-f-scale 10`
- Qwen3.5-35B-A3B, TP=2
- Pipe A intended GPUs 1,2. Trials l2_only t2+t3 collided with an
  external workload on GPU 1 at ~23:44 (132 GB/143 GB occupied) → engine
  init OOM. Reran l2_only t2+t3 on GPUs 5,6 between 23:53 and 00:23.
- All other trials (lru t1-3, l1_only t1-3, l2_only t1) ran on GPUs 1,2
  before the contention.

## Numbers (swarm2 batch TTFT — Phase H production-pattern headline metric)

| mode | source | swarm ms | swarm2 ms | swarm2 Δ vs LRU |
|---|---|---:|---:|---:|
| lru | 11h archive | 449.1 ± 26 | 478.7 ± 11 | — |
| lru | **post-cleanup** | 475.6 ± 7 | **480.1 ± 29** | — |
| l1_only | 11h archive | 477.1 ± 6 | 436.5 ± 18 | **−8.8 %** |
| l1_only | **post-cleanup** | 473.3 ± 16 | **428.6 ± 15** | **−10.7 %** |
| full (L1+L2) | 11h archive | 473.9 ± 9 | 427.6 ± 15 | **−10.7 %** |
| l2_only | (no archive) | — | — | — |
| l2_only | **post-cleanup** | 457.2 ± 27 | **522.5 ± 30** | **+8.8 %** |

## Findings

1. **LRU baseline reproduces.** 480.1 ms vs archive 478.7 ms (within
   0.3 %, well below 1 σ). The cleanup did not perturb the LRU code
   path.

2. **L1-only win reproduces, slightly larger.** −10.7 % vs LRU
   (post-cleanup) vs −8.8 % (archive). Within 1 σ — same direction,
   same magnitude class. The Phase H headline (L1 LPB beats LRU on
   the post-pressure swarm) is intact.

3. **L2-only is a regression.** +8.8 % vs LRU (post-cleanup). Magnitude
   smaller than verify/3's archived `+26.6 %` because that earlier
   number was vs *stale-LRU* (different epoch). On fresh same-epoch
   LRU, L2 alone costs ~42 ms (~9 %) of swarm2 TTFT.

4. **Full ≈ L1-only.** Phase 11h's full (L1+L2) was 427.6 ± 15 ms;
   post-cleanup l1_only is 428.6 ± 15 ms — bit-identical within
   noise. Implication: on PathA the headline win comes from L1; L2's
   regression cancels what L2 might have added. This is consistent
   with verify/3's L2-isolation conclusion.

## Verdict

No cleanup-induced regression. Phase 11h headlines (L1 LPB Path A
win, magnitude class −10 %) preserved on the post-cleanup branch.
The L2-only number on fresh-baseline (+8.8 % instead of +26.6 %)
refines verify/3's earlier "pending fresh re-measure" caveat into a
concrete same-epoch number.

## Files

- Raw JSONL: `../runs/compare_{lru,l1_only,l2_only}_pathA_repro_t{1,2,3}.jsonl`
- Archive: `../../6_lpb_heap_perf/runs/fresh_n3/compare_{lru,l1_only,full}_pathA_fresh_t{1,2,3}.jsonl`
- Aggregation: ad-hoc Python (see chat log) reading
  `swarm_batch.ttft_batch_wall_s` and `swarm2_batch.ttft_batch_wall_s`.
