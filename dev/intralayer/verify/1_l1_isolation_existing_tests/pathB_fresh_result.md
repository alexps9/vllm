# verify/1 Path B fresh n=3 — L1 win confirmed on Qwen3.5-122B-A10B (2026-05-30)

The −17.7 % Path B figure in `vllm.md` was **legacy-archive only** (never
re-measured fresh same-env). This is the fresh n=3: compare_lru_lpb PathA
pipeline on **Qwen3.5-122B-A10B, TP=4, util=0.9, phase-f-scale=10**, GPUs
4,5,6,7, lru vs l1_only.

## Result

| config | n | PhaseH hit% | PhaseH TTFT (swarm2 wall) | anchor cached |
|---|---:|---:|---:|---:|
| l1_only | 3 | 88.87 ± 0.00 | **519 ± 17 ms** | 126720 |
| lru | 3 | 85.91 ± 0.00 | 591 ± 17 ms | 122496 |

**l1_only vs lru PhaseH TTFT: −12.2 %** (n=3, ±17 ms). +2.96 pp hit%,
anchor protected (126720 vs 122496).

## Verdict

L1 (LPB) wins on the large model too. The fresh −12.2 % is the same
direction as the archive's −17.7 % (the archive was a faster/different
environment, so the magnitude differs but the class holds), and it is
**larger than the 35B Path A win (−10.7 %)** — as expected: a bigger model
has costlier prefill, so protecting the warmed anchor saves more wall-clock.

This was the last open experiment. The HiMA L1/L2 picture is now fully
verified across both target models:

| | 35B (Path A) | 122B (Path B) |
|---|---:|---:|
| L1 PhaseH TTFT vs LRU | −10.7 % (n=3) | **−12.2 % (n=3)** |
| L1 PhaseH hit% vs LRU | +2.96 pp | +2.96 pp |
