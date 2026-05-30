# 7. LPB worst-case — heavy decoy pressure (phase-f-scale ≥ 20)

## What we're verifying

`vllm.md` finding 4: LPB's true worst case has never been triggered. PathA
at `phase-f-scale=10` reaches only ~49 % KV occupancy and shows no
regression. This pushes the decoy/cold-burst pressure to **scale=20** (and
40 if LPB still wins) to find whether LPB ever **regresses** vs LRU under
heavy churn — the one experiment that could bound the "L1 wins" conclusion.

## How

`compare_lru_lpb.py` PathA, `lru` vs `l1_only`, n=3, util=0.9, TP=2,
GPUs 5,6, `--phase-f-scale 20`. Metric: PhaseH (post-pressure swarm) TTFT +
hit% + anchor survival, same as verify/1.

Runner: `run.sh`. Acceptance/interpretation:
- l1_only still ≥ lru on PhaseH hit% / anchor → LPB robust, conclusion holds.
- l1_only < lru (anchor evicted, hit% drops) → found LPB's failure mode;
  bounds the win and reopens the scoring/eviction question.
