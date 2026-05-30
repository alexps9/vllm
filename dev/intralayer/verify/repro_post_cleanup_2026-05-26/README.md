# Post-cleanup full re-verification (2026-05-26)

After the legacy/back-compat audit + cleanup pass (commit `5bec64340`,
plus the earlier 7 commits pushed to `origin/HiMA`), re-run the
canonical n=3 verifications to confirm no behavioural regression.

## What we rerun

| pipe | bench | modes | config | GPUs | output |
|---|---|---|---|---|---|
| A | `compare_lru_lpb.py` PathA | `lru`, `l1_only`, `l2_only` | util=0.9, TP=2, phase-f-scale=10, Qwen3.5-35B-A3B | 1,2 | `runs/compare_<mode>_pathA_repro_t{1,2,3}.{jsonl,out}` |

(The original run also had a Pipe B — a partial-cache canonical workload —
which was removed when pcache was deleted from the codebase.)

Results go in `runs/` with `_repro` suffix to keep the Phase 11h archive
files untouched for direct A/B comparison.

## How to launch

```bash
bash dev/intralayer/verify/repro_post_cleanup_2026-05-26/run.sh
# tail log:
tail -f dev/intralayer/verify/repro_post_cleanup_2026-05-26/runs/pipe_a.log
```

## Expected (pre-cleanup archive numbers to match)

**Pipe A — compare_lru_lpb.py PathA (Phase 11h archive, n=3 means)**:
- LRU baseline last-window TTFT: ~440 ms ± noise
- L1-only: ~8.8 % faster than LRU
- L2-only: ~26 % slower than LRU (regression noted in verify/3)

Acceptance gate: post-cleanup n=3 means within ±10 % of archive on both
TTFT and hit% (hit% is essentially deterministic on these workloads).

## Journal

- `journal/01_kickoff.md` — this run's plan + launch timestamps
- `journal/02_pipe_a_pathA_results.md` — A/B vs archive (Phase 11h)
- `journal/04_summary.md` — go/no-go for next phase
