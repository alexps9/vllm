# 04 — Summary + Go/No-Go (2026-05-27)

## TL;DR

**GO.** Cleanup commits `a9a087759 … 469bdb947` (8 commits, including
the legacy/back-compat audit) did NOT introduce any behavioural
regression on the Path A bench. All headline findings reproduce.

(This run originally also had a Pipe B partial-cache bench; pcache was
later removed from the codebase, so the Pipe B coverage/findings have
been dropped from this summary.)

## Coverage

| pipe | bench | mode set | n | source-of-truth doc | verdict |
|---|---|---|---|---|---|
| A | `compare_lru_lpb.py` PathA util=0.9 | lru / l1_only / l2_only | 3 | `dev/intralayer/vllm.md` + verify/6 journal/08 | ✅ matches |

## Headlines preserved

1. **L1 LPB win on Path A** — −10.7 % swarm2 TTFT vs LRU, n=3
   (archive: −8.8 %, same direction, magnitude class).
2. **Full ≈ L1-only on Path A** — confirms L1 carries the headline
   win; L2 alone is +8.8 % regression on fresh baseline.

## What's next

- Phase 13 (hybrid W1 repro on Qwen3.5-35B-A3B) remains the open
  in-progress task. Negative result on synthetic workload already
  noted; need real-data follow-up or accept-and-document.
- L2-only's +8.8 % regression vs fresh LRU is real but smaller than
  the +26.6 % flagged in verify/3 against stale-LRU. Worth tightening
  the verify/3 conclusion.

## Files

- `02_pipe_a_pathA_results.md`
- `../runs/` — Path A raw JSONL + .out logs
