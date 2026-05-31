# 9. Songyang's real SWE-bench scenario (W2 — SWE-Bench-Lite agents)

## What we're verifying

The decisive question: **does L1 (LPB) help on a REAL workload**, not just
the designed anchor-survival pipeline (verify/1)? L1 wins big on the
synthetic Phase A→H pattern (warm one anchor 500×, then decoy pressure).
Songyang's real scenario is **W2** — `runs/scripts/workload2.py`, which
loads real `princeton-nlp/SWE-Bench_Lite` problem statements (300 instances)
and replays them as K-turn concurrent agents (mock tool observations, no
real shell) against a live vLLM server, sweeping concurrency.

Prior W2 data (full HiMA, pre-cleanup) showed **HiMA ≈ baseline** — but that
was the buggy L1+L2+pcache stack. This re-measures **clean L1 vs LRU**.

## Hypothesis / what to look for

L1 only helps when the workload creates "a high-value prefix under eviction
pressure." SWE-Bench-Lite agents share a system/instance prelude but diverge
per turn; whether that produces an L1-protectable anchor under pressure is
exactly what this answers. The LPB eviction instrumentation (cold vs hot
eviction source) tells us if L1's hot-heap even **engaged**.

## How to repro

```bash
bash dev/intralayer/verify/9_swebench_w2_real/run.sh
# server: runs/scripts/start_server.sh {baseline,l1_only} on Qwen3.5-35B-A3B
# client: runs/scripts/workload2.py (SWE-Bench-Lite, concurrency sweep), n=3
```

## Status

🔄 running. Results → [`RESULTS.md`](RESULTS.md); raw → `runs/`.
