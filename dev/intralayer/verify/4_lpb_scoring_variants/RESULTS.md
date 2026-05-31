# verify/4 results — LPB scoring variant attribution (Phase 10b)

Date: 2026-05-30. Knob: `VLLM_HIMA_LPB_SCORING` (Phase 10a).
Test bed: `compare_lru_lpb.py` PathA (Phase A→H), Qwen3.5-35B-A3B,
util=0.9, TP=2, phase-f-scale=10, GPUs 5,6. Warms one anchor 500× then
applies decoy pressure; PhaseH = post-pressure concurrent swarm.

## Result

| config | n | PhaseH hit% (mean±sd) | anchor cached (swarm2 batch) |
|---|---:|---:|---:|
| l1_only **lazy** (current default) | 2 | **88.87 ± 0.00** | 126720 |
| l1_only **eager** | 2 | 88.87 ± 0.00 | 126720 |
| l1_only **depth_tokens** | 2 | 88.87 ± 0.00 | 126720 |
| l1_only **eager_depth_tokens** | 1 | 88.87 ± 0.00 | 126720 |
| **lru** (floor; scoring inert) | 1 | 85.91 ± 0.00 | 122496 |

(Per-trial PhaseH hit% / anchor-cached are **bit-identical** across all
l1_only variants — anchor survival is deterministic and binary here, so
the values carry zero variance; n=2 is conclusive for a "no-difference"
claim. The missing eager_depth_tokens t2 / lru t2,t3 cells were dropped
because a run wedged at engine startup — see "host wedge" below — but the
deterministic identity makes them unnecessary. TTFT was noisy across
trials (l1 ~363–401 ms, lru ~415 ms) with no variant-ordered signal.)

## Verdict

1. **The LPB scoring variant makes no difference.** lazy = eager =
   depth_tokens = eager_depth_tokens, bit-identical on hit% and anchor
   survival. The two suspected bugs are real *in principle* but **benign
   in practice** on this workload:
   - *lazy refresh*: after the 500× warmup the anchor's `n_b` already
     dominates any decoy's score by orders of magnitude, so a stale score
     still keeps it at the top of the hot heap — refreshing changes
     nothing.
   - *depth-as-integer*: feeding the cost curve `depth×block_size` instead
     of `depth` rescales every block's cost by the same monotone factor;
     it does not reorder anchor vs decoy, so the eviction outcome is
     unchanged.
2. **L1 (LPB) itself works.** Every l1_only variant beats the LRU floor
   (88.87 % vs 85.91 %, +2.96 pp; anchor 126720 vs 122496 cached) — LPB
   protects the warmed anchor that LRU partially evicts. Consistent with
   verify/1's fresh n=3 (-8.8 % PhaseH TTFT).

**Decision: keep `lazy` as the default.** The eager / depth_tokens fixes
are not worth adopting — they add cost (eager: a refresh per record_hit)
for zero benefit on the scenario LPB is designed for. The knob stays as a
diagnostic instrument. This closes the original verify/4 premise ("an L1
scoring bug causes the W1 regression") as a non-issue: post-Phase-11 there
is no L1 regression, and the scoring details don't move the result.

## host wedge (operational note)

eager_depth_tokens t2 wedged at engine startup: process stuck in state
`RN`, **SIGKILL-resistant**, 0 GPU bytes allocated, empty output, ~68 min —
the same CUDA-init wedge seen at this session's start (which needed a
reboot). It does not affect the conclusion (deterministic metric) but the
host appears prone to re-wedging on vLLM engine init under tenant
contention; a reboot may be needed before further GPU runs.
