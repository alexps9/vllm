# verify/3 fresh n=3 — the +26.6% L2-only regression is a phantom (2026-05-30)

The previously published **+26.6% L2-only TTFT regression** was computed
against a STALE LRU baseline (captured on a different GPU pair / system
load). This reran `l2_only` AND `lru` back-to-back in the same environment
(`run_fresh_n3.sh`, compare_lru_lpb.py PathA, util=0.9, TP=2,
phase-f-scale=10, GPUs 5,6).

## Result

| config | n | PhaseH hit% | swarm2 batch wall (ms) | anchor cached |
|---|---:|---:|---:|---:|
| lru | 3 | 85.91 ± 0.00 | 420 ± 8 | 122496 |
| l2_only | 2 | 85.91 ± 0.00 | 415 ± 0 | 122496 |

**l2_only vs fresh lru: −1.3%** (within noise; if anything slightly
faster). Identical PhaseH hit% (85.91) and anchor survival (122496).

(l2_only t3 wedged at engine startup — the recurring SIGKILL-resistant
CUDA-init hang, see below — so l2_only is n=2. The deterministic
hit%/anchor identity makes that sufficient.)

## Verdict

The **+26.6% was a stale-baseline phantom**, exactly the same artifact
class as the L1 "phantom regression" (verify/6 journal/07). On fresh
same-environment measurement **L2-only is within noise of LRU**. This makes
sense structurally: `l2_only` uses the LRU free queue (LPB is gated on
hima_l1), so eviction order is identical to LRU; L2's admitter/budgeter/
planner add only a small (sub-noise) admission overhead and do not change
which blocks survive on this scenario.

## HiMA L1/L2 status — closed

| layer | verdict | evidence |
|---|---|---|
| **L1 (LPB)** | **wins** −8.8% to −10.7% PhaseH TTFT, +~3pp hit | verify/1 (fresh n=3), verify/4 |
| L1 scoring variants | no-op (lazy already optimal) | verify/4 |
| **L2 (admitter/budgeter/planner)** | **neutral** (≈LRU; +26.6% was a phantom) | this doc |
| interlayer pcache | removed (no value on hybrid) | M2_per_group_lift journals (git history) |

Net: HiMA's value is **L1 (LPB anchor protection)**; L2 is neutral on these
workloads; pcache was removed.

## Operational note — recurring host wedge

The host repeatedly wedges on vLLM **engine startup**: process enters state
`RN`, becomes SIGKILL-resistant, 0 GPU bytes, empty output. Seen at session
start, during verify/4 (eager_depth_tokens t2), and again here (l2_only t3)
— it recurred ~40 min after a fresh reboot. A reboot clears it but it comes
back under sustained GPU run cycling. Future GPU sweeps on this host should
expect to lose ~1 cell per ~5–8 runs to this and plan n with margin.
