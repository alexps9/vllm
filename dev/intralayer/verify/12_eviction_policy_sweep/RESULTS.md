# verify/12 — eviction-policy sweep: is there ANY L1 path on real agent traffic?

**Verdict: NO path. On real CC agent traffic, every eviction policy ties LRU
within ~1pp — including current L1. L1 is at the workload ceiling; there is no
eviction-policy improvement to be had on this traffic.** Cheap offline
experiment (no GPU); reproduce: `sim.py`; raw: `runs/`.

## The model-independent kill (independently verified)
Real CC agent sessions have **0% cross-session prefix sharing** — with proper
prefix-chained block ids, of all distinct prefix-blocks across the 106
sessions, **0 are shared by >1 session (max_readers = 1)**. Each session is an
isolated growing prefix (different repos, diverging from the start). So the only
reuse is *within* a session (re-issuing its own growing conversation), which is
**a single growing prefix re-read in order = the access pattern LRU handles
optimally**. There is no popularity/fan-out signal to exploit and no
scan-vs-reuse conflict to resolve → **no policy can beat LRU.**

## What was tested
Offline block-cache simulator on `dev/intralayer/cc_long_traces.jsonl` (106
multi-turn CC sessions; char/4 token proxy, content-hash block identity = true
prefix-cache semantics). 8 policies — **LRU, LFU, windowed-LFU+LRU (= L1's
faithful reduction), ARC, 2Q, LIRS, LRFU, ManyReaders(prefix-popularity)** —
× block_size {16, 1056} × budgets {5/10/25/50%} × 3 seeds × 2 concurrency
models {lockstep round-robin, realistic arrival}.

## Results (hit rate, mean over seeds)
**Realistic arrival concurrency, bs=1056 — everything ties LRU:**

| model / budget | LRU | L1 | ARC | 2Q |
|---|---:|---:|---:|---:|
| arrival conc=24, 10% | 73.3 | 72.4 | 73.4 | 74.6 |
| arrival conc=24, 25% | **98.2** | 97.3 | 98.1 | 97.2 |
| arrival conc=106, 5% | 25.9 | 25.4 | 26.3 | 27.1 |
| arrival conc=106, 10% | 44.8 | 44.2 | 44.9 | 45.9 |
| arrival conc=106, 25% | 82.0 | 81.9 | 82.0 | 82.3 |

All within ~1pp. `ManyReaders ≡ LRU` exactly (nothing to protect — 0% sharing).
LFU collapses (frequency bias is harmful for prefix traffic). L1 is neutral, and
at bs=16 slightly **negative** vs LRU (23.7 vs 26.0 @10%) — windowed-LFU pins
fine blocks LRU would recycle.

## The self-caught artifact (important honesty)
A **lockstep** concurrency model (all 106 sessions cycling prefixes in phase)
showed an apparent **+7–10pp ARC/2Q win** over LRU (e.g. 10% budget: ARC 23.4
vs LRU 13.1). Isolating the variable showed this is **phase-alignment, not a
real win**: under realistic arrival concurrency it vanishes entirely. The only
regime where any policy beats LRU is artificially phase-locked traffic, which is
not representative of serving.

## Corroboration
This matches and explains the prior measured result (L1 ≈ LRU on W2 real
SWE-bench traffic; wins only on the *synthetic* anchor pattern) and this
session's finding that the cost curve is inert for L1's ordering. The ~10%
anchor win requires a dominant prefix hit across many concurrent sessions under
pressure — i.e. **cross-session sharing — which CC/SWE-bench agent traffic does
not have.**

## Where a win WOULD exist (a different workload, not the agent target)
High cross-session sharing — shared system prompts across many users, RAG over a
hot corpus, multi-user shared long context. There, **ARC** (not windowed-LFU/L1)
dominated in the (sharing-heavy) lockstep runs. If such a workload becomes a
target, ARC is the policy to validate; for agent traffic there is no path.

## Recommendation
**No GPU. L1 is done.** No eviction-policy change improves hit rate on agent
traffic; the win is structurally tied to cross-session sharing the traffic
lacks. (Caveats: hit-rate proxy not end-to-end TTFT; char/4 tokenization;
modeled concurrency — but the 0%-sharing structural fact is model-independent
and independently re-verified.)
