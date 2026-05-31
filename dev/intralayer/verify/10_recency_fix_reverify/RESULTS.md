# 10. Recency-aware LPB fix — RESULTS

Date 2026-05-31 · Qwen3.5-35B-A3B · TP=2.

## 1. Path A regression guard — ✅ win PRESERVED (clean n=3)

Long window (3600 s), util=0.9, phase-f-scale=10, GPUs 6,7, sequential.

| metric | LRU (n=3) | L1 (n=3) | Δ |
|---|---:|---:|---:|
| Phase H TTFT (post-pressure swarm) | 415.8 ±5.3 | **370.9 ±6.8** | **−10.8%** |
| Phase G TTFT (pre-pressure) | 395.4 ±13.3 | 395.1 ±14.2 | ~0 |
| Phase H anchor cached tok | 122 496 | **126 720** | L1 holds more |

Per-trial PhaseH: LRU [408, 418, 421], L1 [374, 377, 362]. The recency-aware
rewrite **does not regress** the synthetic anchor win (prior was −8.8%;
−10.8% is within run-to-run variation). As designed, with the long window the
decay stays inert, so behaviour matches the pre-rewrite hot-heap.

## 2. W2 short-window — ✅ inversion ELIMINATED (clean n=3 at conc256)

Short window (60 s) so stale hits decay, util=0.30, conc 128/256, GPUs 2,3.
(The host wedged 3× mid-day; the clean n=3 below was assembled by clearing
leaked GPU workers and running the missing cells in small batches.)

### The decisive, robust signal: decay engages

The LPB eviction instrumentation, every l1 run:

```
[hima/lpb] evicts tot=84000 protected=3507 (4.2%) | evict_q=0 cold=… hot=…
```

**protected-evict = ~4–5%**, versus **37%** pre-fix (window 3600, verify/9).
Pre-fix the recency-blind hot heap pinned stale hits and forced 37% of
evictions to come from the protected set; now stale hits decay to the
evict/cold tiers and are dropped in LRU order, leaving only genuinely-live
hits (~4%) protected. **This is the fix working exactly as designed.**

### cached% — the systematic inversion is gone (n=3)

| | pre-fix (w=3600), n=3 | post-fix (w=60), n=3 |
|---|---|---|
| conc256 baseline | 3.55 % | 3.32 % ±0.41 [3.65, 2.74, 3.58] |
| conc256 **l1_only** | **1.46 %** (−2.1pp, *every* trial lower) | **3.20 % ±0.54** [3.73, 3.40, 2.46] → **Δ −0.13pp** |
| conc128 baseline | 6.16 % | 6.03 % ±0.43 [6.64, 5.70, 5.76] |
| conc128 **l1_only** | 5.99 % | 4.86 % ±1.02 [4.56, 3.78, 6.23] → Δ −1.18pp |

**conc256 (the decisive max-pressure case where pre-fix L1 was −2.1pp below
LRU on *every* trial): the inversion is gone** — Δ = −0.13pp with overlapping
±0.4–0.5 error bars, i.e. L1 is now statistically indistinguishable from LRU.

At conc128 L1 is nominally −1.18pp lower, but with large variance (±1.02; one
trial 6.23 ≈ baseline) — error bars overlap baseline's, so it is
noise-dominated, not a clean deficit. This is a degenerate thrash regime
(1–7 % hit for everyone), inherently high-variance.

L1 does **not win** on W2 (expected — no cross-session shared anchor to
protect, verify/9); it just no longer systematically *loses*. The decisive,
reproducible evidence is the **mechanistic** protected-evict 37%→~4%: the
decay engages and stale hits stop pinning the heap.

## 3. Methodology check — concurrency inflates variance (collect data ISOLATED)

Question: with GPUs free, can we run measurements concurrently (3× TP=2 on
pairs 2,3 / 4,5 / 6,7) to go faster, or does that change the numbers? Tested
the *same* config isolated vs 3-way concurrent, on two metrics:

| metric | isolated (n=3) | 3-way concurrent | effect |
|---|---|---|---|
| W2 l1 conc256 cached% | 3.20 ±0.54 | 2.24 ±0.81 — [3.30, 2.10, 1.32] | variance ↑, mean −1pp |
| Path A l1 PhaseH TTFT | 370.9 **±6.8** ms | 379.5 **±16.8** ms — [397, 384, 357] | **variance ↑ 2.5×**, mean +2.3% |

**Robust finding: concurrency inflates measurement variance (~2.5× on the
tight Path A TTFT metric), with a ~40 ms spread among identical simultaneous
jobs.** The mean bias is small (+2.3% / −1pp).

**Walk-back:** the W2 run showed a monotonic per-pair ordering
(2,3 > 4,5 > 6,7) that initially looked like a fixed NUMA/contention gradient.
It does **not replicate** — Path A's ordering *flipped* (2,3 worst, 6,7 best).
With n=1 per pair a monotonic order arises ~1/3 of the time by chance, so that
was over-read; there is **no proven fixed per-pair bias**, only contention
*noise*.

**Rule:** collect A/B comparison data **isolated** (one job at a time). For a
~10% effect size (e.g. Path A's −10.8% win) the concurrency-added variance is
enough to blur the signal in a single run. (Separately: the CUDA-init host
wedge is intermittent — it did not fire in either concurrent batch here, but
fired 3× earlier in the day. Independent reason to avoid concurrent runs.)

## Verdict

- ✅ Path A win preserved (−10.8%, clean n=3) — no regression.
- ✅ W2 systematic inversion eliminated — decay engages (protected 37%→~4%);
  at conc256 (n=3) L1 = LRU within noise (Δ −0.13pp) instead of −2.1pp below
  on every trial pre-fix.
- L1's value remains regime-specific: it wins where a shared anchor is under
  pressure (Path A), and now safely matches LRU where there isn't (W2).
