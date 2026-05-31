# 9. Songyang's real SWE-bench (W2) — RESULTS

**Verdict: on real W2 traffic at a normal operating point, L1 is a no-op
(L1 ≡ LRU). Not because L1 is broken — because the workload never pressures
the KV pool, so L1's anchor-protection mechanism never engages.**

Date: 2026-05-31 · model Qwen3.5-35B-A3B · TP=2 · util=0.85 · n=3 ·
concurrency sweep {4,16,32,64} · 16 turns · 64 SWE-Bench-Lite instances.
Repro: [`run.sh`](run.sh).

## The decisive signal: the hot-heap never engaged

The LPB queue serves evictions cold-FIFO-first and only pops the hot
(hit-bearing) heap when the cold pool is exhausted. The instrumentation
(`[hima/lpb] evicts … cold=… hot=…`) shows, across **all 3 l1_only trials**:

```
evicts tot=12000 cold=12000 hot=0 (hot=0.0%) | free: cold=4136 hot=4054
```

`hot=0` at 12,000+ evictions every trial. The hot-heap *accumulated*
hit-bearing blocks (free hot grew to ~4000) but was **never popped** — there
were always cold (un-hit) blocks to evict first. L1 therefore never made a
single eviction decision that LRU wouldn't have made. **On W2, L1 ≡ LRU by
construction**, and the end-to-end numbers confirm it.

## n=3 baseline vs l1_only (all within noise)

| conc | metric | baseline | l1_only | Δ |
|---:|---|---:|---:|---:|
| 4  | avg TTFT ms | 184.1 ±37 | 186.9 ±32 | +1.5% |
| 4  | prefix hit  | 0.806 | 0.805 | ~0 |
| 16 | avg TTFT ms | 182.2 ±2 | 192.2 ±6 | +5.5% |
| 16 | throughput  | 593 ±88 | 695 ±7 | +17%* |
| 32 | avg TTFT ms | 235.0 ±8 | 231.3 ±10 | −1.6% |
| 32 | prefix hit  | 0.812 | 0.812 | ~0 |
| 64 | avg TTFT ms | 326.8 ±22 | 321.4 ±23 | −1.6% |
| 64 | p95 TTFT ms | 663 ±72 | 700 ±103 | +5.6% |
| 64 | throughput  | 1248 ±22 | 1232 ±12 | −1.3% |

\* the conc=16 +17% throughput is a single slow baseline trial (baseline
sd ±88 vs l1_only ±7); not a real win. Every signed delta has overlapping
std-devs → **noise**. Prefix-cache hit is bit-identical (0.80–0.81). Most
telling: **0 preemptions and 0% kv_cache_usage at every concurrency level** —
the pool is never pressured.

## Why — and how this fits the established model

This is the *same* mechanism documented in
[verify/1](../1_l1_isolation_existing_tests/why_l1_lost.md): L1's benefit is
**proportional to KV pressure**. L1 wins only at operating points where LRU
would evict a hit-bearing block (verify/1: util=0.35, −5…−12% TTFT; verify/5
pressure curve). W2 at util=0.85 / conc≤64 / 16 turns sits in the *slack*
regime — the working set fits, cold blocks are always available to evict, so
neither LRU nor L1 ever touches a hit-bearing block. There is nothing for L1
to protect, so it can only match LRU (and does, exactly).

The synthetic verify/1 pipeline manufactures pressure on purpose (warm one
anchor 500×, then a decoy swarm) — that's why L1 wins there. Real
SWE-Bench-Lite agent traffic at a normal serving config does not manufacture
that pressure.

## Bottom line for the HiMA decision

- **L1 does not help on realistic W2 traffic** (and does not hurt — overhead
  is in the noise). Honest answer to "is L1 really useful": at a normal
  operating point on this real workload, no.
- **L1 helps only under genuine KV pressure**, which this workload+config
  does not create. To make L1 earn its keep you must either (a) run at a
  pressured operating point (smaller pool / much higher concurrency / longer
  contexts so the working set exceeds the pool), or (b) target a deployment
  whose real traffic is already pressure-bound.
- Follow-up to convert inference→demonstration on W2 specifically: rerun a
  small-pool / high-concurrency cell and confirm `hot>0` appears and L1 then
  separates from LRU. See [`pressure_probe`](#pressure-probe) below once run.

## Pressure probe (util=0.30, conc 128/256, n=3) — L1 *loses* under pressure

Forcing pressure (1/3 pool, 256-way concurrency) made the LPB hot-path
engage (eviction instrumentation showed protected-block evictions climbing
to ~37 %). But L1 did **not** help — it got **worse**:

| conc | baseline cached% (n=3) | l1_only cached% (n=3) |
|---:|---:|---:|
| 128 | 6.16 % | 5.99 % (tied) |
| 256 | **3.55 %** | **1.46 %** (every trial lower) |

Two compounding reasons:
1. **No cross-session anchor.** The cross-agent shared cacheable prefix is
   only ~38 tokens (system + the user-message preamble before the unique
   problem statement); each agent's ~60 K-token context is unique. There is
   no globally-hot prefix for LPB's hit-count to protect.
2. **Recency-blind eviction (the bug).** The old two-tier score
   (`1e12 + n_b×c`, no recency) pinned any long-idle once-hit block above
   every freshly-freed block. The freshly-generated conversation tail
   (`n_b==0` when freed) was evicted before stale once-hit blocks that LRU
   would have dropped first — so L1 lost hits LRU kept. n_b counts *past*
   reuse via `find_longest_cache_hit`; the tail's *imminent* reuse (same
   agent's next turn) is invisible to it, while LRU's recency captures it.

## Outcome: drove the recency-aware LPB rewrite (ideal design)

Reason #2 is a real, general defect, so it was fixed at root: the LPB queue
is now keyed by `(priority, recency)` with window-driven decay, so a stale
hit demotes to LRU order and L1 can no longer fall behind LRU — it only
diverges to protect a genuinely live, repeatedly-reused prefix. See
[`vllm.md`](../../vllm.md) "Implementation" and
`tests/v1/core/test_hima_lpb_recency.py`. e2e re-verification of this fix on
both Path A (synthetic anchor) and W2 (real agents) is pending n=3.

Reason #1 is structural to W2 and not something eviction policy can fix:
without a shared hot prefix, the right tool for this workload is
session-affinity caching, not hit-count protection.
