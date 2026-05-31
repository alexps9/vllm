# vLLM HiMA L1 — LPB implementation + measured results

The vLLM-side LPB implementation, its driver script, and the
two-sweep × 3-trial measurements that landed the headline
production-pattern win.

See [`scenarios.md`](scenarios.md) for the engine-agnostic phase
design that drives both `compare_lru_lpb.py` (here) and the
sglang side ([`sglang.md`](sglang.md)).

## Status (verified 2026-05-30)

| component | verdict | evidence |
|---|---|---|
| **L1 — LPB free-block queue** | ✅ **win**: −8.8…−10.7 % Phase H TTFT, +~3 pp hit vs LRU (fresh same-env n=3) | [`verify/1`](verify/1_l1_isolation_existing_tests/README.md), [`verify/6`](verify/6_lpb_heap_perf/) |
| L1 scoring (`VLLM_HIMA_LPB_SCORING`) | **no-op**: lazy = eager = depth_tokens, bit-identical; keep `lazy` | [`verify/4`](verify/4_lpb_scoring_variants/RESULTS.md) |
| **L1 on real agent traffic (W2)** | **no win on SWE-Bench**: no cross-session shared anchor + (pre-fix) recency-blind eviction *lost* hits under pressure → drove the recency-aware rewrite below | [`verify/9`](verify/9_swebench_w2_real/RESULTS.md) |
| **L1 recency-aware scoring** | ✅ **ideal-design fix** (2026-05-31): three-tier `(priority, recency)` + window decay. Per-op ~3.9× LRU (= old two-tier). **Path A win preserved** (−10.8% PhaseH, clean n=3, long window). **W2 inversion eliminated** (short window: protected-evict 37%→~4%, L1 now ≈ LRU within noise vs −2.1pp before) | `tests/v1/core/test_hima_lpb_recency.py`, [`verify/10`](verify/10_recency_fix_reverify/RESULTS.md) |
| **L2 — admitter / budgeter / planner** | **removed** (2026-05): measured neutral (≈ LRU, −1.3 %); code deleted, slated for from-scratch redesign | [`dev/archive/L2/`](../archive/L2/) |
| interlayer partial-cache (pcache) | ❌ **removed**: no value on hybrid (mamba block-granular state caps the resume point) | git history (`M2_per_group_lift` journals) |

Both earlier "regressions" — L1 **+20.1 %** (verify/1) and L2 **+26.6 %**
(verify/3) — were **stale-baseline phantoms**: they compared current-code
against an LRU archive captured on a different GPU pair / system load. Fresh
same-environment n=3 debunked both. Net: **HiMA's value is L1 (LPB anchor
protection). L2 measured neutral and was removed (archived for redesign);
pcache is gone.**

## Implementation

| component | file |
|---|---|
| Recency-aware LPB free-block queue (heap, `(priority, recency)` key) | `vllm/v1/core/hima/lpb_free_queue.py` |
| Engine knobs (`hima_l1_enabled` / `hima_l2_enabled`) | `vllm/engine/arg_utils.py` (threads through `CacheConfig`) |
| HiMA runtime + windowed hit counter | `vllm/v1/core/hima/runtime.py` (`PathCountedHitCounter`) |
| Cost curves per pool | `vllm/v1/core/hima/cost_curve.py` |
| Hybrid KV-cache coordinator wrapper | `vllm/v1/core/hima/coordinator_hima.py` |
| Driver (this experiment) | `dev/compare_lru_lpb.py` |
| Aggregator + figures | `dev/plot_lru_vs_lpb.py` |
| Dataset | `dev/cc_long_traces.jsonl` (106 real Claude-Code sessions, 44 MB, committed) |

LPB design (recency-aware, the ideal design — 2026-05-31): **three tiers**,
evicted lowest-value first, replacing `FreeKVCacheBlockQueue` wholesale
(every pool, not just mamba):

1. **evict-first set** — stale-hit blocks demoted from the hot heap (hits
   expired). Drained first; a `set` (O(1)) since order among stale blocks
   doesn't matter.
2. **cold FIFO** — never-hit blocks (`priority == 0`), in
   `FreeKVCacheBlockQueue` recency order. **O(1) per op — the common case.**
3. **hot heap** — in-window hits, keyed by `(priority, recency)`:
   `priority = n_b_windowed × c_pool(depth)`, `recency` = monotonic access
   stamp (recent re-hit out-survives a stale hit of equal value). Touched
   only when tiers 1+2 are exhausted.

`_decay_hits` (throttled to once per ~window/16 of clock time) migrates a hot
block whose hits have aged out of the window into the evict-first tier — so
it is dropped *before* the fresh tails (which sit at the cold-FIFO tail),
exactly like LRU, while a live re-hit block stays in the hot heap. Keeping
the O(1) cold FIFO for the common case holds per-op cost at **~3.9× LRU**
(846 ns vs 217 ns, verify/6 microbench) — same as the old two-tier; a naive
single all-blocks heap was 8.1×.

**Why recency is first-class.** The prior design was a two-tier
`cold-FIFO + hot-heap` scoring `1e12 + n_b×c` with no recency term — so a
block hit once long ago out-ranked every freshly freed block *forever*
(within the window). On multi-turn agent traffic the freshly-generated
conversation tail is `n_b==0` at the instant it is freed, so it was evicted
before a long-idle once-hit block that plain LRU would have dropped first —
costing L1 hits LRU kept (see [`verify/9`](verify/9_swebench_w2_real/RESULTS.md):
SWE-Bench conc=256, L1 cached **1.5 %** vs LRU **3.5 %**, n=3). The
recency-aware score makes stale hits *decay* and rejoin LRU order, so L1 can
no longer fall behind LRU; it only diverges to *protect* a genuinely live,
repeatedly-reused prefix. This mirrors sglang's LPB (`n_hits/bytes` priority
+ `last_access_time` tie-break + a real decay window); see
[`sglang.md`](sglang.md) "Why eviction outcomes converge". Regression test:
`tests/v1/core/test_hima_lpb_recency.py`.

Wiring: `EngineArgs(hima_l1_enabled=True)` enables it (the L1 sub-flag,
post-cleanup). Window via `VLLM_HIMA_HPB_WINDOW_S` (default 60 s). The
window is now a real *decay* lever — it must be **shorter than the gap
between reuses you want to stop protecting across**. Note the inherent
tension: a long window (e.g. Path A's 3600 s, set to keep the synthetic
500×-warmed anchor protected across a multi-minute pressure run) *disables*
the decay that protects real agent traffic. There is no single window that
is right for both an artificially-pinned anchor and live session tails —
that gap is what a session-liveness signal (future work) would close.

## Sweeps

| tag | model | util | TP | KV budget | Phase F scale | files |
|---|---|---|---|---|---|---|
| `_pathA` | Qwen/Qwen3.5-35B-A3B   | 0.9 | 2 | 8.46 M tokens | 10 (50 decoys × 30 K + 500 cold) | `dev/intralayer/runs/vllm/compare_{lru,lpb}_pathA_t{1,2,3}.{jsonl,out}` |
| `_pathB` | Qwen/Qwen3.5-122B-A10B | 0.9 | 4 | 5.08 M tokens | 10                                | `dev/intralayer/runs/vllm/compare_{lru,lpb}_pathB_t{1,2,3}.{jsonl,out}` |

## Headline — anchor protection (Phase C FINAL probe)

|                                       | LRU         | LPB           | trials confirming |
|---------------------------------------|------------:|--------------:|---:|
| BASELINE anchor probe (after warmup)  | 4224 / 4737 | 4224 / 4737   | 6/6 each, stddev = 0 |
| FINAL probe (post all phases)         | **0 / 4737** | **4224 / 4737** | 6/6 each, stddev = 0 |

Binary outcome, perfectly reproducible across model size.

## Workload-metric results (mean ± sample stddev, n=3 each)

### Phase B (cc-burst average) — no regression

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr (tok/s) | LPB thr (tok/s) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A |  92.20 ± 0.68 |  92.41 ± 0.27 | +0.2 % | 12.57 ± 0.10 | 12.58 ± 0.05 | +0.1 % | 124.30 ± 0.75 | 124.73 ± 0.45 | +0.3 % |
| Path B | 125.51 ± 0.60 | 124.61 ± 2.29 | −0.7 % |  8.85 ± 0.13 |  8.75 ± 0.12 | −1.1 % | 126.36 ± 0.30 | 126.42 ± 0.95 | +0.0 % |

### Phase G (pre-pressure swarm) — tied at util=0.9 by design (control)

| sweep | LRU batch TTFT (ms) | LPB batch TTFT (ms) | Δ | LRU cached | LPB cached |
|---|---:|---:|---:|---:|---:|
| Path A | 331.66 ± 2.51 | 331.20 ± 3.08 | −0.1 % | 126 720 ±0 | 126 720 ±0 |
| Path B | 481.40 ± 2.42 | 478.32 ± 0.81 | −0.6 % | 126 720 ±0 | 126 720 ±0 |

At util=0.9 the anchor survives Phase B in both modes, so both swarms
hit. Confirms G alone isn't enough at production util — the LRU
disadvantage shows up after E + F apply pressure (Phase H below).

### Phase E (cold-unique random) — tied

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr | LPB thr | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A | 63.25 ± 0.61 | 62.82 ± 0.76 | −0.7 % |  6.32 ± 0.67 |  6.37 ± 0.57 | +0.8 % | 173.3 ± 1.9 | 171.8 ± 1.3 | −0.8 % |
| Path B | 87.15 ± 0.17 | 86.60 ± 0.55 | −0.6 % | 12.36 ± 0.85 | 12.52 ± 0.70 | +1.2 % | 115.7 ± 2.8 | 114.9 ± 2.2 | −0.7 % |

### Phase F (adversarial decoy waste) — tied

| sweep | LRU TTFT (ms) | LPB TTFT (ms) | Δ | LRU TPOT | LPB TPOT | Δ | LRU thr | LPB thr | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Path A | 62.56 ± 0.26 | 63.18 ± 0.61 | +1.0 % |  6.51 ± 0.12 |  6.59 ± 0.16 | +1.3 % | 172.97 ± 0.36 | 171.33 ± 0.19 | −0.9 % |
| Path B | 86.37 ± 0.19 | 86.58 ± 0.28 | +0.2 % | 11.83 ± 0.07 | 11.84 ± 0.09 | +0.1 % | 119.15 ± 1.22 | 119.11 ± 1.07 | −0.0 % |

Path A Phase F is the worst LPB number across all runs (+1.3 % TPOT,
~2.4 LRU-stddev gap, magnitude sub-2 %). Consistent with LPB's
heap/path-counter overhead being non-zero per request, not a
correctness regression. Note: `--phase-f-scale 10` only reaches
~49 % of KV budget on Path B; truly saturating decoys (scale ≥ 20)
would be needed to actually trigger the protect-useless-blocks
failure mode, and that hasn't been done.

### Phase H (POST-pressure SWARM) — **the production LPB win**

#### About the baseline

The headline below has two sources. Path A has been re-measured under
a **fresh n=3 same-environment sweep** (LRU + full HiMA back-to-back
on the same GPU pair); that data is authoritative and lives in
[`verify/6_lpb_heap_perf/runs/fresh_n3/`](verify/6_lpb_heap_perf/runs/fresh_n3/),
documented in
[`verify/6_lpb_heap_perf/journal/08_fresh_n3_final.md`](verify/6_lpb_heap_perf/journal/08_fresh_n3_final.md).
Path B has **not** been re-measured on a fresh same-environment
baseline — its numbers below remain from the legacy archive at
`dev/intralayer/runs/vllm/compare_{lru,lpb}_pathB_t*.jsonl`. The
legacy archive was captured at ~00:18 on 2026-05-26, when the GPUs
were under noticeably lighter load; absolute TTFT levels on that
archive run ~80–110 ms lower than under the fresh measurement
environment, so Path B's −17.7 % is a same-epoch comparison within
the archive but its absolute ms numbers are not comparable to Path A
below. See
[`verify/6_lpb_heap_perf/journal/07_phantom_regression.md`](verify/6_lpb_heap_perf/journal/07_phantom_regression.md)
for the discovery.

| sweep | source | LRU batch TTFT (ms) | LPB batch TTFT (ms) | Δ | LRU req/s | LPB req/s | total wall Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Path A | **fresh n=3** | **479 ±14** | **428 ±19** | **−10.7 %** ✓ | — | — | tied (tput 1191 ±11 vs 1211 ±36 tok/s) |
| Path A | archive (stale) | 370.03 ± 7.40 | 325.51 ± 7.41  | −12.0 % (~6σ) | 39.5 | 41.7 | −5.5 % |
| Path B | archive only | 565.35 ± 2.79 | 465.30 ± 18.07 | **−17.7 %** (~5σ) | 26.1 | 28.3 | **−8.5 %** (saves 100 ms) |

- **Headline direction is preserved**: the Phase H production-pattern
  win on Path A reproduces under fresh same-environment measurement
  (−10.7 % vs fresh LRU, n=3; the legacy archive's −12.7 %-ish reading
  was within the same direction but inflated by environment-level
  drift between LRU and LPB archive runs).
- **Path B remains archive-only**: −17.7 % is reported against an
  archive LRU captured in the same epoch as the archive LPB, so the
  *delta* is internally consistent, but Path B has not been re-run on
  the verify/6 GPU pair. Treat its magnitude as archive-confidence.
- **Hit rate is binary, stddev = 0**: LRU 85.91 % (29/30 hit, 1
  request pays the anchor prefill, the other 29 share via vLLM's
  prefix-cache merge) vs LPB 88.87 % (30/30 hit, anchor still
  cached from Phase A's warmup).
- **44 ms (Path A archive) / ~51 ms (Path A fresh) → 100 ms (Path B
  archive) gap** is approximately one anchor-prefill cost on the
  respective hardware — bigger model → slower per-token prefill →
  bigger saving.
- **Token throughput is essentially unchanged** because batched
  decode wall already dominates. The win lives in TTFT (and the
  resulting request throughput / total batch wall, where the
  ~46-100 ms shows up directly).

## Findings

1. **Production-pattern win (Phase H)**: **−10.7 % Path A (35B)** and
   **−12.2 % Path B (122B)**, both under fresh n=3 same-environment
   measurement (full/l1_only HiMA vs LRU). The legacy-archive Path B
   figure was −17.7 % (faster/different environment — same direction,
   magnitude class). The bigger model shows the larger win, as expected
   (costlier prefill ⇒ more saved by anchor protection).
2. **No regression elsewhere**: every other metric × phase × sweep
   tied within noise or LPB slightly faster. Phase B / E / F on
   the fresh n=3 are all within ≤ 2 ms median of LRU. The legacy
   archive's worst LPB outcome (Path A Phase F +1.3 % TPOT) was at
   the noise edge.
3. **Anchor protection is binary and reproducible**: 6/6 LRU
   evicts vs 6/6 LPB protects under the legacy archive. Fresh n=3
   shows both LRU and full HiMA preserve the *final* anchor at
   89 %; the differentiation under the fresh environment surfaces
   as Phase H TTFT + hit% (+3 pp) rather than catastrophic anchor
   loss.
4. **LPB's true worst case hasn't been triggered**: Phase F at
   scale=10 reaches ~49 % KV occupancy on 122 B and produces no
   measurable regression. Scale ≥ 20 would be needed.

## Follow-ups (all done 2026-05-30)

These extended/stress-tested the verified L1 win — none overturned it:

1. **LPB worst-case** (`verify/7`): ✅ no failure mode — bit-identical L1
   win across decoy scale 10/20/40; at util=0.9 the decoys never reach the
   anchor-eviction cliff. Finding 4 closed.
2. **L1 pressure curve** (`verify/1`): ✅ L1 holds the anchor to cold-burst
   K=20 vs LRU's K=10 — a **2× wider survival window**
   ([`pressure_curve_result.md`](verify/1_l1_isolation_existing_tests/pressure_curve_result.md)).
3. **Path B fresh** (`verify/1`): ✅ Qwen3.5-122B-A10B TP=4 n=3 — **−12.2 %
   PhaseH TTFT** vs LRU
   ([`pathB_fresh_result.md`](verify/1_l1_isolation_existing_tests/pathB_fresh_result.md)),
   replacing the legacy-archive −17.7 %.

## Production implications

- **Production swarm pattern** (N concurrent agents sharing an
  anchor, after the cache has been churned): LPB delivers
  **−10.7 % batch TTFT on Path A (fresh n=3)** and **−17.7 % on
  Path B (archive)**, saves ~50–100 ms per swarm depending on
  model size, with throughput tied (Path A fresh) to +8.5 %
  request throughput (Path B archive).
- **Average workload metrics** (Phases B / E / F): LPB and LRU
  are indistinguishable. No regression on cc-burst, on
  cold-unique, or on adversarial decoy patterns.

## Figures

| Path | anchor survival | per-phase grid |
|---|---|---|
| A | `dev/intralayer/figures/fig_lru_vs_lpb_anchor_pathA.png` | `dev/intralayer/figures/fig_lru_vs_lpb_scenarios_pathA.png` |
| B | `dev/intralayer/figures/fig_lru_vs_lpb_anchor_pathB.png` | `dev/intralayer/figures/fig_lru_vs_lpb_scenarios_pathB.png` |

## Repro

```bash
cd /scratch/yuzhou/projects/vllm-songyang
# venv + build (one-time)
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Path A — Qwen3.5-35B-A3B, TP=2, util=0.9. ~5 min per trial.
for trial in 1 2 3; do
  for mode in lru lpb; do
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u dev/compare_lru_lpb.py \
      --mode $mode --trial $trial \
      --tag _pathA --util 0.9 --tp 2 --phase-f-scale 10 \
      > dev/intralayer/runs/vllm/compare_${mode}_pathA_t${trial}.out 2>&1
  done
done
.venv/bin/python dev/plot_lru_vs_lpb.py --tag _pathA | tee dev/intralayer/runs/vllm/compare_summary_pathA.out

# Path B — Qwen3.5-122B-A10B, TP=4, util=0.9. ~7 min per trial.
for trial in 1 2 3; do
  for mode in lru lpb; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u dev/compare_lru_lpb.py \
      --mode $mode --trial $trial \
      --tag _pathB --util 0.9 --tp 4 --phase-f-scale 10 \
      --model Qwen/Qwen3.5-122B-A10B \
      > dev/intralayer/runs/vllm/compare_${mode}_pathB_t${trial}.out 2>&1
  done
done
.venv/bin/python dev/plot_lru_vs_lpb.py --tag _pathB | tee dev/intralayer/runs/vllm/compare_summary_pathB.out
```
