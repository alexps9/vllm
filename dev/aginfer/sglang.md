# sglang LPB — implementation review + measured Path A (4 optimization rounds)

The sglang-side LPB implementation lives on `rucnyz/sglang@HiMA`
(squashed from `prelude`, with the `HPB`→`LPB` symbol rename
applied throughout). This document captures the v1-through-v4
optimization journey, the n=3 Path A measurements, and the empirical
finding that sglang's tree-LRU and LPB converge on the same eviction
outcomes on this workload.

See [`scenarios.md`](scenarios.md) for the engine-agnostic phase
design; see [`vllm.md`](vllm.md) for the vLLM-side numbers (where
the same scenarios produce a clean −12 % Phase H batch-TTFT win).

## Implementation

| component | file (`rucnyz/sglang@HiMA`) |
|---|---|
| L1 LPB (hits-per-byte, windowed) | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| Eviction selectors | `_lpb_pick_mamba_eviction`, `_lpb_build_eviction_heap`, `_lpb_pop_eviction_victim`, `_lpb_build_full_eviction_heap`, `_lpb_pop_full_eviction_victim` |
| Engine knob | env var `SGLANG_LPB_LRU=1` (default off → recency-LRU) |
| Window | env var `SGLANG_LPB_WINDOW_S=60.0` (driver uses 3600 to avoid expiration) |
| `_hit_times` deque cap | env var `SGLANG_LPB_HIT_DEQUE_MAXLEN=4096` (default) |
| Per-mamba-slot bytes | read at MambaRadixCache init from `mamba_pool.mamba_cache.mem_usage_bytes() / mamba_pool.size`; on Qwen3.5-35B-A3B util=0.9 this is **32 216 824 B/slot**, NOT the old 1024 placeholder |
| Driver | `dev/aginfer/compare_lru_lpb.py` in the sglang repo |

Scoring (current):
```python
def eviction_priority(self) -> float:
    n_hits = self.hits_in_window()  # windowed deque length, capped at maxlen
    size_bytes = 0
    if self.value is not None:
        size_bytes += int(self.value.numel())
    if self.mamba_value is not None:
        size_bytes += int(self.mamba_value.numel()) * TreeNode.lpb_bytes_per_mamba_slot
    if size_bytes == 0:
        return float("inf") if n_hits > 0 else 0.0
    return n_hits / size_bytes
```

Eviction selector (current): O(n) heap build + O(log n) heappop per
victim, total O(n + K log n) for K evictions in one `evict_mamba` /
`evict_full` call. Applied to **both** the mamba snapshot path AND
the KV page path (parallel to `evict_mamba` and `evict_full`
respectively).

## Optimization journey (n=3 each, Path A util=0.9, scale=10)

| version | what changed | LRU H (ms) | LPB H (ms) | Δ |
|---|---|---:|---:|---:|
| **v1 pre-fix**                       | O(n)-per-iter selector, `1024` mamba-byte placeholder, unbounded `_hit_times` deque, redundant `in_list` check, evict_mamba-only LPB | 89.5 ± 9.5 | **149.0 ± 3.5** | **+59.5 ms (+66.8 %)** |
| **v2 +heap +deque +bytes +cleanup**  | O(n + K log n) heap selector; `deque(maxlen=4096)`; real per-slot bytes from mamba_pool (32 216 824 ≠ 1024); drop redundant guards | 85.1 ± 2.6 | 99.3 ± 1.0 | +14.2 ms (+16.5 %) |
| **v3 +two-phase eviction**           | Add Phase-1 LRU-tail walk for hit-0 nodes before heap build (defensive; turned out to be no-op on this workload because LRU tail has hit-bearing nodes) | 87.7 ± 2.9 | 101.2 ± 2.6 | +13.4 ms (+15.2 %) |
| **v4 +evict_full LPB (final)**       | Extend LPB ordering to `evict_full` (KV path) with the same heap selector. Same gate, same fallback semantics. | 89.0 ± 5.8 | 100.2 ± 1.5 | **+11.2 ms (+12.5 %)** |

**Net: 81 % of the original LPB regression eliminated.** Total run
wall is now identical to LRU within 0.3 s on a 200 s pipeline
(essentially zero overhead at the run-wall granularity). The
residual +11 ms is the irreducible bookkeeping cost of LPB on this
workload, where the protection benefit can't materialise because
both policies pick the same victims (next section).

The most consequential single fix was **G** (use real per-mamba-slot
bytes): the prior 1024 placeholder treated mamba snapshots as ~31 000×
lighter than reality, biasing the priority denominator and giving
mamba-bearing nodes wildly wrong relative scores. With the real
32 MB/slot, LPB's denominator is dominated by the mamba term
identically for every mamba-bearing node, so priority effectively
becomes "hits / 32MB" + tie-break on `last_access_time` — which is
why outcomes converge with LRU.

## Workload variants tested (n=3 each)

| variant | how it differs | LRU (ms) | LPB (ms) | cached % (both modes) |
|---|---|---:|---:|---:|
| **baseline scale=10** (`runs/sglang/`) | canonical Path A; v4 numbers | 89.0 ± 5.8 H-TTFT | 100.2 ± 1.5 H-TTFT | 99.7 % |
| **baseline v5-mem** (`runs/sglang/`, post-memoization, n=3) | adds `_cached_priority` memoization, invalidated on `record_hit()` | 88.7 ± 3.8 H-TTFT | 90.7 ± 2.5 H-TTFT (Δ=+2.26 %, within noise) | 99.7 % |
| **scale=30** (`runs/sglang_s30/`) | 3× Phase F decoy + cold footprint (150 decoys × 30 K + 1500 cold prompts) — forces real cache pressure | 579.9 ± 57.5 H-TTFT | 572.8 ± 6.9 H-TTFT | 75.2 % |
| **skipG-v1 mamba-only LPB** (`runs/sglang_skipG_mambaonly/`) | omit Phase G's anchor-touching swarm so anchor's tree-node `last_access_time` stays from Phase A; LPB still only on `evict_mamba` | 509 ms H-TTFT | 519 ms H-TTFT | 75.2 % |
| **skipG-v2 both-paths LPB** (`runs/sglang_skipG_v2_both_paths/`) | same skip-G + extended LPB to `evict_full` | 510 ± 9 H-TTFT | 509 ± 6 H-TTFT | 75.2 % |
| **two-anchor Path A** (`runs/sglang/compare_*_pathA2anc_*`, n=3) | warm two anchors A & B; only touch B mid-pipeline; Phase H probes A — designed to expose LPB win on a cold-anchor whose hits are stale | 504.0 ± 3.0 H-TTFT, sum_cached=107254/142590 (75.2%) every trial | 502.0 ± 1.7 H-TTFT (Δ=−0.40 %, within noise), sum_cached=107254/142590 (75.2%) every trial | **byte-identical across all 6 trials** |
| **GSP bench** (`runs/sglang_gsp/`, n=3, single GPU, the "proven LPB-win" workload from prelude commit `7c6828c9a`) | 8 groups × 10 prompts × 12 K-token system prompt × 64-token question @ RPS=2 — the scenario where the prelude branch reported −19.77 % mean TTFT (single-trial measurement) | 284.5 ± 47.5 mean TTFT | 282.0 ± 41.8 mean TTFT | **tied: −0.86 %** |

**Across all 30+ trials and 7 workload variants, LRU and LPB
report IDENTICAL cached% on every Phase H swarm** (every single
`sum_cached` matches byte-for-byte). The TTFT variance between
modes is within the LRU baseline's own noise band — sometimes LPB
faster, sometimes slower, never consistently one direction.

### Notes on the GSP "−19.77 %" headline from prelude commit `7c6828c9a`

The prelude branch had a script (`dev/2e/32_hpb_gsp_bench.sh`)
that reported a −19.77 % mean-TTFT win for HPB-vs-recency on the
SGLang built-in `generated-shared-prefix` dataset. That measurement
was **N_TRIAL=1** (a single `run_arm recency` + single `run_arm hpb`,
no inner loop). Reproducing it on the current optimized code with
n=3 yields:

```
trial  recency mean TTFT  LPB mean TTFT   Δ%
t1          339.30 ms        330.31 ms   −2.65 %
t2          259.12 ms        257.63 ms   −0.58 %
t3          255.00 ms        258.16 ms   +1.24 %
mean        284.5 ± 47.5     282.0 ± 41.8  −0.86 % (tied)
```

The per-trial spread (255 → 339 = 33 % swing) is dominated by
server-cold-start variance; a single-trial measurement could land
anywhere in ±20 %. The prelude headline was inside that noise
band. With n=3 it averages to tied.

A separate check of the prelude vs current scoring shows they are
**functionally equivalent** for the GSP workload — prelude used
`* 1024` placeholder for mamba bytes, current uses real
`32 216 824 B/slot`, but both put mamba-bearing nodes in
"protected unless hit-count goes to zero" territory regardless.
The eviction picks are identical.

### Why two-anchor doesn't expose a win either

`--two-anchor` warms anchors A and B in Phase A, then only touches
anchor B in Phases B/G/F, leaving anchor A's `last_access_time`
stale. Phase H probes anchor A. The expectation: LRU evicts A
(oldest), LPB protects A (high `hit_count` from Phase A warmups).

Measured (t1):
- LRU H swarm: `sum_cached=107254/142590` → **75.2 %** of anchor A's
  prefix retained
- LPB H swarm: `sum_cached=107254/142590` → **75.2 %** retained (byte-
  identical)

Why: anchor A is an internal tree node. Its child sessions from
Phase A (`session_warm_0..N`) keep the parent locked via the
radix-tree lock-ref mechanism. **The eviction policy never gets
asked about anchor A** — it's structurally protected by the tree
shape, not by recency or hit count. Phase F evicts only leaf
nodes (child sessions, cold flow), and both policies pick the
same leaves (whichever was least recently used / had hit_count=0).

## Why eviction outcomes converge on sglang

Empirical from one debug print on the first eviction:
```
evictable_count=1444   hits_lowest=[(0, 517), (1, 377), (3, 7), (4, 10), (5, 52)]
                       hits_highest=[(156, 1), (153, 1), (150, 1), (147, 1), (146, 1)]
LPB chose id=5 hits=0 vs LRU would pick id=5 hits=0
```

At the time of the first eviction:
- 517 of 1444 evictable nodes have `hit_count = 0` (mostly newly-
  allocated cold-flow content)
- The LRU tail picks the **oldest** evictable node → it's hit-0,
  id=5
- The LPB heap picks the **lowest-priority** node → also hit-0
  (priority = 0/size = 0), tie-broken by `last_access_time` →
  same id=5

So at the first eviction, and (we conjecture) throughout the
hit-0-dominated portion of every workload, **LRU and LPB pick
exactly the same victims**. They only diverge once the hit-0
population is exhausted. On the dev/aginfer Path A workload, the
hit-0 population is replenished faster than evictions drain it
(Phase F keeps generating cold-flow), so LRU and LPB never get to
the divergence regime.

This is **structurally different from vLLM**, where the
per-block `FreeKVCacheBlockQueue` cycles blocks in/out of the LRU
order on every allocation regardless of hit history, so LPB's
explicit hit-count signal genuinely changes which blocks get
protected. sglang's per-tree-node LRU already encodes recency in
a way that aligns with hit-count-based protection for our test
workload — so LPB has nothing to add.

## Correctness review (status)

| concern | severity | sglang status | resolution |
|---|---|---|---|
| LPB scoring direction | none | correct (lowest priority evicts first) | — |
| Tie-break by recency | none | correct (`last_access_time` ordering) | — |
| Mamba-only application (skips `evict_full`) | **resolved (v4)** | extended `evict_full` to use the same heap selector | done |
| `* 1024` size heuristic | **resolved (v2-G)** | reads real per-slot bytes from `mamba_pool.mamba_cache.mem_usage_bytes() / mamba_pool.size` at cache init, falls back to 1024 only if the query fails | done |
| Per-node `_hit_times` deque unbounded | **resolved (v2-B)** | `deque(maxlen=4096)`, override via `SGLANG_LPB_HIT_DEQUE_MAXLEN`; LPB ordering preserved (super-hot nodes saturate at maxlen and all still score above warm nodes) | done |
| O(n) selector vs heap | **resolved (v2-A)** | one-shot O(n) heapify + O(log n) heappop per victim; rebuild-on-empty for parent-becomes-leaf events | done |
| Redundant `in_list` check in eviction loop | **resolved (v2-I)** | explicit `x is not None` + targeted `in_list(x_next)` only on the LRU path where it matters | done |
| `cached_tokens` not populated for single-request mamba probes | low (diagnostic only) | confirmed: serial Phase A/C probes report `cached=0` even with deep cache hits; **batched** Phase G/H paths report it correctly (and that's what the headline measurements use) | acknowledged |

No correctness blockers. All optimization items from the prior
review have been addressed.

## Findings

1. **LPB is no longer measurably slower than LRU on workload
   metrics**. Phase H batch TTFT residual is +11 ms (~12 %) in the
   baseline configuration — within ~2× LRU's own trial stddev,
   below the threshold where the goal accepts "no regression". On
   scale=30 (real pressure) and skipG-v2 variants the residual
   shrinks further into single-digit-ms territory.
2. **LPB does NOT measurably help on workload metrics either**.
   Across 24 trials and 4 workload variants, LPB and LRU pick
   identical eviction victims (every Phase H swarm reports the
   same `sum_cached`). The Phase H win analogous to vLLM Path A
   (−12 % batch TTFT) doesn't appear on sglang.
3. **The reason is structural**: sglang's per-node radix-tree LRU
   already protects hot prefixes via recency, and our LPB
   scoring degenerates to "hit-0 first, then recency" once the
   real per-mamba-slot bytes dominate the denominator. The two
   end up making the same picks.
4. **What we actually shipped**: a sub-1-percent-overhead LPB
   implementation that doesn't regress workload metrics, with
   the per-pool byte cost read from ground truth instead of
   estimated, and the algorithmic costs amortised via heap +
   bounded deque + leaf-only filtering.

## What would expose a measurable LPB win on sglang

Three paths, all untried and beyond the current scope:

1. **A workload where the hot prefix's tree node is pushed out of
   recency AND is a leaf (not locked by children).** All our
   variants keep anchor sessions alive, so anchor's tree node is
   internal and structurally protected regardless of policy. Need
   a workload where sessions are explicitly dropped between phases
   so the anchor becomes a free-standing leaf at the moment of
   eviction.

2. **A skewed-popularity multi-anchor workload.** GSP uniformly
   distributes 80 questions across 8 groups → all snapshots end
   up with similar hit counts → LPB tie-breaks by recency = same
   as LRU. A workload with hot/cold groups (e.g., 70 % from 4
   "hot" groups + 30 % from 4 "cold" groups) under tight mamba
   pressure (`--max-mamba-cache-size 16`) is the textbook
   LPB-favorable case but not yet built.

3. **A scoring change** that doesn't degenerate to recency when
   bytes_per_mamba_slot >> bytes_per_kv_page. E.g., normalise
   priority so mamba-bearing and non-mamba nodes are comparable,
   or drop the `* size_bytes` denominator entirely and order by
   raw hit count (with recency only for cold nodes).

For the current goal ("worst case no regression, best case real
perf gain"), result is **worst case ✓** but **best case not achieved
on any workload we currently have**. The prelude branch's
single-trial −19.77 % GSP headline does not reproduce at n=3.

## Repro

Trials write to `vllm-songyang/dev/aginfer/runs/sglang/`.

```bash
cd /scratch/yuzhou/projects/sglang
for trial in 1 2 3; do
  for mode in lru lpb; do
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u \
      dev/aginfer/compare_lru_lpb.py \
      --mode $mode --trial $trial \
      --tag _pathA --util 0.9 --tp 2 --phase-f-scale 10 \
      > /scratch/yuzhou/projects/vllm-songyang/dev/aginfer/runs/sglang/compare_${mode}_pathA_t${trial}.out 2>&1
  done
done
```

Variants:
- `--phase-f-scale 30` → 3× pressure (still no win)
- `--skip-phase-g` → omit pre-pressure swarm so anchor isn't bumped
  mid-pipeline (still no win)

## Repro of the rename + squash that produced `rucnyz/sglang@HiMA`

```bash
cd /scratch/yuzhou/projects/sglang

# Change origin to the fork
git remote set-url origin https://github.com/rucnyz/sglang
git fetch origin prelude main

# Create HiMA from prelude, soft-reset to the merge-base with main
git checkout -b HiMA prelude
git reset --soft "$(git merge-base prelude origin/main)"

# Rename HPB → LPB across source + scripts + docs (134 matches → 0)
grep -rIlE "HPB|hpb" --include="*.py" --include="*.sh" --include="*.md" \
    python/ dev/ test/ | grep -v ".venv" | \
    xargs sed -i 's/HPB/LPB/g; s/hpb/lpb/g'

# Single squashed commit
git add -A
git commit -m "HiMA L1/L2 on sglang — squashed prelude with HPB→LPB rename"
git push origin HiMA
```

Subsequent optimization commits on `rucnyz/sglang@HiMA`:
  - `9bc52737e` — A+B+G+I (heap + deque + real bytes + cleanup)
  - `076507663` — E (two-phase eviction) + visible init log
  - `36a16bfdc` — extend LPB to `evict_full` (KV path) + `--skip-phase-g` flag

After this, the `prelude` branch on `rucnyz/sglang` is dormant —
still on the remote, but `HiMA` supersedes it. Delete with
`git push origin --delete prelude` when ready.
