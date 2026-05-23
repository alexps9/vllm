# sglang LPB — implementation review + measured Path A

The sglang-side LPB implementation that lives on `rucnyz/sglang@HiMA`
(squashed from `prelude`, with the `HPB`→`LPB` symbol rename
applied throughout).

See [`scenarios.md`](scenarios.md) for the engine-agnostic phase
design that's also run here; see [`vllm.md`](vllm.md) for the
vLLM-side numbers to compare against.

## Implementation

| component | file (`rucnyz/sglang@HiMA`) |
|---|---|
| L1 LPB (hits-per-byte, windowed) | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| Eviction selector | `_lpb_pick_mamba_eviction()` in the same file |
| Engine knob | env var `SGLANG_LPB_LRU=1` (default off → recency-LRU) |
| Window | env var `SGLANG_LPB_WINDOW_S=60.0` (default; we use 3600) |
| L2 inter-pool actuator | `python/sglang/srt/budgeter/` (cost_model, cross_pool_planner, fire_planner, pressure_adapter) |
| Driver (this experiment) | `dev/aginfer/compare_lru_lpb.py` (in sglang repo) |

Score:
```python
def eviction_priority(self) -> float:
    n_hits = self.hits_in_window()
    size_bytes = int(self.value.numel()) + (int(self.mamba_value.numel()) * 1024 if self.mamba_value else 0)
    return n_hits / size_bytes if size_bytes else (float("inf") if n_hits else 0.0)
```

Eviction selector (`_lpb_pick_mamba_eviction`): O(n) scan over the
evictable mamba LRU list; falls back to recency-LRU on tie.

Wiring is gated by env var only — `SGLANG_LPB_LRU=1` is the switch.
Applied only in `evict_mamba()` (not `evict_full()`), so LPB
protection covers mamba snapshots, not raw KV pages.

## Design comparison vs vLLM HiMA L1

| dimension | sglang LPB (`HiMA`) | vLLM HiMA L1 (`HiMA`) |
|---|---|---|
| where the queue lives | only `evict_mamba()` path | full `FreeKVCacheBlockQueue` replacement, all pools |
| score metric | `hits_in_window() / size_bytes` (continuous) | binary: cold = `time.monotonic()`, hot = `+ 1e12` |
| windowing | sliding 60 s deque per node | sliding 60 s `PathCountedHitCounter` feeds the binary threshold |
| selector data structure | O(n) linear scan | O(log N) min-heap (`LPBPriorityQueue`) |
| size denominator | `value.numel() + mamba_value.numel() * 1024` (heuristic) | per-pool `page_size_bytes` (exact) |
| gate | env `SGLANG_LPB_LRU=1`; default off | engine arg `hima_enabled=True` |
| LRU fallback | tie-break by `last_access_time` within selector loop | cold blocks score on `time.monotonic()` < hot threshold |

## Path A measured results (Qwen3.5-35B-A3B, TP=2, util=0.9, n=3 trials)

| metric | LRU | LPB | Δ | notes |
|---|---:|---:|---:|---|
| Phase G TTFT (ms, mean ± stddev across t1..t3) | 87.7 ± 42 | 86.0 ± 41 | tied | t1 was engine-warmup outlier (~136 ms vs 63-64 ms on t2/t3) |
| Phase G TTFT (t2/t3 only, post-warmup) | 63.5 ± 0.7 | 62.5 ± 0.7 | −1.6 % | tied within noise |
| **Phase H TTFT (ms, all 3 trials)** | **89.3 ± 9.5** | **149.0 ± 3.5** | **+66.8 %** | **LPB consistently slower** |
| Phase H swarm cached %                | 99.7 %    | 99.7 %    | tied | both modes have the anchor cached |
| Phase H full batch wall (ms)          | 198 ± 3   | 198 ± 1.5 | tied | decode dominates, prefill diff hidden |
| total wall (s)                        | 209.7 ± 17.4 | 209.0 ± 16.3 | tied | trial-1 outlier inflates both stddevs |

Per-trial breakdown:

| trial | LRU G | LPB G | LRU H | LPB H |
|---:|---:|---:|---:|---:|
| 1 | 136 | 133 | 100 | 153 |
| 2 |  63 |  62 |  86 | 147 |
| 3 |  64 |  63 |  82 | 147 |

Anchor survival (FINAL probe is unreliable for single-request mamba
probes — sglang doesn't populate `cached_tokens` in that path; the
batched-swarm `99.7 %` numbers above are the trustworthy anchor-state
signal):

| metric | LRU | LPB |
|---|---:|---:|
| Phase G swarm cached (per-trial) | 142 110 / 142 590 (99.7 %) | 142 110 / 142 590 (99.7 %) |
| Phase H swarm cached (per-trial) | 142 110 / 142 590 (99.7 %) | 142 110 / 142 590 (99.7 %) |

## Findings

1. **LPB is reproducibly slower than LRU on Phase H** at this
   configuration: **+60 ms per post-pressure swarm batch
   (+66.8 % TTFT)**, stddev ~3.5 ms across 3 trials. Phase G is
   tied.
2. **The reason: sglang's radix-tree LRU keeps the anchor cached
   through Phase F's pressure** at util=0.9, Phase F scale=10.
   Phase H batched swarm reports 99.7 % cached for **both** modes,
   every trial. So there is no anchor-protection benefit available
   for LPB to deliver — the cache is already "good enough" under
   LRU.
3. **What's left for LPB is just its overhead**: extra scoring work
   per eviction (O(n) scan over evictable mamba LRU list), and the
   per-node `_hit_times` deque maintenance during 500-anchor-warmup
   + heavy Phase F churn. That overhead manifests as the +60 ms
   on Phase H.
4. This is the **opposite** sign from vLLM Path A Phase H (LPB
   −12 % batch TTFT) — because vLLM's LRU **does** evict the
   anchor under our Phase F pressure at util=0.9, so LPB has
   real work to do there; sglang's LRU doesn't, so LPB just adds
   cost.

## Why the two engines behave differently on Phase H

| factor | vLLM | sglang |
|---|---|---|
| Eviction granularity | per-block (`FreeKVCacheBlockQueue`) | per-node in the radix tree |
| What gets bumped on a hit | nothing automatically; LRU is purely on block alloc/free order | the tree node's `last_access_time` |
| When Phase G fires | every anchor block has been alloc'd-and-released many times; recency among blocks is mixed | the anchor's tree node was just touched by Phase G, so it's MRU — Phase F doesn't push it out |
| Phase F's eviction pressure | hits the anchor blocks the moment they're cold relative to F's churn | clears decoys + cc-burst nodes first; tree-LRU naturally protects the high-traffic root |

In short, sglang's radix-tree LRU is **already** doing something
LPB-shaped, by accident — recency at the prefix-tree node level
behaves a lot like "protect the high-traffic prefix". So LPB on
sglang is mostly redundant on workloads where the hot prefix is
also the most-recently-traversed.

vLLM's per-block LRU has no such structural protection, so LPB's
explicit hit-count signal is what surfaces the anchor's importance.

## Correctness review

| concern | severity | sglang status | suggested fix |
|---|---|---|---|
| LPB scoring direction | none | correct (lowest priority evicts first) | — |
| Tie-break by recency | none | correct (`last_access_time` ordering on equal priority) | — |
| Mamba-only application (skips `evict_full`) | medium (scope) | by design for hybrid models | extend to `evict_full()` if non-hybrid models need LPB protection |
| `* 1024` size heuristic in `eviction_priority()` | medium (accuracy) | acknowledged as V0 placeholder by the code comment | read real per-pool byte size from `mamba_pool` config |
| Per-node `_hit_times` deque unbounded growth | low (memory drift under no-pressure) | unbounded in theory; bounded in practice because eviction prunes on visit | optional: cap deque size, drop oldest on overflow |
| O(n) selector vs heap | low–medium (the +60 ms Phase H cost is consistent with this) | bounded by `mamba_lru_list` size, but called per-block-alloc under pressure | switch to a heap; vLLM uses O(log N) |
| `cached_tokens` not populated for single-request mamba probes | low (only affects diagnostic; batched path is correct) | confirmed: serial Phase A/C probes report `cached=0` even with deep cache hits, but batched Phase G/H report it correctly | look up where the meta_info aggregation diverges between single and batched paths |

No correctness blockers. The Phase H +60 ms cost is the practical
consequence of two design choices: (a) sglang's LRU already
preserves hot prefixes structurally so LPB has nothing to add at
this op-point, and (b) the O(n) selector compounds the overhead
when many evictions fire.

## What would expose an LPB win on sglang

Two paths, both untested:

1. **Heavier cache pressure** — scale Phase F up until sglang's
   tree-LRU is forced to evict the anchor's node. Estimate from
   the cc-burst residue + decoy footprint: would likely need
   `--phase-f-scale 30` or higher (decoys ≥ KV budget). Risk: the
   workload becomes too big for the experiment to fit in a
   single-engine-load window.
2. **A workload where the hot prefix is NOT also recent** — e.g.
   warm the anchor in Phase A, then a *long* stretch of unrelated
   traffic that pushes the anchor's tree node down the LRU,
   *then* the post-pressure swarm. Sglang's LRU would evict the
   anchor by recency; sglang LPB would keep it by hit count.
   This isn't in our current pipeline (Phase E + F are short
   relative to the cc-burst).

For now, the honest summary is: **on workloads where sglang's
recency-LRU already protects the high-traffic prefix, LPB does
not help and costs ~60 ms per swarm.** Worth fixing the O(n)
selector before any production deployment.

## Repro

```bash
# Trials write directly to vllm-songyang/dev/aginfer/runs/sglang/
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

After this, the `prelude` branch on `rucnyz/sglang` is dormant —
still on the remote, but `HiMA` supersedes it. Delete with
`git push origin --delete prelude` when ready.
