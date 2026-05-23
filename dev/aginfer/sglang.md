# sglang LPB — implementation review (results pending)

The sglang-side LPB implementation that lives on `rucnyz/sglang@hima`
(squashed from `prelude`, with the `HPB`→`LPB` symbol rename
applied throughout).

See [`scenarios.md`](scenarios.md) for the engine-agnostic phase
design we'd want to also run here; see [`vllm.md`](vllm.md) for the
vLLM-side numbers that the sglang side should be compared against.

## Implementation

| component | file (`rucnyz/sglang@hima`) |
|---|---|
| L1 LPB (hits-per-byte, windowed) | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| Eviction selector | `_lpb_pick_mamba_eviction()` in the same file |
| Engine knob | env var `SGLANG_LPB_LRU=1` (default off → recency-LRU) |
| Window | env var `SGLANG_LPB_WINDOW_S=60.0` (default) |
| L2 inter-pool actuator | `python/sglang/srt/budgeter/` (cost_model, cross_pool_planner, fire_planner, pressure_adapter) |
| Eval framework | `dev/eval/main/` (genai-bench-based, INTRA=0/1, INTER=0/1 cells) |

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

| dimension | sglang LPB (`hima`) | vLLM HiMA L1 (`HiMA`) |
|---|---|---|
| where the queue lives | only `evict_mamba()` path | full `FreeKVCacheBlockQueue` replacement, all pools |
| score metric | `hits_in_window() / size_bytes` (continuous) | binary: cold = `time.monotonic()`, hot = `+ 1e12` |
| windowing | sliding 60 s deque per node | sliding 60 s `PathCountedHitCounter` feeds the binary threshold |
| selector data structure | O(n) linear scan | O(log N) min-heap (`LPBPriorityQueue`) |
| size denominator | `value.numel() + mamba_value.numel() * 1024` (heuristic) | per-pool `page_size_bytes` (exact) |
| gate | env `SGLANG_LPB_LRU=1`; default off | engine arg `hima_enabled=True`; threads through `CacheConfig` |
| LRU fallback | tie-break by `last_access_time` within selector loop | cold blocks score on `time.monotonic()` < hot threshold |

The two implementations agree on the spirit (protect heavily-hit
blocks, fall back to recency for the cold ones) via different
mechanics. For typical workloads with one dominantly-hot prefix
(e.g. agent-fleet anchor), the two should behave equivalently on
the headline metric — the binary-vs-continuous distinction only
matters when many blocks are mildly hot and need to be ordered
relative to each other.

## Correctness review

| concern | severity | sglang status | suggested fix |
|---|---|---|---|
| LPB scoring direction | none | correct (lowest priority evicts first) | — |
| Tie-break by recency | none | correct (`last_access_time` ordering on equal priority) | — |
| Mamba-only application (skips `evict_full`) | medium (scope) | by design for hybrid models | extend to `evict_full()` if non-hybrid models need LPB protection |
| `* 1024` size heuristic in `eviction_priority()` | medium (accuracy) | acknowledged as V0 placeholder by the code comment | read real per-pool byte size from `mamba_pool` config |
| Per-node `_hit_times` deque unbounded growth | low (memory drift under no-pressure) | unbounded in theory; bounded in practice because eviction prunes on visit | optional: cap deque size, drop oldest on overflow |
| O(n) selector vs heap | low (perf headroom) | bounded by `mamba_lru_list` size (~10 µs/call at typical sizes) | optional: heap if scale demands |

No correctness blockers. Two design quirks worth fixing before paper
or production: the `* 1024` size heuristic (replace with real
per-pool byte size) and the mamba-only scoping (extend to KV
eviction if needed). Neither blocks the headline measurement.

## Cross-check against existing committed runs

The hima branch already has many ablations under `dev/eval/runs/`.
The closest direct LRU vs LPB comparison I could find:

| run | cell | mean TTFT | output throughput |
|---|---|---:|---:|
| `v9-baseline-rerun`, trial1, Phase B | L1=0 L2=0 (stock LRU) | 160.93 ms | 120.34 tok/s |
| `v9-l1-isolate`,    trial1, Phase B | L1=1 L2=0 (LPB on)    | 161.40 ms | 120.33 tok/s |

Essentially tied. This is the **same shape** we saw in vLLM Path A
Phase G (the pre-pressure swarm control) — when LRU and LPB both
have the prefix cached, there's nothing for LPB to save.

The decisive **Phase H** (post-pressure swarm) measurement hasn't
been run on sglang yet. Without it, we can't confirm that the
sglang LPB implementation translates protection into TTFT savings
the way vLLM's does. The shape of the test we'd run, the expected
direction, and the order of magnitude are all in
[`scenarios.md`](scenarios.md) and [`vllm.md`](vllm.md).

## What's open

1. **Phase H–equivalent driver script for sglang.** Would either
   reuse `dev/eval/main/run_m1.sh` with custom workload, or write
   a new driver analogous to vLLM's `compare_lru_lpb.py` that
   sends the same A→B→G→E→F→H→C sequence to a sglang server.
2. **Two-sweep × n=3 trial run** on the same Qwen models we used
   for vLLM (Qwen3.5-35B-A3B, Qwen3.5-122B-A10B).
3. **Fix the `* 1024` size heuristic** before any results are
   reported as final — until then a workload that's sensitive to
   relative mamba vs KV byte cost would give biased numbers.

## Repro of the rename + squash that produced `rucnyz/sglang@hima`

```bash
cd /scratch/yuzhou/projects/sglang

# Change origin to the fork
git remote set-url origin https://github.com/rucnyz/sglang
git fetch origin prelude main

# Create hima from prelude, soft-reset to the merge-base with main
git checkout -b hima prelude
git reset --soft "$(git merge-base prelude origin/main)"

# Rename HPB → LPB across source + scripts + docs (134 matches → 0)
grep -rIlE "HPB|hpb" --include="*.py" --include="*.sh" --include="*.md" \
    python/ dev/ test/ | grep -v ".venv" | \
    xargs sed -i 's/HPB/LPB/g; s/hpb/lpb/g'

# Single squashed commit
git add -A
git commit -m "HiMA L1/L2 on sglang — squashed prelude with HPB→LPB rename"
git push origin hima
```

After this, the `prelude` branch on `rucnyz/sglang` is dormant —
still on the remote, but `hima` supersedes it. Delete with
`git push origin --delete prelude` when ready.
