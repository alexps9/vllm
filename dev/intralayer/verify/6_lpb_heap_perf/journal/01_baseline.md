# 01 — Baseline profile (2026-05-26)

## Setup

```
N_BLOCKS = 8000   (PathA util=0.9 KV pool)
N_OPS = 10000     (popleft + append rotate)
N_HOT_BLOCKS = 50 (seeded with 3 record_hit() calls so n_b=3)
N_TRIALS = 5
```

`microbench_lpb.py` (this folder) drives:
- `bench_lru` — FreeKVCacheBlockQueue rotate
- `bench_lpb` — LPBFreeBlockQueue rotate with HiMA runtime active
- `bench_lpb_with_update` — same + 1 refresh_lpb_score per op (matches
  the real per-cache_blocks rate observed in compare_lru_lpb)

## Numbers

| op | per-op ns | std | vs LRU |
|---|---:|---:|---:|
| LRU rotate | 313 | ±3 | ref |
| LPB rotate | 5065 | ±79 | 16.2× |
| LPB rotate + 1 refresh | 9608 | ±81 | 30.7× |

## cProfile top hot lines (LPB rotate, 10 000 ops, 0.145 s total)

| line | tottime | % of total |
|---|---:|---:|
| `<string>:2(__lt__)` (dataclass comparison) | **31 ms** | **21 %** |
| `_heapq.heappop` (C builtin) | 21 ms | 14 % |
| `lpb_queue.add` (Python wrapper) | 21 ms | 14 % |
| `lpb_queue.popmin` (Python wrapper) | 6 ms | 4 % |
| `lpb_free_queue.popleft` wrapper | 8 ms | 6 % |
| `lpb_free_queue.append` wrapper | 8 ms | 6 % |
| `_score_for` body | 7 ms | 5 % |
| `path_count.count` | 7 ms | 5 % |
| `_heapq.heappush` (C builtin) | 5 ms | 3 % |

The dataclass `__lt__` is the biggest single cost. `_Entry` is
`@dataclass(order=True)` so Python generates `__lt__` that creates a
tuple `(self.score, self.seq)` per comparison and compares — extra
allocation + dispatch on every heap step.

**Replacing `_Entry` with raw tuples** `(score, seq, key)` would push
the comparison into C-level tuple comparison and eliminate the dunder
overhead entirely. This is a free win for stage 11b.

## Secondary observations

- `add()` does ~3× more work than `heappush` itself: most of it is the
  Python attribute setup for `_Entry`.
- `_score_for` is only 5 % of total — `_score_for` micro-opts (stage
  11d) have limited ceiling unless we also reduce call count.
- The `_purge`/`_evict_expired` loops barely register (2 ms each) —
  not bottlenecks at the current call count.

## What this changes about the plan

Stage **11b (indexed heap)** is the highest-leverage change. The first
sub-step inside 11b should be: convert `_Entry` to raw tuple
`(score, seq, key)` and see how much of the dataclass overhead
evaporates. *That alone* should pull LPB from 16× → ~8× LRU.

Stage **11c (tiered eviction)** is still important for util=0.9 PathA
because most pops will hit the cold path and bypass the heap entirely.
But staging 11b first lets us measure 11c's incremental impact
cleanly.

Stage **11d** is now lower priority but still useful — there's a
non-trivial portion of time in `path_count.count` (5 %), most of
which is the always-firing `_evict_expired` no-op loop.
