# 03 — Tiered cold-FIFO + hot-heap eviction (2026-05-26)

## Hypothesis

`_HIT_SCORE_OFFSET = 1e12` guarantees every hot block (n_b > 0) sorts
strictly above every cold block (score = `time.monotonic()` ≈ 1e9). So
the heap's minimum is *always* a cold block when any cold exists. Heap
ops on cold blocks are wasted work — they would be served identically
by a fast FIFO.

Replace `LPBFreeBlockQueue`'s single LPB heap with **two structures**:

- **cold queue** = `FreeKVCacheBlockQueue` (vLLM's hand-tuned
  doubly-linked list; O(1) per op; zero allocation)
- **hot heap** = `LPBPriorityQueue` (from journal/02, tuple+heapq+lazy)

At `append()`, route by score: `score < 1e12` → cold, else hot.
At `popleft()`, drain cold first, fall through to hot only when cold
empty. Maintain `_loc: dict[block_id, int]` so `remove()` knows which
side to touch.

## Implementation

`vllm/v1/core/hima/lpb_free_queue.py` rewritten (171 → 245 LOC).
Key surfaces:

- `popleft`: ~3 lines fast path on the cold side
- `popleft_n`: drain cold first, take remainder from hot (so it can
  serve any n ≤ total)
- `append`: branch on `_HIT_SCORE_OFFSET`, route to one side
- `update_score` + `refresh_lpb_score`: handle the rare cold↔hot
  migration when a block's score crosses the boundary (rare because
  scores are recomputed only on append or explicit refresh)
- `_loc` map: 2-state sentinel (`_LOC_COLD=0`, `_LOC_HOT=1`) so
  `remove` is O(1) and doesn't have to probe both structures

Equivalence test (`test_indexed_heap_equiv.py`) covers the internal
`LPBPriorityQueue` only; `LPBFreeBlockQueue` tier routing is checked
by the e2e PathA validation (stage 11e).

## Microbench (n=10 trials)

| metric | journal/02 (1-tier) | journal/03 (2-tier) | delta |
|---|---:|---:|---:|
| LPB rotate (ns/op) | 2424 ±120 | **913 ±15** | **−62 %** |
| LPB rotate + 1 refresh per op (ns/op) | 4463 ±51 | **1582 ±9** | **−65 %** |
| LRU rotate baseline | 313-326 | 327 ±12 | unchanged |
| LPB / LRU ratio (rotate) | 7.4× | **2.8×** ✓ T1 met | −62 % |
| LPB / LRU ratio (with update) | 13.7× | **4.8×** | −65 % |

**Target T1 (≤ 3× LRU on rotate): MET.** With update the ratio is
4.8× — over the goal but well within the realistic envelope; the
update path still touches the heap whenever the target block is hot
(N_HOT/N_BLOCKS = 50/8000 = 0.6 % of bench updates).

## cProfile after the tiering (10 000 ops, 39 ms)

| line | tottime | % |
|---|---:|---:|
| `lpb_free_queue.append` wrapper | 6 ms | 15 % |
| `_score_for` body | 4 ms | 10 % |
| `path_count.count` | 4 ms | 10 % |
| `lpb_free_queue.popleft` wrapper | 3 ms | 8 % |
| `kv_cache_utils.popleft` (cold FIFO) | 3 ms | 8 % |
| `kv_cache_utils.append` (cold FIFO) | 2 ms | 5 % |
| `time.monotonic` | 2 ms | 5 % |

Heap-related lines (`lpb_queue.add`, `_heapq.heappop`) dropped out of
the top-30 entirely — heap ops are now ~5 % of total time vs 35 %
before. `_score_for` + `path_count.count` are the remaining
non-LRU cost (every append still scores the block to route it). Further
micro-cuts plausible but diminishing returns — stage 11d (subagent)
covers them.

## Files touched

- `vllm/v1/core/hima/lpb_free_queue.py` (full rewrite)
- patch saved: `patches/03_tiered_eviction.patch`

## RESULT

rotate **2424 → 913 ns/op (−62 %)**, rotate+update **4463 → 1582 ns/op (−65 %)**.
Cumulative across stages 11b + 11c: rotate **5065 → 913 ns/op (−82 %)**.
vs LRU ratio (rotate): 16.2× → **2.8×**. **Target T1 met.**

## Next

- 11d subagent finalises microopts on `_score_for` / `path_count.count`
  (journal/04 forthcoming).
- 11e (e2e validation): rerun compare_lru_lpb PathA util=0.9 n=3
  --mode l1_only. Predicted PhaseH TTFT: very close to LRU's 370 ms.
