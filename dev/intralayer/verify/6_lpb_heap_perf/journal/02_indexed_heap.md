# 02 — Replace dataclass+lazy-delete with tuple+heapq+lazy-delete (2026-05-26)

## Hypothesis

cProfile baseline (journal/01) showed `_Entry.__lt__` dataclass
comparison taking **21 % of total time** (31 ms / 145 ms across 10 000
rotates). Replacing the `_Entry` dataclass with raw 3-tuples
`(score, seq, key)` lets the heap use C-level tuple comparison and
removes a per-comparison Python dispatch.

## Attempt A — Indexed binary heap (Sedgewick) — REVERTED

First tried a full Sedgewick-style indexed heap: array of tuples plus
`key → heap_index` map, custom `_sift_up` / `_sift_down` in Python,
no lazy deletion. Equivalence test passed. But microbench:

| | LRU | LPB rotate | LPB rotate+update |
|---|---:|---:|---:|
| ns/op | 313 | **5530** | **8064** |
| vs LRU | ref | 17.4× | 25.4× |

Rotate got *slightly worse* than the dataclass baseline (5065 → 5530)
because the Python `_sift_down` loop replaced C-level `_heapq.heappop`
+ dataclass `__lt__`. The Python sift turned out to be more expensive
than the dataclass dispatch. Reverted.

## Attempt B — Keep `heapq`, just swap dataclass for tuples — KEPT

Stayed with C-level `heapq.heappush`/`heapq.heappop`. Replaced
`_Entry` dataclass with raw `(score, seq, key)` tuples. Lazy delete
via a `_current_seq: dict[K, int]` map: a heap entry is valid iff
`current_seq[key] == entry.seq`. Pop loops skip stale entries. Also
added a `_score: dict[K, float]` so `score_of` is O(1) without walking
the heap.

`vllm/v1/core/hima/intra_pool/lpb_queue.py` rewritten end-to-end;
~125 LOC (down from 128). Public API unchanged. Equivalence test
(`test_indexed_heap_equiv.py`, 3 seeds × 20 000 random ops)
passes.

### Microbench (5 trials, N=8000 blocks, 10 000 rotates)

| metric | baseline | new | delta |
|---|---:|---:|---:|
| LPB rotate (ns/op) | 5065 ±79 | **2424 ±120** | **−52 %** |
| LPB rotate + 1 refresh per op (ns/op) | 9608 ±81 | **4463 ±51** | **−54 %** |
| LRU rotate | 313 ±3 | 326 ±6 | unchanged |
| LPB vs LRU ratio (rotate) | 16.2× | **7.4×** | halved |
| LPB vs LRU ratio (with update) | 30.7× | **13.7×** | halved |

### cProfile top after change (10 000 rotates, 0.079 s — was 0.145 s)

| line | tottime | before |
|---|---:|---:|
| `lpb_queue.add` | 12 ms | 21 ms |
| `_heapq.heappop` | 8 ms | 21 ms |
| `path_count.count` | 7 ms | 7 ms (unchanged) |
| `lpb_free_queue.append` | 7 ms | 8 ms |
| `lpb_free_queue.popleft` | 7 ms | 8 ms |
| `lpb_queue.popmin` | 6 ms | 6 ms |
| `_score_for` | 5 ms | 7 ms |
| `_heapq.heappush` | 3 ms | 5 ms |
| `<string>:__lt__` (dataclass) | **gone** | 31 ms |

The dataclass `__lt__` cost evaporated entirely. The remaining
distribution is now spread across: `path_count.count` (~9 % of total),
`_score_for` body (~6 %), `add` / `popmin` wrappers (~22 % combined),
and the C-level heappush/heappop themselves (~14 %). No single
dominant hot spot remains — further wins need either fewer ops
(stage 11c tiered) or per-call micro-cuts (stage 11d).

## Files touched

- `vllm/v1/core/hima/intra_pool/lpb_queue.py` (full rewrite)
- patch saved: `patches/02_indexed_heap.patch`
- equivalence test: `test_indexed_heap_equiv.py`

## RESULT

rotate **5065 → 2424 ns/op (−52 %)**, rotate+update **9608 → 4463 ns/op (−54 %)**.
Ratio vs LRU: rotate 16.2× → **7.4×**; rotate+update 30.7× → **13.7×**.

Target T1 (≤3× LRU on rotate) not yet hit. Need stage 11c (tiered) +
11d (_score_for opts) to close the remaining gap.
