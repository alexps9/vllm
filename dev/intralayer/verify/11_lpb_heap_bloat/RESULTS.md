# verify/11 — L1 LPB hot-heap bloat (#99): REAL, now FIXED

**Verdict: the bloat is real in L1's normal multi-turn-agent regime (not just a
latent/theoretical risk), and is now fixed by compaction.** Confirmed on the
**real** `LPBFreeBlockQueue` (not a model), reproduce: `probe.py`; raw:
[`runs/probe.out`](runs/probe.out).

## Origin
Raised by the interlayer `decision_cost` audit: the production
`LPBPriorityQueue` (L1's "hot" tier) is a lazy-delete heap with no compaction.
`add`/`update` push to `_heap`; `remove` pops only from `_current_seq` — the
`_heap` entry stays **stale**, trimmed only from the top by `peek`/`popmin`.

## Mechanism (default 'lazy' scoring — confirmed by code)
Per free/re-acquire **cycle** of a hot block: `append → _hot.add` (one heap
entry) then re-acquire `→ _hot.remove` (leaves that entry stale) = **+1
permanent stale leaf**, trimmed only when `popleft` reaches tier-3 and pops it.
In multi-turn agents, popular prefix blocks are freed+re-acquired every turn
while evictions are served from the **cold** tier first → the hot tier is rarely
drained → stale grows ~linearly with turns. (`eager` refresh is off by default,
and `refresh_lpb_score` on a request's own in-use blocks is a no-op, so this
cycle churn — not per-hit refresh — is the dominant source.)

## Measured (256 hot blocks, cold tier absorbs evictions)

**BEFORE (no compaction) — Regime A, hot tier not drained:**

| turns | hot `_heap` physical | logical | bloat | first hot-tier popleft |
|---:|---:|---:|---:|---:|
| 500 | 128,256 | 256 | 501× | 0.20 s |
| 1,000 | 256,256 | 256 | 1001× | 0.42 s |
| 2,000 | 512,256 | 256 | 2001× | 0.86 s |
| 4,000 | 1,024,256 | 256 | 4001× | **1.06 s** |

→ physical grows **exactly linearly with turns** (memory leak), and the first
hot-tier eviction after a quiet period pays **O(stale) = up to ~1 second** on
the scheduler hot path. Control (hot tier drained every 50 turns): bounded
(9.8×, 6.4 µs) — confirming it's the pop-light regime that bloats.

**AFTER (compaction, `_COMPACT_FACTOR=8`) — same Regime A:**

| turns | physical | logical | bloat | first hot-tier popleft |
|---:|---:|---:|---:|---:|
| 500 | 953 | 256 | 3.7× | 357 µs |
| 1,000 | 1,650 | 256 | 6.4× | 696 µs |
| 2,000 | 1,251 | 256 | 4.9× | 4.6 µs |
| 4,000 | 453 | 256 | 1.8× | 6.5 µs |

→ physical **bounded at ≤ ~8× logical** regardless of turns; the popmin spike
is gone (~µs). Memory leak and latency time-bomb both eliminated.

## Fix
`vllm/v1/core/hima/intra_pool/lpb_queue.py`: `_maybe_compact()` rebuilds `_heap`
from valid entries when `len(_heap) > _COMPACT_FACTOR × max(8, len(self))`,
called from `add`, `update`, **and `remove`**. **Behaviour-preserving** — it
drops only entries `popmin`/`peek` already skip, so heap order / eviction
decisions are unchanged; amortized O(1) per op; bounds `_heap` to `8× logical`.

> **Audit correction (agent `a141f6a0`).** The first cut compacted only in
> `add`/`update`. An adversarial audit found a real gap: `remove` shrinks the
> logical set without touching `_heap` and didn't trigger compaction, so a
> **remove-dominated drain** (an allocation burst re-acquiring many free hot
> blocks — a documented multi-turn shape) re-created the full #99 failure:
> add 200,001 → `remove` all but one → a single `popmin` took **208 ms** over
> 200k stale leaves. Fixed by also compacting in `remove` (the `max(8,…)` floor
> prevents thrash). After: same drain → heap **0**, popmin **19.5 µs**. The
> audit also confirmed (differential test, 200 trials × 400 mixed ops) the
> compaction is behaviour-equivalent and the heap invariant holds.

## Regression coverage (`tests/v1/core/test_hima_lpb_recency.py`)
- `test_lpb_pq_add_remove_churn_bounded_and_correct` — 5000×256 cycles, **non-
  uniform changing scores**, asserts an **absolute** mid-loop heap bound (not
  referencing `_COMPACT_FACTOR`, so a loosened factor is caught) + a **drain
  check** (popmin returns exactly the live keys in score order → catches a
  compaction that silently drops a live entry, which the old test could not).
- `test_lpb_pq_remove_dominated_no_bloat` — the remove-only drain path (catches
  the audit gap above).
- `test_lpb_pq_update_only_bounded` — the `update`/`_decay_hits` re-score path.
- The two existing recency-order tests still pass (behaviour unchanged). **5 passed.**
