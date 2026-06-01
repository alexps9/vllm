# SPDX-License-Identifier: Apache-2.0
"""verify/11 — does L1's LPB hot heap (LPBPriorityQueue) bloat in REAL usage?

Task #99 (raised by the interlayer decision_cost audit): the production
`LPBPriorityQueue` is a lazy-delete heap with no compaction. `remove`/`update`
leave stale leaves in `_heap`; they are trimmed ONLY by `peek`/`popmin`, and
only from the top. The question is whether L1's real update:pop ratio makes it
bloat — verified here on the REAL `LPBFreeBlockQueue`, not a model.

Mechanism (confirmed by reading the code, default 'lazy' scoring):
  - a hot block freed -> `append` -> `_hot.add` (one heap entry);
  - re-acquired (cache hit, ref 0->1) -> `block_pool` calls `remove` ->
    `_hot.remove` -> pops it from `_current_seq` but leaves the entry stale in
    `_heap`;
  - so each free/re-acquire CYCLE of a hot block = +1 permanent stale leaf,
    trimmed only when `popleft` reaches tier-3 (the hot heap) and pops it.
  - In multi-turn agents, popular prefix blocks are freed/re-acquired every
    turn, while evictions are served from the cold tier first -> the hot tier
    is rarely drained -> stale grows ~linearly with turns.

This drives the real queue and measures `_hot._heap` (physical) vs `len(_hot)`
(logical) over turns, plus the latency of the popleft that finally drains the
hot tier (the O(stale) spike). Regime A = realistic (cold absorbs evictions);
Regime B = control (hot tier drained periodically -> should stay bounded).

Run: .venv/bin/python dev/intralayer/verify/11_lpb_heap_bloat/probe.py
"""

from __future__ import annotations

import json
import time

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue


class _FakeBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.is_null = False


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_runtime(clock: _Clock, window_s: float):
    from vllm.v1.core.hima.integration import HiMARuntime
    from vllm.v1.core.hima.intra_pool.cost_curve import LEGACY_DEFAULT
    from vllm.v1.core.hima.intra_pool.path_count import PathCountedHitCounter

    pc = PathCountedHitCounter(window_seconds=window_s, clock=clock)
    return HiMARuntime(
        config=HiMAConfig(hima_l1_enabled=True, hima_lpb_window_s=window_s),
        cost_curves=LEGACY_DEFAULT,
        path_counter=pc,
    )


def run(n_hot: int, n_cold: int, turns: int, drain_hot_every: int,
        window_s: float = 3600.0) -> dict:
    """drain_hot_every=0 => never drain hot tier (realistic agent regime);
    >0 => popleft enough to reach tier-3 every N turns (control)."""
    clock = _Clock()
    rt = _make_runtime(clock, window_s)
    hot = [_FakeBlock(i) for i in range(n_hot)]
    cold = [_FakeBlock(n_hot + i) for i in range(n_cold)]
    q = LPBFreeBlockQueue(hot + cold, runtime=rt, pool_kind=PoolKind.KV)

    # Make the hot blocks hot: record a hit + refresh so they move to the hot
    # heap (priority = n_b * c_pool(depth) > 0).
    for b in hot:
        rt.record_hit([b.block_id])
        q.refresh_lpb_score(b)

    phys_trace = []
    # warm cold tier stays populated (cold blocks remain free for eviction)
    for t in range(turns):
        # multi-turn churn: each hot block re-acquired (remove) then freed
        # again (append), staying hot (re-hit within window).
        for b in hot:
            q.remove(b)              # cache-hit re-acquire -> stale leaf
            rt.record_hit([b.block_id])
            q.append(b)              # freed again -> new hot.add
        clock.advance(2.0)           # small step; stays within window

        if drain_hot_every and (t + 1) % drain_hot_every == 0:
            # control: evict enough to drain cold + reach the hot tier, then
            # re-free them (so the hot heap gets popmin'd -> stale trimmed).
            need = q.num_free_blocks
            popped = q.popleft_n(need)
            for b in popped:
                rt.record_hit([b.block_id])
                q.append(b)
        if t % 50 == 0:
            phys_trace.append((t, q._hot.heap_physical_len()
                               if hasattr(q._hot, "heap_physical_len")
                               else len(q._hot._heap), len(q._hot)))

    physical = len(q._hot._heap)
    logical = len(q._hot)
    # measure the latency of the popleft that finally drains the hot tier
    # (pops through all accumulated stale). Drain cold first so popleft hits
    # tier-3.
    while q._cold.num_free_blocks > 0 or q._evict_first:
        q.popleft()
    t0 = time.perf_counter_ns()
    if q.num_free_blocks > 0:
        q.popleft()                  # first hot-tier pop -> trims O(stale)
    drain_first_pop_us = (time.perf_counter_ns() - t0) / 1000.0

    return {
        "n_hot": n_hot, "turns": turns, "drain_hot_every": drain_hot_every,
        "hot_heap_physical": physical,
        "hot_heap_logical": logical,
        "bloat_x": round(physical / max(1, logical), 1),
        "first_hot_pop_us": round(drain_first_pop_us, 1),
        "phys_trace_(turn,phys,logical)": phys_trace[::max(1, len(phys_trace)//6)],
    }


def main() -> None:
    print("=== verify/11: L1 LPB hot-heap bloat under real usage ===\n")
    print("--- Regime A: realistic agent (hot blocks churn; hot tier NOT drained) ---")
    for turns in (500, 1000, 2000, 4000):
        r = run(n_hot=256, n_cold=1024, turns=turns, drain_hot_every=0)
        print(json.dumps({k: r[k] for k in
                          ("turns", "hot_heap_physical", "hot_heap_logical",
                           "bloat_x", "first_hot_pop_us")}))
    print("  (if physical grows ~linearly with turns while logical stays ~n_hot,")
    print("   the hot heap bloats in real L1 usage -> #99 is real, not latent)\n")

    print("--- Regime B: control (hot tier drained every 50 turns) ---")
    r = run(n_hot=256, n_cold=1024, turns=4000, drain_hot_every=50)
    print(json.dumps({k: r[k] for k in
                      ("turns", "hot_heap_physical", "hot_heap_logical",
                       "bloat_x", "first_hot_pop_us")}))
    print("  (periodic hot-tier popmin should keep physical bounded)")


if __name__ == "__main__":
    main()
