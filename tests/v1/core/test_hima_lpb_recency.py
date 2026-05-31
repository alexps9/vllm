# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the LPB recency-aware eviction fix.

Pins the property that distinguishes the fixed design from the old
recency-blind two-tier: a *stale* hit-bearing block (whose windowed hit
count has decayed to 0) must be evicted BEFORE a *fresh* never-hit block,
and a *live* hit-bearing block must outlive both.

This is the exact inversion that made HiMA L1 lose hits versus plain LRU
on multi-turn agent traffic (verify/9): the freshly-generated conversation
tail (n_b==0 at the moment it is freed) was evicted while a long-idle
once-hit block stayed pinned. See dev/intralayer/vllm.md.
"""
from __future__ import annotations

import pytest

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue


class _FakeBlock:
    """Minimal stand-in for KVCacheBlock (only block_id is read here)."""

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.is_null = False


class _Clock:
    """Manually-advanced monotonic clock for deterministic window expiry."""

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


def test_stale_hit_evicts_before_fresh_unhit():
    clock = _Clock()
    window_s = 60.0
    rt = _make_runtime(clock, window_s)

    # Three blocks: S (will be hit then go stale), F (fresh, never hit),
    # L (live hit, kept warm).
    S, F, L = _FakeBlock(1), _FakeBlock(2), _FakeBlock(3)
    q = LPBFreeBlockQueue([S, F, L], runtime=rt, pool_kind=PoolKind.KV)

    # t=1000: S and L are hit (enter the protected/hit set). F never hit.
    rt.record_hit([S.block_id])
    rt.record_hit([L.block_id])
    # Re-place all three in the free queue with current scores.
    for b in (S, F, L):
        q.refresh_lpb_score(b)

    # Advance past the window so S's hit expires, but keep L warm by
    # re-hitting it just before the window closes.
    clock.advance(window_s - 1)
    rt.record_hit([L.block_id])
    q.refresh_lpb_score(L)
    clock.advance(2)  # now S's original hit (t=1000) is outside the window

    # Under pressure we pop one victim. It must be S (stale hit, oldest),
    # NOT F (fresh) and NOT L (live).
    victim = q.popleft()
    assert victim.block_id == S.block_id, (
        f"expected stale-hit S evicted first, got block {victim.block_id}"
    )

    # Next victim is F (fresh unhit) before L (still-live hit).
    victim2 = q.popleft()
    assert victim2.block_id == F.block_id, (
        f"expected fresh-unhit F before live-hit L, got {victim2.block_id}"
    )

    assert q.popleft().block_id == L.block_id


def test_unhit_blocks_evict_in_lru_order():
    """With no hits at all, eviction order is pure recency (LRU)."""
    clock = _Clock()
    rt = _make_runtime(clock, 60.0)
    blocks = [_FakeBlock(i) for i in range(4)]
    q = LPBFreeBlockQueue(blocks, runtime=rt, pool_kind=PoolKind.KV)
    # Append in order 0,1,2,3 (already in ctor); oldest-stamped evicts first.
    order = [q.popleft().block_id for _ in range(4)]
    assert order == [0, 1, 2, 3], order


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
