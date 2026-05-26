"""Smoke for the tiered LPBFreeBlockQueue.

Covers cold-only churn, cold→hot migration (refresh_lpb_score after
record_hit), hot→cold migration (window expiry), and remove from both
sides. Run from repo root:
    .venv/bin/python -u dev/intralayer/verify/6_lpb_heap_perf/test_tiered_wrapper.py
"""
from __future__ import annotations

import os
import sys
import time

for k in ("VLLM_HIMA_L1_ENABLE", "VLLM_HIMA_L2_ENABLE"):
    os.environ.pop(k, None)

sys.path.insert(0, "/data/yuzhou/projects/vllm-songyang")

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.integration import enable_runtime, disable_runtime
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue, _HIT_SCORE_OFFSET
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def fresh_queue(n=100):
    cfg = HiMAConfig(hima_l1_enabled=True)
    rt = enable_runtime(config=cfg)
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = LPBFreeBlockQueue(blocks, runtime=rt, pool_kind=PoolKind.KV)
    rt.register_lpb_queue(q)
    return q, blocks, rt


def case_cold_only():
    q, blocks, _rt = fresh_queue()
    # All cold by construction. popleft should return blocks in
    # FIFO order matching the underlying FreeKVCacheBlockQueue's
    # initialization order (block_id increasing).
    popped_ids = []
    for _ in range(10):
        b = q.popleft()
        popped_ids.append(b.block_id)
    assert popped_ids == list(range(10)), f"cold order broken: {popped_ids}"
    # popleft_n on cold-only path
    bulk = q.popleft_n(20)
    assert [b.block_id for b in bulk] == list(range(10, 30))
    # Re-append; should land back in cold and round-robin.
    for b in bulk + [blocks[i] for i in popped_ids]:
        q.append(b)
    assert q.num_free_blocks == 100, q.num_free_blocks
    disable_runtime()
    print("  case_cold_only: OK")


def case_cold_to_hot_migration():
    q, blocks, rt = fresh_queue()
    # Record hits on blocks 0..4 so they become "hot".
    rt.record_hit([0, 1, 2, 3, 4])
    rt.record_hit([0, 1, 2, 3, 4])
    # Pop blocks 0..4 (they're still cold in the queue — only the hit
    # *count* changed, not the queue tier).
    cold_pops = [q.popleft() for _ in range(5)]
    assert [b.block_id for b in cold_pops] == [0, 1, 2, 3, 4]
    # Re-append: these should be detected as hot (their _score_for is
    # > _HIT_SCORE_OFFSET) and land in the hot heap.
    for b in cold_pops:
        q.append(b)
    # 5 blocks should be hot now.
    n_hot = sum(1 for bid in q._loc if q._loc[bid] == 1)  # _LOC_HOT == 1
    assert n_hot == 5, f"expected 5 hot, got {n_hot}"
    # popleft should still pull cold first.
    next_b = q.popleft()
    assert next_b.block_id == 5, f"expected cold pop next, got {next_b.block_id}"
    # Drain remaining cold (blocks 5..99 minus the 0..4 that are hot)
    remaining_cold = q.popleft_n(94)
    assert all(b.block_id >= 5 for b in remaining_cold)
    # Now cold is empty; popleft should drain the hot heap.
    hot_pop = q.popleft()
    assert hot_pop.block_id in (0, 1, 2, 3, 4)
    disable_runtime()
    print("  case_cold_to_hot_migration: OK")


def case_refresh_migrates_back():
    """refresh_lpb_score on a hot block whose hits have decayed should
    migrate it back to cold."""
    cfg = HiMAConfig(hima_l1_enabled=True, hima_lpb_window_s=0.05)
    # Use a small window so hits expire quickly.
    rt = enable_runtime(config=cfg)
    blocks = [KVCacheBlock(i) for i in range(50)]
    q = LPBFreeBlockQueue(blocks, runtime=rt, pool_kind=PoolKind.KV)
    rt.register_lpb_queue(q)

    # Make block 0 hot.
    rt.record_hit([0])
    rt.record_hit([0])
    b0 = q.popleft()  # block 0 (still cold in queue ordering)
    q.append(b0)
    assert q._loc[0] == 1, "block 0 should be hot after re-append with hits"

    # Sleep past the window so hits expire.
    time.sleep(0.1)

    # refresh_lpb_score recomputes; the now-zero hit count drops it cold.
    q.refresh_lpb_score(b0)
    assert q._loc[0] == 0, f"block 0 should be cold after refresh; loc={q._loc[0]}"
    disable_runtime()
    print("  case_refresh_migrates_back: OK")


def case_remove_routes_correctly():
    q, blocks, rt = fresh_queue()
    rt.record_hit([0, 1])
    rt.record_hit([0, 1])
    # Make blocks 0, 1 hot.
    for bid in (0, 1):
        q.append(q.popleft())  # ugly — pop the head and re-append... wait,
        # popleft pops the head (which is the lowest cold). Re-appending
        # is fine: it just appends to cold or hot based on score.
    # Better: just remove and re-add blocks 0, 1 explicitly.
    # Reset state.
    while q.num_free_blocks > 0:
        q.popleft()
    for b in blocks:
        q.append(b)
    # Now record hits and re-append 0 to make it hot.
    q.remove(blocks[0])
    q.append(blocks[0])  # cold (no hits yet for 0 in path_counter — hits were lost when we drained)
    # remove from cold should work.
    q.remove(blocks[0])
    assert 0 not in q._loc
    # Add it back; now record hits.
    rt.record_hit([0])
    rt.record_hit([0])
    q.append(blocks[0])
    assert q._loc[0] == 1, "expected hot"
    # remove from hot should work.
    q.remove(blocks[0])
    assert 0 not in q._loc
    disable_runtime()
    print("  case_remove_routes_correctly: OK")


def main():
    case_cold_only()
    case_cold_to_hot_migration()
    case_refresh_migrates_back()
    case_remove_routes_correctly()
    print("ALL tiered-wrapper cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
