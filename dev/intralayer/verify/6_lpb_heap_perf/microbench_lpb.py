"""Realistic microbench for LPBFreeBlockQueue vs FreeKVCacheBlockQueue.

Drives the same op pattern compare_lru_lpb.py produces (mix of popleft_n,
append_n, record_hit, refresh) and emits per-op nanoseconds + cProfile
hot lines. Used for Phase 11a baseline and 11e validation.

Usage:
    .venv/bin/python -u dev/intralayer/verify/1_l1_isolation_existing_tests/microbench_lpb.py [--profile]
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import statistics
import sys
import time

# Strip env so HiMA flags don't leak into the bench.
for k in ("VLLM_HIMA_ENABLE", "VLLM_HIMA_L1_ENABLE", "VLLM_HIMA_L2_ENABLE"):
    os.environ.pop(k, None)

sys.path.insert(0, "/data/yuzhou/projects/vllm-songyang")

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue
from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.integration import enable_runtime, disable_runtime

N_BLOCKS = 8000   # PathA util=0.9 KV pool
N_OPS = 10000     # generous to amortize startup noise
N_UPDATES_PER_OP = 0  # also try with 1 (matches refresh_lpb_score firing)
N_HOT_BLOCKS = 50  # blocks with n_b > 0 (anchor-like)
N_TRIALS = 10  # bumped from 5; perf claims require ≥3 trials mean ±sd


def make_blocks(n: int) -> list[KVCacheBlock]:
    """Construct vLLM blocks with the linked-list scaffold the LRU queue
    relies on (prev_free_block / next_free_block default to None)."""
    return [KVCacheBlock(block_id=i) for i in range(n)]


def bench_lru(n_blocks: int, n_ops: int) -> int:
    """Pure LRU FIFO rotate."""
    blocks = make_blocks(n_blocks)
    q = FreeKVCacheBlockQueue(blocks)
    t0 = time.perf_counter_ns()
    for _ in range(n_ops):
        b = q.popleft()
        q.append(b)
    return (time.perf_counter_ns() - t0) // n_ops


def bench_lpb(n_blocks: int, n_ops: int, hot_blocks: int) -> int:
    """LPB rotate. Bootstrap a runtime so _score_for has real path_counter
    + cost_curves to consult. Seed `hot_blocks` blocks with hits so the
    score path matches realistic Phase H usage."""
    cfg = HiMAConfig(hima_l1_enabled=True, hima_l2_enabled=False)
    rt = enable_runtime(config=cfg)
    try:
        blocks = make_blocks(n_blocks)
        q = LPBFreeBlockQueue(blocks, runtime=rt, pool_kind=PoolKind.KV)
        # Seed `hot_blocks` blocks with hits so most popmin returns cold
        # (mirrors real workload where anchor blocks are hot, swarm tails
        # are cold).
        hot_ids = list(range(hot_blocks))
        rt.record_hit(hot_ids)
        rt.record_hit(hot_ids)
        rt.record_hit(hot_ids)
        rt.register_lpb_queue(q)
        t0 = time.perf_counter_ns()
        for _ in range(n_ops):
            b = q.popleft()
            q.append(b)
        return (time.perf_counter_ns() - t0) // n_ops
    finally:
        disable_runtime()


def bench_lpb_with_update(
    n_blocks: int, n_ops: int, hot_blocks: int
) -> int:
    """LPB rotate + 1 update per op (matches refresh_lpb_score firing)."""
    cfg = HiMAConfig(hima_l1_enabled=True, hima_l2_enabled=False)
    rt = enable_runtime(config=cfg)
    try:
        blocks = make_blocks(n_blocks)
        q = LPBFreeBlockQueue(blocks, runtime=rt, pool_kind=PoolKind.KV)
        hot_ids = list(range(hot_blocks))
        rt.record_hit(hot_ids)
        rt.record_hit(hot_ids)
        rt.record_hit(hot_ids)
        rt.register_lpb_queue(q)
        t0 = time.perf_counter_ns()
        for i in range(n_ops):
            b = q.popleft()
            q.append(b)
            # Simulate refresh_lpb_score firing on a (hot) block that's in queue.
            target = blocks[(i * 17 + 7) % n_blocks]
            q.refresh_lpb_score(target)
        return (time.perf_counter_ns() - t0) // n_ops
    finally:
        disable_runtime()


def run_trials(fn, label, *args):
    samples = [fn(*args) for _ in range(N_TRIALS)]
    mean = statistics.mean(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0
    print(f"  {label:<38} {mean:>8.0f} ±{sd:>5.0f} ns/op  ({samples})")
    return mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Profile bench_lpb with cProfile and dump top 30 hot lines",
    )
    ap.add_argument(
        "--profile-with-update",
        action="store_true",
        help="Profile bench_lpb_with_update instead",
    )
    args = ap.parse_args()

    print(f"\nN_BLOCKS={N_BLOCKS}, N_OPS={N_OPS}, N_HOT={N_HOT_BLOCKS}, N_TRIALS={N_TRIALS}")
    print("-" * 78)
    lru_ns = run_trials(bench_lru, "LRU FreeKVCacheBlockQueue rotate", N_BLOCKS, N_OPS)
    lpb_ns = run_trials(
        bench_lpb, "LPB rotate (hot=50)", N_BLOCKS, N_OPS, N_HOT_BLOCKS
    )
    lpb_upd_ns = run_trials(
        bench_lpb_with_update,
        "LPB rotate + 1 refresh per op",
        N_BLOCKS, N_OPS, N_HOT_BLOCKS,
    )

    print(f"\nLPB overhead per op (rotate only):  {lpb_ns - lru_ns:>6.0f} ns ({lpb_ns / lru_ns:.1f}×)")
    print(f"LPB overhead per op (with update):  {lpb_upd_ns - lru_ns:>6.0f} ns ({lpb_upd_ns / lru_ns:.1f}×)")

    if args.profile or args.profile_with_update:
        print("\n=== cProfile top 30 ===")
        target_fn = bench_lpb_with_update if args.profile_with_update else bench_lpb
        pr = cProfile.Profile()
        pr.enable()
        target_fn(N_BLOCKS, N_OPS, N_HOT_BLOCKS)
        pr.disable()
        ps = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
        ps.print_stats(30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
