# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LPB-ordered drop-in replacement for vLLM's ``FreeKVCacheBlockQueue``.

Evicts the lowest-LPB block first (O(log N) via heap). Falls back to LRU on cold start.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vllm.v1.core.hima.config import PoolKind
from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = logging.getLogger(__name__)

# Offset that guarantees any block with n_b > 0 sorts STRICTLY ABOVE any
# block whose score is a wall-clock time.monotonic() (~1e9 on Linux).
# We want LPB ordering: cold/unused blocks evict first (lowest score),
# heavily-hit blocks survive (highest score). The two paths must not
# accidentally compare across scales.
_HIT_SCORE_OFFSET = 1e12


class LPBFreeBlockQueue:
    """Min-LPB-ordered free-block queue; drop-in for ``FreeKVCacheBlockQueue``.

    Score = n_b * c_i(depth) where n_b is the path-counted hit frequency and
    c_i(depth) is the pool's recovery cost curve. Falls back to monotonic time
    (LRU) when the runtime is unavailable or the block is cold.
    """

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        runtime: HiMARuntime | None = None,
        pool_kind: PoolKind = PoolKind.KV,
    ) -> None:
        self._blocks_by_id: dict[int, KVCacheBlock] = {b.block_id: b for b in blocks}
        self._queue: LPBPriorityQueue[int] = LPBPriorityQueue()
        self._runtime = runtime
        self.pool_kind = pool_kind
        # block_id → prefix-tree depth (set by coordinator on cache_blocks)
        self._block_depth: dict[int, int] = {}
        base = time.monotonic()
        for i, blk in enumerate(blocks):
            self._queue.add(blk.block_id, score=base + i * 1e-6)
        self._runtime = runtime

    # ----------------------- legacy attribute ---------------------------- #

    @property
    def num_free_blocks(self) -> int:
        return len(self._queue)

    # -------------------- legacy queue interface ------------------------- #

    def popleft(self) -> KVCacheBlock:
        if self._queue.is_empty():
            raise ValueError("No free blocks available")
        block_id, _ = self._queue.popmin()
        return self._blocks_by_id[block_id]

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if n > self.num_free_blocks:
            raise AssertionError(
                f"popleft_n({n}) but only {self.num_free_blocks} blocks are free"
            )
        popped = self._queue.popmin_n(n)
        return [self._blocks_by_id[bid] for bid, _ in popped]

    def append(self, block: KVCacheBlock) -> None:
        score = self._score_for(block)
        if block.block_id in self._queue:
            self._queue.update(block.block_id, score)
        else:
            self._queue.add(block.block_id, score=score)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for blk in blocks:
            self.append(blk)

    def remove(self, block: KVCacheBlock) -> None:
        if block.block_id not in self._queue:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        self._queue.remove(block.block_id)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return [self._blocks_by_id[bid] for bid in self._queue]

    # ------------------ HiMA-specific helpers (optional) ----------------- #

    def update_score(self, block: KVCacheBlock, score: float) -> None:
        """Re-score ``block``; no-op when not in the free queue."""
        if block.block_id in self._queue:
            self._queue.update(block.block_id, score)

    def score_of(self, block: KVCacheBlock) -> float:
        return self._queue.score_of(block.block_id)

    def set_block_depth(self, block_id: int, depth: int) -> None:
        """Record prefix-tree depth for LPB scoring; called by HiMACoordinator."""
        self._block_depth[block_id] = depth

    def refresh_lpb_score(self, block: KVCacheBlock) -> None:
        """Recompute and push the LPB score for ``block`` into the heap."""
        if block.block_id in self._queue:
            self._queue.update(block.block_id, self._score_for(block))

    # --------------------------- internals ------------------------------- #

    def _score_for(self, block: KVCacheBlock) -> float:
        """LPB eviction score.

        Layout (popmin = evict-first):

          * cold blocks (never hit while in cache) → `last_accessed`
            time.monotonic() (~1e9). Among cold, LRU still works.
          * hit blocks → ``_HIT_SCORE_OFFSET + n_b × c_pool(depth)``,
            which is *always* > any cold block's score, so cold evicts first.
            Within hit blocks, lower (hits × cost) evicts first (the actual
            HiMA LPB ordering from the paper).
        """
        rt = self._runtime
        if rt is None:
            last_access: float | None = getattr(block, "last_accessed", None)
            return float(last_access) if last_access is not None else time.monotonic()
        n_b = rt.path_counter.count(block.block_id)
        if n_b == 0:
            last_access = getattr(block, "last_accessed", None)
            return float(last_access) if last_access is not None else time.monotonic()
        depth = self._block_depth.get(block.block_id, 1)
        if self.pool_kind == PoolKind.KV:
            c = rt.cost_curves.c_kv_ms(depth)
        else:
            c = rt.cost_curves.c_m_ms(depth)
        return _HIT_SCORE_OFFSET + float(n_b) * c


__all__ = ["LPBFreeBlockQueue"]
