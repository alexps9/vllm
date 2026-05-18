# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LPB-ordered drop-in replacement for vLLM's ``FreeKVCacheBlockQueue``.

Evicts the lowest-LPB block first (O(log N) via heap). Falls back to LRU on cold start.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = logging.getLogger(__name__)


class LPBFreeBlockQueue:
    """Min-LPB-ordered free-block queue; drop-in for ``FreeKVCacheBlockQueue``."""

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        runtime: object | None = None,
    ) -> None:
        self._blocks_by_id: dict[int, KVCacheBlock] = {b.block_id: b for b in blocks}
        self._queue: LPBPriorityQueue[int] = LPBPriorityQueue()
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

    # --------------------------- internals ------------------------------- #

    def _score_for(self, block: KVCacheBlock) -> float:
        last_access: float | None = getattr(block, "last_accessed", None)
        if last_access is None:
            last_access = time.monotonic()
        return float(last_access)


__all__ = ["LPBFreeBlockQueue"]
