# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LPB-ordered drop-in replacement for vLLM's ``FreeKVCacheBlockQueue``.

Internally a two-tier structure:

* **cold queue** — vLLM-style ``FreeKVCacheBlockQueue`` (hand-tuned
  doubly-linked list, O(1) every op, zero allocation per call). Holds
  every block whose current score is the wall-clock LRU fallback
  (i.e. ``n_b == 0``).
* **hot heap** — ``LPBPriorityQueue`` (tuple-based heapq + lazy delete).
  Holds every block whose score is ``_HIT_SCORE_OFFSET + n_b × c``.

Because the offset is 1e12 (≫ any wall-clock score), the hot heap's
minimum is always strictly greater than any cold block's score. So
``popleft`` can always serve from cold first and only touch the heap
when cold is empty — which is the rare path for typical workloads.
That eliminates ~all heap overhead on the common alloc/free cycle.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vllm.v1.core.hima.config import PoolKind
from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

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

# Location sentinel for the per-block ``_loc`` map.
_LOC_COLD = 0
_LOC_HOT = 1


class LPBFreeBlockQueue:
    """Min-LPB-ordered free-block queue; drop-in for ``FreeKVCacheBlockQueue``.

    Internally split into a fast cold FIFO and a hot LPB heap (see module
    docstring). Public API mirrors ``FreeKVCacheBlockQueue`` plus a few
    HiMA-specific helpers (``set_block_depth`` / ``refresh_lpb_score``).
    """

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        runtime: HiMARuntime | None = None,
        pool_kind: PoolKind = PoolKind.KV,
    ) -> None:
        self._blocks_by_id: dict[int, KVCacheBlock] = {b.block_id: b for b in blocks}
        # Cold queue starts with every block (none have hits yet).
        self._cold = FreeKVCacheBlockQueue(blocks)
        # Hot heap is empty until something records a hit on a block that
        # later gets re-appended to the free queue.
        self._hot: LPBPriorityQueue[int] = LPBPriorityQueue()
        # block_id → current location sentinel. Used for O(1) remove
        # routing without searching both structures.
        self._loc: dict[int, int] = {bid: _LOC_COLD for bid in self._blocks_by_id}
        self._runtime = runtime
        self.pool_kind = pool_kind
        # block_id → prefix-tree depth (set by coordinator on cache_blocks)
        self._block_depth: dict[int, int] = {}
        # Hot-loop attribute caches; bound on first ``_score_for`` call
        # after ``_runtime`` is available.
        self._pc_count = None  # bound method: path_counter.count
        self._c_curve = None  # bound method: c_kv_ms or c_m_ms
        # Memoise ``c_curve(depth)``; depths are small integers (1..~30).
        self._c_cache: dict[int, float] = {}

    # ----------------------- legacy attribute ---------------------------- #

    @property
    def num_free_blocks(self) -> int:
        return self._cold.num_free_blocks + len(self._hot)

    # -------------------- legacy queue interface ------------------------- #

    def popleft(self) -> KVCacheBlock:
        if self._cold.num_free_blocks > 0:
            b = self._cold.popleft()
            del self._loc[b.block_id]
            return b
        if not self._hot.is_empty():
            block_id, _ = self._hot.popmin()
            del self._loc[block_id]
            return self._blocks_by_id[block_id]
        raise ValueError("No free blocks available")

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        cold_n = self._cold.num_free_blocks
        if n > cold_n + len(self._hot):
            raise AssertionError(
                f"popleft_n({n}) but only {cold_n + len(self._hot)} blocks are free"
            )
        if n <= cold_n:
            ret = self._cold.popleft_n(n)
            loc = self._loc
            for b in ret:
                del loc[b.block_id]
            return ret
        # Drain cold first, then take the remainder from hot.
        ret = self._cold.popleft_n(cold_n)
        loc = self._loc
        for b in ret:
            del loc[b.block_id]
        remainder = n - cold_n
        popped = self._hot.popmin_n(remainder)
        blocks_by_id = self._blocks_by_id
        for bid, _ in popped:
            ret.append(blocks_by_id[bid])
            del loc[bid]
        return ret

    def append(self, block: KVCacheBlock) -> None:
        score = self._score_for(block)
        if score < _HIT_SCORE_OFFSET:
            # Cold: goes to the fast FIFO. Block must not currently be in
            # either queue — block_pool guarantees this by ref-count.
            self._cold.append(block)
            self._loc[block.block_id] = _LOC_COLD
        else:
            self._hot.add(block.block_id, score)
            self._loc[block.block_id] = _LOC_HOT

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for blk in blocks:
            self.append(blk)

    def remove(self, block: KVCacheBlock) -> None:
        loc = self._loc.pop(block.block_id, None)
        if loc is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        if loc == _LOC_COLD:
            self._cold.remove(block)
        else:
            self._hot.remove(block.block_id)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        # Cold side: iterate the doubly-linked list head→tail; hot side:
        # iterate the heap dict. Both yield block ids only; map via
        # ``_blocks_by_id``.
        blocks_by_id = self._blocks_by_id
        out: list[KVCacheBlock] = []
        # Cold FIFO doesn't expose iteration; walk the linked list directly.
        cur = self._cold.fake_free_list_head.next_free_block
        tail = self._cold.fake_free_list_tail
        while cur is not None and cur is not tail:
            out.append(cur)
            cur = cur.next_free_block
        for bid in self._hot:
            out.append(blocks_by_id[bid])
        return out

    # ------------------ HiMA-specific helpers (optional) ----------------- #

    def update_score(self, block: KVCacheBlock, score: float) -> None:
        """Re-score ``block``; migrate across cold/hot if needed."""
        loc = self._loc.get(block.block_id)
        if loc is None:
            return
        if score < _HIT_SCORE_OFFSET:
            if loc == _LOC_HOT:
                # Migrate hot → cold.
                self._hot.remove(block.block_id)
                self._cold.append(block)
                self._loc[block.block_id] = _LOC_COLD
            # If already cold, the FIFO doesn't track scores — no-op.
        else:
            if loc == _LOC_COLD:
                # Migrate cold → hot.
                self._cold.remove(block)
                self._hot.add(block.block_id, score)
                self._loc[block.block_id] = _LOC_HOT
            else:
                self._hot.update(block.block_id, score)

    def score_of(self, block: KVCacheBlock) -> float:
        loc = self._loc.get(block.block_id)
        if loc is None:
            raise KeyError(block.block_id)
        if loc == _LOC_HOT:
            return self._hot.score_of(block.block_id)
        # Cold FIFO doesn't track per-block scores; return wall-clock-ish
        # approximation so callers get a deterministic monotonic value.
        return time.monotonic()

    def set_block_depth(self, block_id: int, depth: int) -> None:
        """Record prefix-tree depth for LPB scoring; called by HiMACoordinator."""
        self._block_depth[block_id] = depth

    def refresh_lpb_score(self, block: KVCacheBlock) -> None:
        """Recompute the LPB score for ``block``; re-place if it crossed tier."""
        loc = self._loc.get(block.block_id)
        if loc is None:
            return
        self.update_score(block, self._score_for(block))

    # --------------------------- internals ------------------------------- #

    def _score_for(self, block: KVCacheBlock) -> float:
        """LPB eviction score.

        Layout (popmin = evict-first):

          * cold blocks (never hit while in cache) → ``time.monotonic()``
            (~1e9). Among cold, FIFO order from the underlying doubly-
            linked list does the LRU job.
          * hit blocks → ``_HIT_SCORE_OFFSET + n_b × c_pool(depth)``,
            which is *always* > any cold block's score, so cold evicts
            first. Within hot, lower (hits × cost) evicts first.
        """
        rt = self._runtime
        if rt is None:
            return time.monotonic()
        pc_count = self._pc_count
        if pc_count is None:
            pc_count = rt.path_counter.count
            self._pc_count = pc_count
            curves = rt.cost_curves
            self._c_curve = (
                curves.c_kv_ms if self.pool_kind == PoolKind.KV else curves.c_m_ms
            )
        bid = block.block_id
        n_b = pc_count(bid)
        if n_b == 0:
            return time.monotonic()
        depth = self._block_depth.get(bid, 1)
        cache = self._c_cache
        c = cache.get(depth)
        if c is None:
            c = self._c_curve(depth)
            cache[depth] = c
        # ``n_b`` is already int; Python int*float is a single C call, no
        # need for an explicit ``float(n_b)`` conversion.
        return _HIT_SCORE_OFFSET + n_b * c


__all__ = ["LPBFreeBlockQueue"]
