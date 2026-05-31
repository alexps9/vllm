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
import os
import time
from typing import TYPE_CHECKING

from vllm.v1.core.hima.config import PoolKind
from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = logging.getLogger(__name__)

# LPB scoring variant (verify/4 attribution knob). Selects how the eviction
# score is computed, to discriminate two suspected LPB scoring bugs:
#   - lazy refresh : score is only (re)computed when a block is appended to
#     the free queue; ``record_hit`` increments ``n_b`` but the heap score
#     stays stale until the next append.
#   - depth-as-integer : ``c_pool(depth)`` is fed the integer path index
#     (1..K) instead of a token count, so the cost curve barely varies and
#     LPB degenerates to LFU-on-hit-count.
#
#   "lazy"               current behaviour (default; reproduces both bugs)
#   "eager"              refresh the score on every record_hit
#   "depth_tokens"       feed c_pool(depth * block_size) instead of c_pool(depth)
#   "eager_depth_tokens" both fixes combined
_VALID_SCORING = {"lazy", "eager", "depth_tokens", "eager_depth_tokens"}
_SCORING_MODE = os.environ.get("VLLM_HIMA_LPB_SCORING", "lazy")
if _SCORING_MODE not in _VALID_SCORING:
    logger.warning(
        "[hima] unknown VLLM_HIMA_LPB_SCORING=%r; falling back to 'lazy' "
        "(valid: %s)", _SCORING_MODE, sorted(_VALID_SCORING),
    )
    _SCORING_MODE = "lazy"
_SCORING_LOGGED = False

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
        block_size: int = 1,
    ) -> None:
        # verify/4 scoring variant (see module-level _SCORING_MODE).
        self._eager = _SCORING_MODE in ("eager", "eager_depth_tokens")
        self._depth_tokens = _SCORING_MODE in ("depth_tokens", "eager_depth_tokens")
        # Tokens per prefix-tree depth unit; only used by the depth_tokens
        # variants to feed the cost curve a real token count instead of the
        # integer path index.
        self._block_size = max(1, block_size)
        global _SCORING_LOGGED
        if not _SCORING_LOGGED:
            logger.info(
                "[hima] LPB scoring variant: %s (eager=%s depth_tokens=%s, "
                "block_size=%d)",
                _SCORING_MODE, self._eager, self._depth_tokens, self._block_size,
            )
            _SCORING_LOGGED = True
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
        # --- analysis instrumentation (cheap; counts only) -----------------
        # Whether L1's hot-heap actually *engages*: popleft serves cold
        # (n_b==0, LRU-order) first and only evicts a hot (hit) block when
        # cold is exhausted. So ``_ev_hot`` > 0 means real pressure forced
        # eviction of a hit-bearing block — the regime where LPB ordering
        # can differ from LRU. ``_ev_hot==0`` for a whole run ⇒ L1 ≡ LRU on
        # that workload (the hot-heap never mattered). Logged every 2000
        # evictions at INFO so a run is analyzable post-hoc.
        self._ev_cold = 0
        self._ev_hot = 0
        self._ev_total = 0

    # ----------------------- legacy attribute ---------------------------- #

    @property
    def num_free_blocks(self) -> int:
        return self._cold.num_free_blocks + len(self._hot)

    # -------------------- legacy queue interface ------------------------- #

    def popleft(self) -> KVCacheBlock:
        if self._cold.num_free_blocks > 0:
            b = self._cold.popleft()
            del self._loc[b.block_id]
            self._note_evict(1, 0)
            return b
        if not self._hot.is_empty():
            block_id, _ = self._hot.popmin()
            del self._loc[block_id]
            self._note_evict(0, 1)
            return self._blocks_by_id[block_id]
        raise ValueError("No free blocks available")

    def _note_evict(self, n_cold: int, n_hot: int) -> None:
        """verify/9 analysis: track eviction source (cold FIFO vs hot LPB
        heap) so we can tell whether L1's hot-heap actually engaged."""
        self._ev_cold += n_cold
        self._ev_hot += n_hot
        self._ev_total += n_cold + n_hot
        if self._ev_total % 2000 == 0:
            logger.info(
                "[hima/lpb] evicts tot=%d cold=%d hot=%d (hot=%.1f%%) | "
                "free: cold=%d hot=%d",
                self._ev_total, self._ev_cold, self._ev_hot,
                100.0 * self._ev_hot / max(self._ev_total, 1),
                self._cold.num_free_blocks, len(self._hot),
            )

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
            self._note_evict(n, 0)
            return ret
        # Drain cold first, then take the remainder from hot.
        self._note_evict(cold_n, n - cold_n)
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

    def maybe_eager_refresh(self, block_id: int) -> None:
        """verify/4 'eager' variant: re-score ``block_id`` right after a hit
        was recorded against it (``n_b`` just incremented), so the hot-heap
        score doesn't go stale. No-op for the 'lazy' variants, and no-op if
        the block isn't currently in the free queue."""
        if not self._eager:
            return
        block = self._blocks_by_id.get(block_id)
        if block is not None:
            self.refresh_lpb_score(block)

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
            # depth_tokens variant feeds the cost curve a token count
            # (depth × block_size) instead of the integer path index, so
            # c_pool actually varies with prefix length. Memoised by depth
            # (block_size is fixed per queue).
            curve_arg = depth * self._block_size if self._depth_tokens else depth
            c = self._c_curve(curve_arg)
            cache[depth] = c
        # ``n_b`` is already int; Python int*float is a single C call, no
        # need for an explicit ``float(n_b)`` conversion.
        return _HIT_SCORE_OFFSET + n_b * c


__all__ = ["LPBFreeBlockQueue"]
