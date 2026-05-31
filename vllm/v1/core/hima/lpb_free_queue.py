# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recency-aware LPB free-block queue; drop-in for ``FreeKVCacheBlockQueue``.

Three tiers, evicted in this order (lowest value first):

1. **evict-first set** — blocks whose windowed hits have *expired* (demoted
   from the hot heap). They were hit once but went stale; they belong ahead
   of fresh never-hit blocks. Order among them doesn't matter (all stale),
   so a ``set`` gives O(1) add/remove/pop.
2. **cold FIFO** — never-hit blocks (``priority == 0``), in
   ``FreeKVCacheBlockQueue`` recency order (oldest at head). O(1) per op;
   this is the common case for cold-flow traffic.
3. **hot heap** — blocks carrying an in-window hit, keyed by the tuple
   ``(priority, recency)`` and popped minimum-first. Only touched when tiers
   1+2 are exhausted (real eviction pressure on hit-bearing blocks).

   * ``priority = n_b_windowed × c_pool(depth)`` — value of keeping the block.
   * ``recency`` — monotonic access stamp; tie-break so a *recently* re-hit
     block out-survives a *stale* one of equal hit-value.

Why recency is first-class (the verify/9 fix)
---------------------------------------------
The previous design scored hit blocks ``1e12 + n_b×c`` with **no recency**
and **never demoted** stale hits. A block hit once long ago out-ranked every
freshly freed block forever (within the window). On multi-turn agent traffic
the freshly-generated conversation tail is ``n_b==0`` when freed, so it was
evicted before a long-idle once-hit block that plain LRU would have dropped
first — costing L1 hits LRU kept (verify/9: SWE-Bench conc=256, L1 cached
1.5% vs LRU 3.5%, n=3).

``_decay_hits`` (throttled to once per ``_decay_interval`` of clock time)
migrates a hit block whose hits have aged out of the window into the
evict-first tier, so it is dropped *before* the fresh tails — exactly like
LRU — while a *live* re-hit block stays protected. This mirrors sglang's LPB
(``n_hits/bytes`` priority + ``last_access_time`` tie-break + a real decay
window); see ``dev/intralayer/sglang.md`` and ``dev/intralayer/vllm.md``.
Keeping the O(1) cold FIFO for the common case keeps per-op cost ~3× LRU
(verify/6 microbench), versus ~8× for a single all-blocks heap.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from vllm.v1.core.hima.config import PoolKind
from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = logging.getLogger(__name__)

# LPB scoring variant (verify/4 attribution knob). With the recency-aware
# design the two suspected bugs are moot; kept as a diagnostic:
#   "lazy"               recency stamped on free/refresh (default)
#   "eager"              additionally re-stamp recency on every record_hit
#   "depth_tokens"       feed c_pool(depth * block_size) instead of c_pool(depth)
#   "eager_depth_tokens" both
_VALID_SCORING = {"lazy", "eager", "depth_tokens", "eager_depth_tokens"}
_SCORING_MODE = os.environ.get("VLLM_HIMA_LPB_SCORING", "lazy")
if _SCORING_MODE not in _VALID_SCORING:
    logger.warning(
        "[hima] unknown VLLM_HIMA_LPB_SCORING=%r; falling back to 'lazy' "
        "(valid: %s)", _SCORING_MODE, sorted(_VALID_SCORING),
    )
    _SCORING_MODE = "lazy"
_SCORING_LOGGED = False

# Per-block location sentinels for O(1) remove/refresh routing.
_LOC_COLD = 0
_LOC_HOT = 1
_LOC_EVICT = 2


class LPBFreeBlockQueue:
    """Recency-aware three-tier free-block queue; drop-in for
    ``FreeKVCacheBlockQueue``.

    BlockPool surface: ``num_free_blocks`` (attr), ``popleft`` /
    ``popleft_n`` / ``append`` / ``append_n`` / ``remove`` /
    ``get_all_free_blocks``. HiMA helpers: ``set_block_depth`` /
    ``refresh_lpb_score`` / ``maybe_eager_refresh`` / ``score_of``.
    """

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        runtime: HiMARuntime | None = None,
        pool_kind: PoolKind = PoolKind.KV,
        block_size: int = 1,
    ) -> None:
        self._eager = _SCORING_MODE in ("eager", "eager_depth_tokens")
        self._depth_tokens = _SCORING_MODE in ("depth_tokens", "eager_depth_tokens")
        self._block_size = max(1, block_size)
        global _SCORING_LOGGED
        if not _SCORING_LOGGED:
            logger.info(
                "[hima] LPB recency-aware scoring (variant=%s eager=%s "
                "depth_tokens=%s block_size=%d)",
                _SCORING_MODE, self._eager, self._depth_tokens, self._block_size,
            )
            _SCORING_LOGGED = True

        self._blocks_by_id: dict[int, KVCacheBlock] = {b.block_id: b for b in blocks}
        self._runtime = runtime
        self.pool_kind = pool_kind

        # Tier 1: demoted, stale-hit blocks (drained first). Tier 2: cold
        # FIFO of never-hit blocks. Tier 3: hot heap of in-window hits.
        self._evict_first: set[int] = set()
        self._cold = FreeKVCacheBlockQueue(blocks)
        self._hot: LPBPriorityQueue[int] = LPBPriorityQueue()
        self._loc: dict[int, int] = {bid: _LOC_COLD for bid in self._blocks_by_id}
        # Recency stamp for hot-heap keys (lower = older).
        self._stamp = 0.0
        self._recency: dict[int, float] = {}
        # block_id → prefix-tree depth (set by coordinator on cache_blocks).
        self._block_depth: dict[int, int] = {}

        # Decay throttle: re-scoring the hot set on every pop is wasteful;
        # stale hits only need to demote on the window timescale.
        window_s = (
            runtime.config.hima_lpb_window_s if runtime is not None else 60.0
        )
        self._clock = runtime.path_counter.clock if runtime is not None else None
        self._decay_interval = max(0.05, window_s / 16.0)
        self._next_decay = 0.0  # 0 ⇒ scan on first eviction

        # Hot-loop attribute caches; bound on first ``_priority`` call.
        self._pc_count = None
        self._c_curve = None
        self._c_cache: dict[int, float] = {}

        # Instrumentation: ``protected`` counts evictions of a block that
        # still carried a windowed hit (came from the hot tier). ``0`` for a
        # whole run ⇒ pressure never reached the hot tier ⇒ L1 ≡ LRU.
        self._ev_total = 0
        self._ev_protected = 0

        self.num_free_blocks = len(blocks)

    # ------------------------------ scoring ------------------------------ #

    def _next_stamp(self) -> float:
        self._stamp += 1.0
        return self._stamp

    def _priority(self, block_id: int) -> float:
        """Windowed hit value ``n_b_windowed × c_pool(depth)``; 0 if no
        in-window hit (then the block belongs in the cold/evict tiers)."""
        rt = self._runtime
        if rt is None:
            return 0.0
        pc_count = self._pc_count
        if pc_count is None:
            pc_count = rt.path_counter.count
            self._pc_count = pc_count
            curves = rt.cost_curves
            self._c_curve = (
                curves.c_kv_ms if self.pool_kind == PoolKind.KV else curves.c_m_ms
            )
        n_b = pc_count(block_id)
        if n_b == 0:
            return 0.0
        depth = self._block_depth.get(block_id, 1)
        cache = self._c_cache
        c = cache.get(depth)
        if c is None:
            curve_arg = depth * self._block_size if self._depth_tokens else depth
            c = self._c_curve(curve_arg)
            cache[depth] = c
        return n_b * c

    def _decay_hits(self) -> None:
        """Demote hot blocks whose windowed hits have expired into the
        evict-first tier. Throttled to once per ``_decay_interval`` of clock
        time; only re-pushes a still-hot block when its priority changed."""
        if self._clock is None or self._hot.is_empty():
            return
        now = self._clock()
        if now < self._next_decay:
            return
        self._next_decay = now + self._decay_interval
        hot = self._hot
        for bid in list(hot):
            prio = self._priority(bid)
            if prio == 0.0:
                hot.remove(bid)
                self._evict_first.add(bid)
                self._loc[bid] = _LOC_EVICT
            elif prio != hot.score_of(bid)[0]:
                hot.update(bid, (prio, self._recency[bid]))

    # ------------------------- queue operations -------------------------- #

    def _note_evict(self, protected: bool) -> None:
        self._ev_total += 1
        if protected:
            self._ev_protected += 1
        if self._ev_total % 2000 == 0:
            logger.info(
                "[hima/lpb] evicts tot=%d protected=%d (%.1f%%) | "
                "evict_q=%d cold=%d hot=%d",
                self._ev_total, self._ev_protected,
                100.0 * self._ev_protected / self._ev_total,
                len(self._evict_first), self._cold.num_free_blocks, len(self._hot),
            )

    def popleft(self) -> KVCacheBlock:
        if self.num_free_blocks <= 0:
            raise ValueError("No free blocks available")
        self._decay_hits()
        self.num_free_blocks -= 1
        if self._evict_first:
            bid = self._evict_first.pop()
            del self._loc[bid]
            self._recency.pop(bid, None)
            self._note_evict(False)
            return self._blocks_by_id[bid]
        if self._cold.num_free_blocks > 0:
            b = self._cold.popleft()
            del self._loc[b.block_id]
            self._note_evict(False)
            return b
        bid, _ = self._hot.popmin()
        del self._loc[bid]
        self._recency.pop(bid, None)
        self._note_evict(True)
        return self._blocks_by_id[bid]

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if n > self.num_free_blocks:
            raise AssertionError(
                f"popleft_n({n}) but only {self.num_free_blocks} blocks are free"
            )
        self._decay_hits()
        self.num_free_blocks -= n
        ret: list[KVCacheBlock] = []
        bb = self._blocks_by_id
        # Tier 1: stale-hit evict-first set.
        while self._evict_first and len(ret) < n:
            bid = self._evict_first.pop()
            del self._loc[bid]
            self._recency.pop(bid, None)
            self._note_evict(False)
            ret.append(bb[bid])
        # Tier 2: cold FIFO.
        need = n - len(ret)
        if need > 0 and self._cold.num_free_blocks > 0:
            k = min(need, self._cold.num_free_blocks)
            for b in self._cold.popleft_n(k):
                del self._loc[b.block_id]
                self._note_evict(False)
                ret.append(b)
        # Tier 3: hot heap.
        need = n - len(ret)
        if need > 0:
            for bid, _ in self._hot.popmin_n(need):
                del self._loc[bid]
                self._recency.pop(bid, None)
                self._note_evict(True)
                ret.append(bb[bid])
        return ret

    def append(self, block: KVCacheBlock) -> None:
        bid = block.block_id
        prio = self._priority(bid)
        if prio > 0.0:
            stamp = self._next_stamp()
            self._recency[bid] = stamp
            self._hot.add(bid, (prio, stamp))
            self._loc[bid] = _LOC_HOT
        else:
            self._cold.append(block)
            self._loc[bid] = _LOC_COLD
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for blk in blocks:
            self.append(blk)

    def remove(self, block: KVCacheBlock) -> None:
        bid = block.block_id
        loc = self._loc.pop(bid, None)
        if loc is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        if loc == _LOC_COLD:
            self._cold.remove(block)
        elif loc == _LOC_HOT:
            self._hot.remove(bid)
            self._recency.pop(bid, None)
        else:  # _LOC_EVICT
            self._evict_first.discard(bid)
            self._recency.pop(bid, None)
        self.num_free_blocks -= 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        bb = self._blocks_by_id
        out: list[KVCacheBlock] = [bb[bid] for bid in self._evict_first]
        cur = self._cold.fake_free_list_head.next_free_block
        tail = self._cold.fake_free_list_tail
        while cur is not None and cur is not tail:
            out.append(cur)
            cur = cur.next_free_block
        out.extend(bb[bid] for bid in self._hot)
        return out

    # ------------------ HiMA-specific helpers (optional) ----------------- #

    def set_block_depth(self, block_id: int, depth: int) -> None:
        """Record prefix-tree depth for LPB scoring; called by HiMACoordinator."""
        self._block_depth[block_id] = depth

    def refresh_lpb_score(self, block: KVCacheBlock) -> None:
        """Recompute a free block's score after a hit. Bumps recency (a hit
        is an access) and migrates it across tiers if its priority crossed 0.
        No-op if the block isn't currently free."""
        bid = block.block_id
        loc = self._loc.get(bid)
        if loc is None:
            return
        prio = self._priority(bid)
        if prio > 0.0:
            stamp = self._next_stamp()
            self._recency[bid] = stamp
            if loc == _LOC_HOT:
                self._hot.update(bid, (prio, stamp))
            else:  # promote cold/evict → hot
                if loc == _LOC_COLD:
                    self._cold.remove(block)
                else:
                    self._evict_first.discard(bid)
                self._hot.add(bid, (prio, stamp))
                self._loc[bid] = _LOC_HOT
        elif loc == _LOC_HOT:
            # Lost all in-window hits → demote to evict-first.
            self._hot.remove(bid)
            self._recency.pop(bid, None)
            self._evict_first.add(bid)
            self._loc[bid] = _LOC_EVICT

    def maybe_eager_refresh(self, block_id: int) -> None:
        """verify/4 'eager' variant: re-score right after a hit is recorded.
        No-op for 'lazy' variants and for blocks not currently free."""
        if not self._eager:
            return
        block = self._blocks_by_id.get(block_id)
        if block is not None:
            self.refresh_lpb_score(block)

    def score_of(self, block: KVCacheBlock) -> tuple[float, float]:
        bid = block.block_id
        loc = self._loc.get(bid)
        if loc == _LOC_HOT:
            return self._hot.score_of(bid)
        # cold / evict tiers carry no in-window hit.
        return (0.0, self._recency.get(bid, 0.0))


__all__ = ["LPBFreeBlockQueue"]
