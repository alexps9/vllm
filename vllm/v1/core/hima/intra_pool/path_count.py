# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sliding-window, path-counted hit statistics for HiMA LPB scoring.

When a request matches a prefix of depth k, all k ancestor block IDs get +1
so shared internal nodes are protected from eviction (HiMA paper §3.2).
"""

from __future__ import annotations

import collections
import math
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field

BlockId = Hashable

# Hoisted to module scope: ``_INF`` allocates a new float each call
# and the ``count()`` hot path runs once per ``_score_for``.
_INF = math.inf


@dataclass
class _Hit:
    """One observed hit on a single block, kept for sliding-window expiry."""

    block_id: BlockId
    timestamp: float


@dataclass
class PathCountedHitCounter:
    """Sliding-window counter with path-counted updates."""

    window_seconds: float
    clock: Callable[[], float] = field(default=time.monotonic)
    _events: collections.deque = field(default_factory=collections.deque)
    _counts: dict = field(default_factory=dict)
    # Stage 11d opt #1: deadline (clock() timestamp) at which the oldest
    # still-live event will become expired. ``count()`` compares
    # ``clock()`` against this once, skipping the deque walk when nothing
    # is yet ripe. +inf when the deque is empty. Maintained by
    # ``record_hit`` and ``_evict_expired``.
    _expiry_deadline: float = _INF

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds}")

    # -------------------------- public api ------------------------------ #

    def record_hit(self, path: Iterable[BlockId]) -> None:
        """Increment all block IDs in the path (root → deepest match)."""

        now = self.clock()
        self._evict_expired(now)
        events = self._events
        counts = self._counts
        was_empty = not events
        for block_id in path:
            counts[block_id] = counts.get(block_id, 0) + 1
            events.append(_Hit(block_id=block_id, timestamp=now))
        if was_empty and events:
            self._expiry_deadline = now + self.window_seconds

    def count(self, block_id: BlockId) -> int:
        """Sliding-window hit count for ``block_id``."""

        # Hot path: avoid ``clock()`` entirely when the deque is empty
        # (deadline=+inf). When non-empty, a single ``clock() < deadline``
        # test elides the deque walk + dict pops while nothing has ripened.
        deadline = self._expiry_deadline
        if deadline != _INF:
            now = self.clock()
            if now >= deadline:
                self._evict_expired(now)
        return self._counts.get(block_id, 0)

    def discard(self, block_id: BlockId) -> None:
        """Forget a block; stale events expire lazily on the next ``_evict_expired``."""

        self._counts.pop(block_id, None)

    def __len__(self) -> int:
        self._evict_expired(self.clock())
        return len(self._counts)

    # -------------------------- internals ------------------------------- #

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        events = self._events
        counts = self._counts
        while events and events[0].timestamp < cutoff:
            ev = events.popleft()
            remaining = counts.get(ev.block_id, 0) - 1
            if remaining > 0:
                counts[ev.block_id] = remaining
            else:
                counts.pop(ev.block_id, None)
        # Refresh the deadline gate for the next ``count()`` call.
        self._expiry_deadline = (
            events[0].timestamp + self.window_seconds if events else _INF
        )


__all__ = ["BlockId", "PathCountedHitCounter"]
