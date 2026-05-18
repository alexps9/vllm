# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Min-heap with lazy-deletion for LPB eviction ordering (O(log N) add/popmin)."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)


@dataclass(order=True)
class _Entry(Generic[K]):
    score: float
    seq: int
    key: K = field(compare=False)
    alive: bool = field(default=True, compare=False)


class LPBPriorityQueue(Generic[K]):
    """Min-heap of ``(score, key)`` with lazy-deletion semantics.

    Methods all raise :class:`KeyError` for missing keys, mirroring ``dict``.
    """

    def __init__(self) -> None:
        self._heap: list[_Entry[K]] = []
        self._index: dict[K, _Entry[K]] = {}
        self._counter = itertools.count()

    # -------------------------- public api ------------------------------ #

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: K) -> bool:
        return key in self._index

    def __iter__(self) -> Iterator[K]:
        return iter(self._index)

    def is_empty(self) -> bool:
        return not self._index

    def add(self, key: K, score: float) -> None:
        """Insert ``key`` with ``score``, or refuse if already present."""
        if key in self._index:
            raise KeyError(f"key {key!r} already present; use update()")
        entry = _Entry(score=score, seq=next(self._counter), key=key)
        self._index[key] = entry
        heapq.heappush(self._heap, entry)

    def update(self, key: K, score: float) -> None:
        """Re-score ``key`` (logical move within the heap)."""
        if key not in self._index:
            raise KeyError(key)
        old = self._index[key]
        old.alive = False
        new = _Entry(score=score, seq=next(self._counter), key=key)
        self._index[key] = new
        heapq.heappush(self._heap, new)

    def remove(self, key: K) -> None:
        """Remove ``key``; idempotent for already-removed keys."""
        entry = self._index.pop(key, None)
        if entry is not None:
            entry.alive = False

    def peek(self) -> tuple[K, float]:
        """Return ``(key, score)`` of the lowest-scoring entry without popping."""
        self._purge()
        if not self._heap:
            raise KeyError("queue is empty")
        top = self._heap[0]
        return top.key, top.score

    def popmin(self) -> tuple[K, float]:
        """Pop the lowest-scoring entry."""
        self._purge()
        if not self._heap:
            raise KeyError("queue is empty")
        top = heapq.heappop(self._heap)
        del self._index[top.key]
        return top.key, top.score

    def popmin_n(self, n: int) -> list[tuple[K, float]]:
        """Pop the ``n`` lowest-scoring entries in ascending order."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        out: list[tuple[K, float]] = []
        for _ in range(n):
            self._purge()
            if not self._heap:
                break
            top = heapq.heappop(self._heap)
            del self._index[top.key]
            out.append((top.key, top.score))
        return out

    def peek_n_scores(self, n: int) -> list[float]:
        """Return the ``n`` lowest scores without removing entries (O(n log N))."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return []
        popped = self.popmin_n(n)
        scores = [s for _, s in popped]
        for key, score in popped:
            entry = _Entry(score=score, seq=next(self._counter), key=key)
            self._index[key] = entry
            heapq.heappush(self._heap, entry)
        return scores

    def score_of(self, key: K) -> float:
        return self._index[key].score

    # -------------------------- internals ------------------------------- #

    def _purge(self) -> None:
        while self._heap and not self._heap[0].alive:
            heapq.heappop(self._heap)


__all__ = ["LPBPriorityQueue"]
