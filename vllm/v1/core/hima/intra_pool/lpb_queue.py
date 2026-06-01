# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Min-heap with lazy-deletion for LPB eviction ordering (O(log N) add/popmin).

Entries are raw ``(score, seq, key)`` tuples so heap ordering uses the
built-in C-level tuple comparison (no dunder dispatch into Python).
Updates and removes mark the previous entry stale via the
``_current_seq`` map; pop loops skip stale entries until they find one
whose seq matches the current map state.

Public API matches the original dataclass-based implementation exactly so
``LPBFreeBlockQueue`` and its tests need no changes.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)


class LPBPriorityQueue(Generic[K]):
    """Lazy-delete min-heap with tuple comparison.

    Methods raise :class:`KeyError` for missing keys (mirrors ``dict``).
    """

    __slots__ = ("_heap", "_current_seq", "_score", "_counter")

    # Compaction: ``add``/``update``/``remove`` leave stale leaves in ``_heap``
    # (lazy deletion); they are otherwise only trimmed from the top by
    # ``peek``/``popmin``. Under update-heavy / pop-light load (e.g. multi-turn
    # agents repeatedly freeing+re-acquiring hot prefix blocks while evictions
    # are served from the cold tier), ``_heap`` would grow linearly with the
    # number of free/re-acquire cycles — a memory leak, and an O(stale) latency
    # spike on the first ``popmin`` that finally drains the tier (measured:
    # ~1s at 4000 turns; see dev/intralayer/verify/11_lpb_heap_bloat/). Rebuild
    # when physical exceeds this factor × logical. Compaction only drops
    # entries that are already ignored, so heap order / eviction behaviour is
    # unchanged; it bounds ``_heap`` to ``_COMPACT_FACTOR × len(self)``.
    _COMPACT_FACTOR = 8

    def __init__(self) -> None:
        # Tuples (score, seq, key); tuple comparison is C-level.
        self._heap: list[tuple[float, int, K]] = []
        # key → seq of the currently-valid entry (older seqs are stale).
        self._current_seq: dict[K, int] = {}
        # key → current score (avoids walking the heap for score_of).
        self._score: dict[K, float] = {}
        self._counter = itertools.count()

    # -------------------------- public api ------------------------------ #

    def __len__(self) -> int:
        return len(self._current_seq)

    def __contains__(self, key: K) -> bool:
        return key in self._current_seq

    def __iter__(self) -> Iterator[K]:
        return iter(self._current_seq)

    def is_empty(self) -> bool:
        return not self._current_seq

    def add(self, key: K, score: float) -> None:
        """Insert ``key`` with ``score``, raising if already present."""
        if key in self._current_seq:
            raise KeyError(f"key {key!r} already present; use update()")
        seq = next(self._counter)
        self._current_seq[key] = seq
        self._score[key] = score
        heapq.heappush(self._heap, (score, seq, key))
        self._maybe_compact()

    def update(self, key: K, score: float) -> None:
        """Re-score ``key``; the old entry becomes a stale heap leaf."""
        if key not in self._current_seq:
            raise KeyError(key)
        seq = next(self._counter)
        self._current_seq[key] = seq
        self._score[key] = score
        heapq.heappush(self._heap, (score, seq, key))
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """Drop stale leaves (rebuild ``_heap`` from valid entries) when it has
        grown past ``_COMPACT_FACTOR × len(self)``. Behaviour-preserving: stale
        entries are already skipped by ``popmin``/``peek``; this only reclaims
        their memory and caps the worst-case trim cost. Amortized O(1)/push."""
        heap = self._heap
        if len(heap) <= self._COMPACT_FACTOR * max(8, len(self._current_seq)):
            return
        current = self._current_seq
        self._heap = [e for e in heap if current.get(e[2]) == e[1]]
        heapq.heapify(self._heap)

    def remove(self, key: K) -> None:
        """Remove ``key``; idempotent for already-removed keys."""
        if self._current_seq.pop(key, None) is not None:
            self._score.pop(key, None)

    def peek(self) -> tuple[K, float]:
        """Return ``(key, score)`` of the lowest-scoring entry."""
        self._discard_stale()
        if not self._heap:
            raise KeyError("queue is empty")
        score, _seq, key = self._heap[0]
        return key, score

    def popmin(self) -> tuple[K, float]:
        """Pop and return the lowest-scoring entry."""
        heap = self._heap
        current = self._current_seq
        while heap:
            score, seq, key = heapq.heappop(heap)
            if current.get(key) == seq:
                del current[key]
                del self._score[key]
                return key, score
            # else: stale (was removed or superseded by an update)
        raise KeyError("queue is empty")

    def popmin_n(self, n: int) -> list[tuple[K, float]]:
        """Pop the ``n`` lowest-scoring entries in ascending order."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0 or not self._current_seq:
            return []
        out: list[tuple[K, float]] = []
        heap = self._heap
        current = self._current_seq
        score_map = self._score
        for _ in range(n):
            while heap:
                score, seq, key = heapq.heappop(heap)
                if current.get(key) == seq:
                    del current[key]
                    del score_map[key]
                    out.append((key, score))
                    break
            else:
                break
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
            self.add(key, score)
        return scores

    def score_of(self, key: K) -> float:
        score = self._score.get(key)
        if score is None and key not in self._current_seq:
            raise KeyError(key)
        return score

    # -------------------------- internals ------------------------------- #

    def _discard_stale(self) -> None:
        """Drop heap-top entries whose seq no longer matches current_seq."""
        heap = self._heap
        current = self._current_seq
        while heap:
            top = heap[0]
            if current.get(top[2]) == top[1]:
                return
            heapq.heappop(heap)

    # Kept for backwards compatibility with any external caller.
    def _purge(self) -> None:  # pragma: no cover
        self._discard_stale()


__all__ = ["LPBPriorityQueue"]
