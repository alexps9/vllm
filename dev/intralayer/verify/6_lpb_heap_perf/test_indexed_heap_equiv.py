"""Smoke / property test: indexed-heap LPBPriorityQueue behaves identically
to a reference dataclass+lazy-delete heap on a randomized workload.

Run from repo root:
    .venv/bin/python -u dev/intralayer/verify/6_lpb_heap_perf/test_indexed_heap_equiv.py
"""
from __future__ import annotations

import heapq
import itertools
import random
import sys

sys.path.insert(0, "/data/yuzhou/projects/vllm-songyang")

from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue


# Reference: the OLD lazy-delete implementation, locked in here so we can
# diff against it forever.
class _RefEntry:
    __slots__ = ("score", "seq", "key", "alive")

    def __init__(self, score, seq, key):
        self.score = score
        self.seq = seq
        self.key = key
        self.alive = True

    def __lt__(self, other):
        return (self.score, self.seq) < (other.score, other.seq)


class ReferenceLPBQueue:
    def __init__(self):
        self._heap = []
        self._index = {}
        self._counter = itertools.count()

    def __len__(self):
        return len(self._index)

    def __contains__(self, key):
        return key in self._index

    def add(self, key, score):
        if key in self._index:
            raise KeyError(key)
        e = _RefEntry(score, next(self._counter), key)
        self._index[key] = e
        heapq.heappush(self._heap, e)

    def update(self, key, score):
        if key not in self._index:
            raise KeyError(key)
        self._index[key].alive = False
        e = _RefEntry(score, next(self._counter), key)
        self._index[key] = e
        heapq.heappush(self._heap, e)

    def remove(self, key):
        e = self._index.pop(key, None)
        if e is not None:
            e.alive = False

    def _purge(self):
        while self._heap and not self._heap[0].alive:
            heapq.heappop(self._heap)

    def peek(self):
        self._purge()
        if not self._heap:
            raise KeyError
        e = self._heap[0]
        return e.key, e.score

    def popmin(self):
        self._purge()
        if not self._heap:
            raise KeyError
        e = heapq.heappop(self._heap)
        del self._index[e.key]
        return e.key, e.score

    def score_of(self, key):
        return self._index[key].score


def run_property(seed=0xC0FFEE, n_keys=200, n_ops=20000):
    rng = random.Random(seed)
    ref = ReferenceLPBQueue()
    new = LPBPriorityQueue()
    ops = ("add", "update", "remove", "popmin", "score_of", "peek")
    n_mismatch = 0
    for step in range(n_ops):
        op = rng.choices(ops, weights=(40, 25, 5, 25, 3, 2))[0]
        if op == "add":
            key = rng.randrange(n_keys)
            if key in ref:
                continue
            score = rng.uniform(0, 1e6)
            ref.add(key, score)
            new.add(key, score)
        elif op == "update":
            if not ref._index:
                continue
            key = rng.choice(list(ref._index.keys()))
            score = rng.uniform(0, 1e6)
            ref.update(key, score)
            new.update(key, score)
        elif op == "remove":
            if not ref._index:
                continue
            key = rng.choice(list(ref._index.keys()))
            ref.remove(key)
            new.remove(key)
            # Idempotent remove (re-remove same key on a sample of steps)
            if step % 50 == 0:
                ref.remove(key)
                new.remove(key)
        elif op == "popmin":
            if not ref._heap:
                continue
            ref._purge()
            if not ref._heap:
                continue
            ref_kv = ref.popmin()
            new_kv = new.popmin()
            if ref_kv != new_kv:
                # The two implementations are allowed to break score ties in
                # different orders — accept any popmin whose score matches and
                # whose key is currently min-score in both maps.
                if ref_kv[1] != new_kv[1]:
                    print(f"step={step} popmin score mismatch ref={ref_kv} new={new_kv}")
                    n_mismatch += 1
                else:
                    # Score tied but key differs. Restore the new-side key into ref's index
                    # to keep the structures synced.
                    new.add(ref_kv[0], ref_kv[1])
                    new.remove(new_kv[0])
                    ref.remove(ref_kv[0])
                    ref.add(new_kv[0], new_kv[1])
        elif op == "score_of":
            if not ref._index:
                continue
            key = rng.choice(list(ref._index.keys()))
            if ref.score_of(key) != new.score_of(key):
                print(f"step={step} score_of mismatch")
                n_mismatch += 1
        elif op == "peek":
            if not ref._index:
                continue
            ref._purge()
            if not ref._heap:
                continue
            ref_kv = ref.peek()
            new_kv = new.peek()
            if ref_kv[1] != new_kv[1]:
                print(f"step={step} peek score mismatch ref={ref_kv} new={new_kv}")
                n_mismatch += 1
        # Length invariant.
        if len(ref) != len(new):
            print(f"step={step} len mismatch ref={len(ref)} new={len(new)}")
            n_mismatch += 1
            break
    return n_mismatch


def main() -> int:
    total = 0
    for seed in (0xC0FFEE, 0xBEEFCAFE, 0xDEAD0F00):
        m = run_property(seed=seed)
        print(f"seed={hex(seed)}: mismatches={m}")
        total += m
    if total == 0:
        print("OK — all property runs match (score-equivalent popmin order)")
        return 0
    print(f"FAIL — {total} mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
