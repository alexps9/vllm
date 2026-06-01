# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 unit tests for the interlayer SubBlockPool + IndexedHeap.

Re-runs, against the REAL classes, the properties the standalone prototypes
proved in dev/interlayer/0_feasibility/ (sub_block_allocator/fuzz_refcount.py
memory-safety; decision_cost IndexedHeap correctness):
  * per-sub-block ref-counting + cached lifecycle + prefix sharing (ref>1),
  * packing bias (fill a page before opening a fresh one),
  * a page returns to the source iff all its sub-blocks are free,
  * reclaim vacates the cheapest reclaimable page,
  * append-only ids, no aliasing, leak-clean after draining,
  * IndexedHeap == dict+min reference under interior remove / remove-min / churn.
"""

import random

import pytest

from vllm.v1.core.interlayer.sub_block_pool import IndexedHeap, SubBlockPool


class FakePages:
    """Stand-in for BlockPool's whole-page allocation (Stage 2 wires the real one)."""

    def __init__(self, n: int) -> None:
        self._free = list(range(n))
        self.in_use: set[int] = set()

    def alloc_page(self) -> int | None:
        if not self._free:
            return None
        p = self._free.pop()
        self.in_use.add(p)
        return p

    def free_page(self, pid: int) -> None:
        assert pid in self.in_use, f"double-free of page {pid}"
        self.in_use.discard(pid)
        self._free.append(pid)


# --------------------------------------------------------------------------- #
# IndexedHeap correctness (mirrors decision_cost property test)
# --------------------------------------------------------------------------- #
def _check_heap(h: IndexedHeap, ref: dict[int, float]) -> None:
    arr, pos = h._h, h._pos
    assert len(arr) == len(pos) == len(ref)
    assert {k for _, k in arr} == set(pos) == set(ref)
    for i, (sc, k) in enumerate(arr):
        assert pos[k] == i
        assert abs(ref[k] - sc) < 1e-12
        if i > 0:
            assert arr[(i - 1) >> 1][0] <= sc


@pytest.mark.parametrize("seed", range(8))
def test_indexed_heap_matches_reference(seed: int) -> None:
    rng = random.Random(seed)
    h = IndexedHeap()
    ref: dict[int, float] = {}
    keyspace = (8, 25, 200)[seed % 3]
    for _ in range(6000):
        r = rng.random(); k = rng.randrange(keyspace)
        if r < 0.5:
            sc = round(rng.random() * 8, 3)
            h.add_or_update(k, sc); ref[k] = sc
        elif r < 0.8:
            h.remove(k); ref.pop(k, None)
        elif r < 0.87 and ref:
            mk = min(ref, key=ref.get)
            h.remove(mk); ref.pop(mk)
        else:
            ans = h.peek()
            truth = (min(ref, key=ref.get), min(ref.values())) if ref else None
            if truth is None:
                assert ans is None
            else:
                assert ans is not None and abs(ans[1] - truth[1]) < 1e-9
        _check_heap(h, ref)
    for k in list(ref):
        h.remove(k); ref.pop(k); _check_heap(h, ref)


# --------------------------------------------------------------------------- #
# SubBlockPool — targeted behaviors
# --------------------------------------------------------------------------- #
def test_packing_bias_fills_page_before_opening_new() -> None:
    src = FakePages(10)
    pool = SubBlockPool(sub_per_page=4, page_source=src)
    ids = pool.alloc_blocks(4)            # exactly one page's worth
    assert len(ids) == 4
    assert pool.num_open_pages() == 1     # packed into ONE page, not 4
    assert all(i // 4 == ids[0] // 4 for i in ids)
    pool.check()
    pool.alloc_blocks(1)                  # 5th -> must open a 2nd page
    assert pool.num_open_pages() == 2
    pool.check()


def test_page_returns_to_source_when_fully_free() -> None:
    src = FakePages(10)
    pool = SubBlockPool(sub_per_page=4, page_source=src)
    ids = pool.alloc_blocks(3)
    assert len(src.in_use) == 1
    pool.free_blocks(ids)                 # all freed -> cached, page still carved
    assert len(src.in_use) == 1           # kept (cached for reuse), reclaimable
    assert pool.reclaim_page_for_mamba() is not None
    assert len(src.in_use) == 0           # vacated -> returned to source
    pool.check()


def test_reclaim_picks_cheapest_page() -> None:
    src = FakePages(10)
    R = 4
    pool = SubBlockPool(sub_per_page=R, page_source=src)
    # page A: 1 cached; page B: 3 cached  -> reclaim must pick A (cheaper)
    a = pool.alloc_blocks(R)              # page A full
    b = pool.alloc_blocks(R)              # page B full
    pa, pb = a[0] // R, b[0] // R
    pool.free_blocks(a[:1])              # A: 1 cached, 3 still... wait free 1 -> live 3
    pool.free_blocks(a[1:])             # free rest of A -> A live0, 4 cached
    pool.free_blocks(b[:3])             # B: free 3 -> live 1 (not reclaimable yet)
    # A is reclaimable (live0, 4 cached); B is not (live1). reclaim -> A.
    pool.check()
    assert pool.reclaim_page_for_mamba() == pa
    pool.check()


def test_prefix_sharing_refcount() -> None:
    src = FakePages(10)
    R = 4
    pool = SubBlockPool(sub_per_page=R, page_source=src)
    ids = pool.alloc_blocks(2)           # req1 owns 2 sub-blocks
    pool.touch(ids)                      # req2 shares them -> ref 2
    assert all(pool.ref[i] == 2 for i in ids)
    pool.free_blocks(ids)               # req1 frees -> ref 1 (still live, NOT cached)
    assert all(pool.ref[i] == 1 for i in ids)
    assert not (set(ids) & pool.cached)
    pool.free_blocks(ids)               # req2 frees -> ref 0 -> cached
    assert all(i in pool.cached for i in ids)
    pool.check()


# --------------------------------------------------------------------------- #
# SubBlockPool — randomized fuzz with every-op invariant check + leak check
# (mirrors sub_block_allocator/fuzz_refcount.py against the real class)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R,n_pages,seed", [(4, 20, 0), (4, 20, 1), (33, 40, 2)])
def test_fuzz_memory_safe(R: int, n_pages: int, seed: int) -> None:
    rng = random.Random(seed)
    src = FakePages(n_pages)
    pool = SubBlockPool(sub_per_page=R, page_source=src)
    reqs: dict[int, list[int]] = {}      # req_id -> sub_ids it references (once each)
    hot: list[int] = []                  # shareable sub-block ids
    nxt = 0
    max_ref = 0
    shared = 0
    n_ops = 8000
    check_every = 1 if R == 4 else 25

    for i in range(n_ops):
        op = rng.random()
        if op < 0.34:                                  # alloc new
            nxt += 1
            got = pool.alloc_blocks(rng.randint(1, 2 * R))
            if got:
                reqs[nxt] = list(got)
                if rng.random() < 0.2:
                    hot.extend(got[:3]); hot[:] = hot[-100:]
        elif op < 0.55 and hot:                        # shared-prefix touch
            nxt += 1
            k = rng.randint(1, min(6, len(hot)))
            picks = rng.sample(hot, k)
            # only touch ids that are still carved (not vacated)
            picks = [s for s in picks
                     if pool.page_of(s) in pool._free_slots]
            if picks:
                pool.touch(picks)
                reqs[nxt] = list(picks)
                shared += 1
        elif op < 0.78 and reqs:                       # free a request
            r = rng.choice(list(reqs))
            pool.free_blocks(reqs.pop(r))
        else:                                          # mamba reclaim
            pool.reclaim_page_for_mamba()
            # any sub_ids on a vacated page are gone; drop dead reqs lazily
        if pool.ref:
            max_ref = max(max_ref, max(pool.ref.values()))
        if i % check_every == 0:
            pool.check()
        # prune reqs whose pages were reclaimed out from under them
        if op >= 0.78:
            for r in list(reqs):
                reqs[r] = [s for s in reqs[r]
                           if pool.page_of(s) in pool._free_slots
                           and pool.ref.get(s, 0) > 0]
                if not reqs[r]:
                    del reqs[r]

    pool.check()
    # drain: free everything still referenced, then vacate all reclaimable pages
    for r in list(reqs):
        live = [s for s in reqs[r] if pool.ref.get(s, 0) > 0]
        # free each reference once
        seen: dict[int, int] = {}
        for s in live:
            seen[s] = seen.get(s, 0) + 1
        for s, c in seen.items():
            pool.free_blocks([s] * min(c, pool.ref.get(s, 0)))
    while pool.reclaim_page_for_mamba() is not None:
        pass
    pool.check()
    assert max_ref >= 2, "prefix sharing (ref>1) was never exercised"
    assert len(src.in_use) == 0, "leak: pages not all returned after drain"
    assert not pool.ref and not pool.cached, "leak: dangling refs/cached"
    assert pool.num_open_pages() == 0
