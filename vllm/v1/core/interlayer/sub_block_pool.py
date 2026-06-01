# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SubBlockPool — packs attention sub-blocks into mamba-sized physical pages.

Sits *between* the attention manager and the shared physical page pool
(``BlockPool``). It draws a whole page from the page source, carves it into
``R = block_size // kernel_block_size`` sub-blocks, and hands those out with a
**packing bias** (fill a partially-used page before opening a fresh one). A page
is returned to the source (mamba-usable) only when **all** its sub-blocks are
free of references; a page that still holds cached (prefix-reusable) sub-blocks
is *reclaimable* and ranked by an :class:`IndexedHeap` so the cheapest page can
be vacated on demand.

Append-only ids: a sub-block id is ``page_id * R + slot`` — stable, never
relocated (reclaim is by eviction, never movement; mirrors ``block_pool.py``).

This module is **inert** until wired into the attention allocation path
(1_allocator Stage 2). The design + feasibility proofs are in ``dev/interlayer/``
(``sub_block_allocator`` = memory-safety, ``decision_cost`` = the IndexedHeap
choice). Cost values used for the reclaim ranking here are **placeholders**
(``#cached``); the real recompute-cost model is 2_cost_reclaim (L2).
"""

from __future__ import annotations

import heapq  # noqa: F401  (kept for parity; IndexedHeap is hand-rolled below)
from typing import Protocol


class PageSource(Protocol):
    """The shared physical page pool (e.g. ``BlockPool``), page = mamba size."""

    def alloc_page(self) -> int | None:
        """Return a free physical page id, or None if none available."""
        ...

    def free_page(self, page_id: int) -> None:
        """Return a fully-free page to the pool (mamba-usable again)."""
        ...


class IndexedHeap:
    """Min-heap with EAGER delete/update: O(1) peek, O(log n) add/update/remove,
    no stale entries (no lazy-delete bloat, no O(P) peek-trim spikes). The
    structure chosen by ``dev/interlayer/0_feasibility/decision_cost`` (audited
    ×4) for the "cheapest page to vacate" decision."""

    __slots__ = ("_h", "_pos")

    def __init__(self) -> None:
        self._h: list[tuple[float, int]] = []   # (score, key)
        self._pos: dict[int, int] = {}          # key -> index in _h

    def __len__(self) -> int:
        return len(self._h)

    def __contains__(self, key: int) -> bool:
        return key in self._pos

    def _swap(self, i: int, j: int) -> None:
        h = self._h
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][1]] = i
        self._pos[h[j][1]] = j

    def _sift_up(self, i: int) -> None:
        h = self._h
        while i > 0:
            p = (i - 1) >> 1
            if h[i][0] < h[p][0]:
                self._swap(i, p); i = p
            else:
                break

    def _sift_down(self, i: int) -> None:
        h = self._h; n = len(h)
        while True:
            l, r, sm = 2 * i + 1, 2 * i + 2, i
            if l < n and h[l][0] < h[sm][0]:
                sm = l
            if r < n and h[r][0] < h[sm][0]:
                sm = r
            if sm == i:
                break
            self._swap(i, sm); i = sm

    def add_or_update(self, key: int, score: float) -> None:
        pos = self._pos
        if key in pos:
            i = pos[key]; old = self._h[i][0]
            self._h[i] = (score, key)
            if score < old:
                self._sift_up(i)
            elif score > old:
                self._sift_down(i)
        else:
            self._h.append((score, key))
            i = len(self._h) - 1
            pos[key] = i
            self._sift_up(i)

    def remove(self, key: int) -> None:
        pos = self._pos
        i = pos.pop(key, None)
        if i is None:
            return
        h = self._h
        last = h.pop()
        if i < len(h):
            h[i] = last
            pos[last[1]] = i
            self._sift_up(i)
            self._sift_down(i)

    def peek(self) -> tuple[int, float] | None:
        if not self._h:
            return None
        score, key = self._h[0]
        return key, score


class SubBlockPool:
    """Two-level allocator: attention sub-blocks within shared physical pages."""

    def __init__(self, sub_per_page: int, page_source: PageSource) -> None:
        assert sub_per_page >= 1
        self.R = sub_per_page
        self.src = page_source
        # carved pages currently held for attention:
        self._free_slots: dict[int, list[int]] = {}   # pid -> ref0 slot indices (stack)
        self._live: dict[int, int] = {}               # pid -> # sub-blocks with ref>0
        self._ncached: dict[int, int] = {}            # pid -> # cached (ref0, has content)
        self.ref: dict[int, int] = {}                 # sub_block_id -> ref_cnt (>0 only)
        self.cached: set[int] = set()                 # sub_block_ids holding cache content
        # selection sets (packing bias): partial = live>0 & has free; idle = live==0 (reclaimable)
        self._partial: set[int] = set()
        self._idle: set[int] = set()
        self.heap = IndexedHeap()                     # reclaimable pages by vacate-cost
        self.evicts = 0
        self.starves = 0

    # ---- id helpers (append-only) ----
    def sid(self, pid: int, slot: int) -> int:
        return pid * self.R + slot

    def page_of(self, sub_id: int) -> int:
        return sub_id // self.R

    # ---- internal: (re)classify a page into the right selection set ----
    def _classify(self, pid: int) -> None:
        if pid not in self._free_slots:
            return
        nf = len(self._free_slots[pid])
        lv = self._live[pid]
        nc = self._ncached[pid]
        self._partial.discard(pid)
        self._idle.discard(pid)
        self.heap.remove(pid)
        if lv > 0:
            if nf > 0:
                self._partial.add(pid)          # else FULL: in no set
        else:                                   # live == 0: all R slots are ref0
            if nc > 0:
                self._idle.add(pid)             # reclaimable + packable (evict on reuse)
                self.heap.add_or_update(pid, float(nc))  # vacate-cost placeholder
            else:                               # pure-empty page -> return to source
                del self._free_slots[pid]; del self._live[pid]; del self._ncached[pid]
                self.src.free_page(pid)

    def _open_page(self) -> int | None:
        pid = self.src.alloc_page()
        if pid is None:
            return None
        self._free_slots[pid] = list(range(self.R - 1, -1, -1))
        self._live[pid] = 0
        self._ncached[pid] = 0
        return pid

    def _take_slot(self, pid: int) -> int:
        slot = self._free_slots[pid].pop()
        sid = self.sid(pid, slot)
        if sid in self.cached:                  # reuse a cached-but-free slot -> evict
            self.cached.discard(sid)
            self._ncached[pid] -= 1
            self.evicts += 1
        self.ref[sid] = 1
        self._live[pid] += 1
        return sid

    # ---- attention: allocate n NEW sub-blocks (ref 0 -> 1), packing bias ----
    def alloc_blocks(self, n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            if self._partial:
                pid = next(iter(self._partial))
            elif self._idle:
                pid = next(iter(self._idle))
            else:
                pid = self._open_page()
                if pid is None:
                    self.starves += 1
                    break                        # caller must preempt/defer (phase 2)
            out.append(self._take_slot(pid))
            self._classify(pid)
        return out

    # ---- prefix sharing: req references existing sub-blocks (ref++) ----
    def touch(self, sub_ids: list[int]) -> None:
        for sid in sub_ids:
            pid = self.page_of(sid)
            cur = self.ref.get(sid, 0)
            if cur == 0:                         # cached/free -> live; consume its slot
                if sid in self.cached:
                    self.cached.discard(sid)
                    self._ncached[pid] -= 1
                self._free_slots[pid].remove(sid % self.R)
                self._live[pid] += 1
            self.ref[sid] = cur + 1
            self._classify(pid)

    # ---- free: ref-- ; at 0 the sub-block becomes cached (reusable) ----
    def free_blocks(self, sub_ids: list[int]) -> None:
        for sid in sub_ids:
            pid = self.page_of(sid)
            cur = self.ref.get(sid, 0)
            if cur <= 0:
                raise AssertionError(f"free of unreferenced sub-block {sid}")
            cur -= 1
            if cur == 0:
                del self.ref[sid]
                self.cached.add(sid)             # freed-but-cached (evictable)
                self._ncached[pid] += 1
                self._free_slots[pid].append(sid % self.R)
                self._live[pid] -= 1
            else:
                self.ref[sid] = cur
            self._classify(pid)

    # ---- mamba demand: vacate the cheapest reclaimable page, return to source ----
    def reclaim_page_for_mamba(self) -> int | None:
        top = self.heap.peek()
        if top is None:
            return None                          # nothing reclaimable -> preempt (phase 2)
        pid, _cost = top
        base = pid * self.R
        for slot in range(self.R):
            sid = base + slot
            if sid in self.cached:
                self.cached.discard(sid)
                self.evicts += 1
        self.heap.remove(pid)
        self._idle.discard(pid)
        del self._free_slots[pid]; del self._live[pid]; del self._ncached[pid]
        self.src.free_page(pid)
        return pid

    # ---- introspection / invariants (used by tests) ----
    def num_open_pages(self) -> int:
        return len(self._free_slots)

    def check(self) -> None:
        """Raise on any invariant breach. O(open pages * R)."""
        for pid, slots in self._free_slots.items():
            lv = self._live[pid]
            if not (0 <= lv <= self.R):
                raise AssertionError(f"page {pid}: live {lv} out of [0,{self.R}]")
            if len(slots) != self.R - lv:
                raise AssertionError(
                    f"page {pid}: free_slots {len(slots)} != R-live {self.R - lv}")
            if len(set(slots)) != len(slots):
                raise AssertionError(f"page {pid}: duplicate free slots")
            base = pid * self.R
            nc = 0
            for slot in range(self.R):
                sid = base + slot
                r = self.ref.get(sid, 0)
                free = slot in slots
                if r > 0 and free:
                    raise AssertionError(f"sub {sid}: ref>0 but in free_slots")
                if r == 0 and not free:
                    raise AssertionError(f"sub {sid}: ref0 but not free")
                if sid in self.cached:
                    if r != 0:
                        raise AssertionError(f"cached sub {sid} has ref {r}")
                    nc += 1
            if nc != self._ncached[pid]:
                raise AssertionError(f"page {pid}: ncached {self._ncached[pid]} != {nc}")
            # heap membership: reclaimable iff live==0 and cached>0
            in_heap = pid in self.heap
            should = lv == 0 and nc > 0
            if in_heap != should:
                raise AssertionError(
                    f"page {pid}: heap={in_heap} but reclaimable={should}")
        # no sub-block references a returned page
        for sid in self.ref:
            if self.page_of(sid) not in self._free_slots:
                raise AssertionError(f"ref on sub {sid} of returned page")
        for sid in self.cached:
            if self.page_of(sid) not in self._free_slots:
                raise AssertionError(f"cached sub {sid} of returned page")
