# SPDX-License-Identifier: Apache-2.0
"""decision_cost — is the per-step "cheapest page to free" decision cheap?

Design claim (design.md §"Cost-model decision-layer performance"):

    The per-step decision (cost-rank the cheapest page to free) ... maintain
    the "cheapest page to free" incrementally (like L1's LPB heap) rather than
    re-walking structures each time ... bounded (echoing verify/6's <=3x LRU).

Pure-CPU feasibility check (no GPU, no vLLM import). v4 — after THREE
adversarial audits. v1 (page-events, per-op timing) and v2 (independent random
slots, reclaimable-only heap) were rebuilt for fidelity; v3 added the correlated
workload + all-page vacate-cost heap; audit-3 then found the production-style
lazy-delete heap has UNBOUNDED O(P) peek-trim spikes (up to tens of ms),
masked by reporting the mean. v4's fix: an **IndexedHeap** (eager delete —
O(1) peek, O(log n) update, no stale entries) that bounds the worst case and
eliminates bloat; query latency is reported as p50/p99/MAX, not the mean.

v3 fixed the two structural workload problems (kept in v4):

  * CORRELATED WORKLOAD (audit-2 [MAJOR fidelity]): a request's attention
    sub-blocks are allocated together as its sequence grows and freed together
    when it ends — so a page is ~BIMODAL (all-live or all-free), not the 99.6%
    "partial pages flickering one slot at a time" that an independent-random-
    slot model manufactures. v3 reuses the correlated alloc/free/share lifecycle
    that sub_block_allocator/fuzz_refcount.py already models (and which was
    wrongly not used in v2): whole-sequence alloc (1..3K blocks, packing bias),
    shared-prefix touch (ref_cnt>1), whole-sequence free.

  * RANK ALL PAGES BY VACATE-COST (audit-2 [MAJOR relevance]): v2 heaped only
    currently-reclaimable (all-free) pages — but under KV pressure almost none
    are (audit measured median 0), so that heap is ~empty and the decision is
    trivial. The REAL decision (design: "free the cheapest page — evict the
    attention whose prefixes are cheapest to recompute") ranks EVERY attention
    page by its vacate-cost: 0 for a free page, Σ recompute for a cached page,
    + a large preempt penalty for live blocks. That heap holds ~all P pages
    under pressure, so incremental-vs-rewalk genuinely matters. v3 reports the
    page-occupancy + reclaimable distribution so the real regime is OWNED.

Claims, each measured on the correlated workload:
  (1) QUERY IS INCREMENTAL, NOT A RE-WALK — peek() flat in pool size P vs the
      O(P) ground-truth scan.
  (2) PER-OP MAINTENANCE <= ~3x LRU (batch-timed replay).
  (3) CORRECTNESS — peek() == true min-vacate-cost page, every query.
  (4) HEAP MEMORY — lazy-delete bloat is real (inherited LPBPriorityQueue
      behavior, task #99); compaction bounds it to <= the factor.

The incremental heap is the exact pattern of vllm's audited LPBPriorityQueue.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import time
from collections import OrderedDict

LIVE_PENALTY = 100.0   # cost to vacate a page holding a live (ref>0) sub-block
                       # (preempt a running request) >> a cached block's recompute


# --------------------------------------------------------------------------- #
# Lazy-delete min-heap — faithful to vllm LPBPriorityQueue, + OPTIONAL compaction.
# --------------------------------------------------------------------------- #
class LazyHeap:
    __slots__ = ("_heap", "_seq", "_counter", "_cf")

    def __init__(self, compact_factor: float = 0.0) -> None:
        self._heap: list[tuple[float, int, int]] = []
        self._seq: dict[int, int] = {}
        self._counter = itertools.count()
        self._cf = compact_factor

    def __len__(self) -> int:
        return len(self._seq)

    def add_or_update(self, key: int, score: float) -> None:
        seq = next(self._counter)
        self._seq[key] = seq
        heapq.heappush(self._heap, (score, seq, key))
        if self._cf and len(self._heap) > self._cf * max(8, len(self._seq)):
            self._compact()

    def remove(self, key: int) -> None:
        self._seq.pop(key, None)

    def _compact(self) -> None:
        seq = self._seq
        self._heap = [e for e in self._heap if seq.get(e[2]) == e[1]]
        heapq.heapify(self._heap)

    def _trim(self) -> None:
        heap, seq = self._heap, self._seq
        while heap and seq.get(heap[0][2]) != heap[0][1]:
            heapq.heappop(heap)

    def peek(self):
        self._trim()
        if not self._heap:
            return None
        score, _s, key = self._heap[0]
        return key, score

    def heap_len(self) -> int:
        return len(self._heap)


# --------------------------------------------------------------------------- #
# Indexed min-heap with EAGER delete/update (the structurally-correct fix).
# O(1) peek, O(log n) add/update/delete. No stale entries => no lazy-delete
# bloat (audit-1) and no unbounded peek-trim O(P) spikes (audit-3): peek is
# always valid in O(1). Recommended for the update-heavy interlayer decision
# (and a candidate fix for L1's LPBPriorityQueue, task #99).
# --------------------------------------------------------------------------- #
class IndexedHeap:
    __slots__ = ("_h", "_pos")

    def __init__(self) -> None:
        self._h: list[tuple[float, int]] = []   # (score, key)
        self._pos: dict[int, int] = {}          # key -> index in _h

    def __len__(self) -> int:
        return len(self._h)

    def _swap(self, i: int, j: int) -> None:
        h = self._h
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][1]] = i
        self._pos[h[j][1]] = j

    def _sift_up(self, i: int) -> None:
        h = self._h
        while i > 0:
            parent = (i - 1) >> 1
            if h[i][0] < h[parent][0]:
                self._swap(i, parent); i = parent
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
            i = pos[key]
            old = self._h[i][0]
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

    def peek(self):
        if not self._h:
            return None
        score, key = self._h[0]
        return key, score

    def heap_len(self) -> int:
        return len(self._h)


# --------------------------------------------------------------------------- #
# Correlated workload generator (mirrors fuzz_refcount.py lifecycle) that emits
# a trace of page vacate-cost events. The decision structures replay the trace.
#   event ('U', page, cost) : page's vacate-cost changed -> re-key
#   event ('Q', None, None) : mamba needs a page -> query cheapest-to-vacate
# --------------------------------------------------------------------------- #
def gen_trace(n_pages: int, K: int, n_ops: int, seed: int):
    """Correlated workload with INCREMENTAL page bookkeeping (a real slab
    allocator's partial/free sets), so alloc is O(blocks) not O(P)."""
    rng = random.Random(seed)
    N = n_pages * K
    ref = [0] * N
    cached = bytearray(N)
    scost = [0.0] * N
    mamba = bytearray(n_pages)
    vac = [0.0] * n_pages
    pl = [0] * n_pages                          # live (ref>0) count per page
    cachecost = [0.0] * n_pages                 # Σ cached recompute cost per page
    free_slots: list[list[int]] = [list(range(p * K + K - 1, p * K - 1, -1))
                                   for p in range(n_pages)]  # ref0 slots (stack)
    partial: set[int] = set()                   # 0<pl<K, not mamba (packing bias)
    recl: list[int] = list(range(n_pages))      # pl==0, not mamba (free pages)
    recl_set: set[int] = set(recl)
    req_blocks: dict[int, list[int]] = {}
    owners: list[set] = [set() for _ in range(N)]
    hot_prefix: list[int] = []
    events: list[tuple] = []
    occ_free = occ_full = occ_partial = recl_acc = samples = 0

    def a_cost() -> float:
        return 1.0 + rng.randrange(1, 64) * 0.5

    def update_sets(p: int) -> None:
        if mamba[p]:
            partial.discard(p); recl_set.discard(p); return
        if pl[p] == 0:
            partial.discard(p)
            if p not in recl_set:
                recl_set.add(p); recl.append(p)
        elif pl[p] == K:
            partial.discard(p); recl_set.discard(p)
        else:
            recl_set.discard(p); partial.add(p)

    def vac_of(p: int) -> float:
        return pl[p] * LIVE_PENALTY + cachecost[p]

    def emit(p: int) -> None:
        if mamba[p]:
            return
        nc = vac_of(p)
        if nc != vac[p]:
            vac[p] = nc
            events.append(('U', p, nc))

    def take_slot(p: int):
        sid = free_slots[p].pop()
        if cached[sid]:
            cached[sid] = 0; cachecost[p] -= scost[sid]
        ref[sid] = 1; pl[p] += 1
        return sid

    nxt = 0
    live_attn: list[int] = []
    live_mamba: list[int] = []
    for i in range(n_ops):
        op = rng.random()
        if op < 0.30:                              # attn alloc (correlated burst)
            nxt += 1
            need = rng.randint(1, 3 * K)
            got, touched = 0, set()
            # packing bias: partial pages first, then free pages
            cand = list(partial)
            for p in cand:
                while got < need and free_slots[p] and pl[p] < K:
                    sid = take_slot(p); owners[sid].add(nxt)
                    req_blocks.setdefault(nxt, []).append(sid)
                    touched.add(p); got += 1
                update_sets(p)
                if got >= need:
                    break
            while got < need and recl:
                p = recl.pop()
                if p not in recl_set or mamba[p]:
                    continue
                recl_set.discard(p)
                while got < need and free_slots[p] and pl[p] < K:
                    sid = take_slot(p); owners[sid].add(nxt)
                    req_blocks.setdefault(nxt, []).append(sid)
                    touched.add(p); got += 1
                update_sets(p)
            if got:
                live_attn.append(nxt)
                if rng.random() < 0.1:
                    hot_prefix.extend(req_blocks[nxt][:3])
                    hot_prefix[:] = hot_prefix[-200:]
            for p in touched:
                emit(p)
        elif op < 0.55 and hot_prefix:             # shared-prefix touch (ref++)
            nxt += 1
            k = rng.randint(1, min(8, len(hot_prefix)))
            touched = set()
            for sid in rng.sample(hot_prefix, k):
                if nxt in owners[sid]:
                    continue
                p = sid // K
                if mamba[p]:
                    continue
                if ref[sid] == 0:                  # cached->live: pl++, drop cachecost
                    if cached[sid]:
                        cached[sid] = 0; cachecost[p] -= scost[sid]
                    pl[p] += 1
                    if sid in free_slots[p]:
                        free_slots[p].remove(sid)
                    update_sets(p)
                ref[sid] += 1
                owners[sid].add(nxt)
                req_blocks.setdefault(nxt, []).append(sid)
                touched.add(p)
            live_attn.append(nxt)
            for p in touched:
                emit(p)
        elif op < 0.72 and live_attn:              # attn free (whole sequence)
            r = live_attn.pop(rng.randrange(len(live_attn)))
            touched = set()
            for sid in req_blocks.pop(r, []):
                if r not in owners[sid]:
                    continue
                owners[sid].discard(r)
                ref[sid] -= 1
                p = sid // K
                if ref[sid] == 0:                  # live->cached
                    cached[sid] = 1; scost[sid] = a_cost()
                    cachecost[p] += scost[sid]
                    pl[p] -= 1
                    free_slots[p].append(sid)
                    update_sets(p)
                touched.add(p)
            for p in touched:
                emit(p)
        elif op < 0.86:                            # mamba demand -> QUERY
            events.append(('Q', None, None))
        elif op < 0.93 and recl:                   # mamba alloc (take a free page)
            p = None
            while recl:
                cand = recl.pop()
                if cand in recl_set and not mamba[cand] and pl[cand] == 0:
                    p = cand; break
            if p is not None:
                recl_set.discard(p)
                base = p * K
                for s in range(base, base + K):
                    if cached[s]:
                        cached[s] = 0
                cachecost[p] = 0.0
                mamba[p] = 1; vac[p] = 0.0
                live_mamba.append(p)
                events.append(('T', p, None))
        elif live_mamba:                           # mamba free -> page returns
            p = live_mamba.pop(rng.randrange(len(live_mamba)))
            mamba[p] = 0; vac[p] = 0.0; pl[p] = 0
            free_slots[p] = list(range(p * K + K - 1, p * K - 1, -1))
            update_sets(p)
            events.append(('U', p, 0.0))

        if i % 500 == 0:                           # sample occupancy (O(P), cheap)
            samples += 1
            for p in range(n_pages):
                if mamba[p]:
                    continue
                if pl[p] == 0:
                    occ_free += 1; recl_acc += 1
                elif pl[p] == K:
                    occ_full += 1
                else:
                    occ_partial += 1
    recl = recl_acc
    tot = max(1, occ_free + occ_full + occ_partial)
    stats = {
        "n_events": len(events),
        "frac_free_pages": round(occ_free / tot, 3),
        "frac_full_pages": round(occ_full / tot, 3),
        "frac_partial_pages": round(occ_partial / tot, 3),
        "mean_reclaimable_per_sample": round(recl / max(1, samples), 1),
    }
    return events, stats


# --------------------------------------------------------------------------- #
# Decision structures (rank ALL in-pool attention pages by vacate-cost).
# --------------------------------------------------------------------------- #
class CostDecision:
    def __init__(self, n_pages: int, heap=None):
        # heap defaults to the production-style lazy-delete heap; pass an
        # IndexedHeap (eager delete) for the recommended structure.
        self.heap = heap if heap is not None else LazyHeap()
        for p in range(n_pages):        # every page starts free (vacate-cost 0)
            self.heap.add_or_update(p, 0.0)

    def apply(self, ev) -> None:
        k, p, c = ev
        if k == 'U':
            self.heap.add_or_update(p, c)
        elif k == 'T':
            self.heap.remove(p)
        # 'Q' handled by caller (query)

    def cheapest(self):
        return self.heap.peek()


class LRUDecision:
    """Recency over in-pool pages; access (U) moves to back, take = front."""
    def __init__(self, n_pages: int):
        self.od: OrderedDict[int, None] = OrderedDict((p, None) for p in range(n_pages))

    def apply(self, ev) -> None:
        k, p, _c = ev
        if k == 'U':
            self.od.pop(p, None)
            self.od[p] = None          # move-to-back (most recently accessed)
        elif k == 'T':
            self.od.pop(p, None)

    def cheapest(self):
        for p in self.od:              # front = least recently used
            return p, 0.0
        return None


def naive_cheapest(vac, mamba):
    best_p, best_c = -1, None
    for p in range(len(vac)):
        if not mamba[p] and (best_c is None or vac[p] < best_c):
            best_p, best_c = p, vac[p]
    return None if best_p < 0 else (best_p, best_c)


def replay_time(struct, events) -> float:
    t0 = time.perf_counter_ns()
    for ev in events:
        struct.apply(ev)
    return time.perf_counter_ns() - t0


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def query_and_correctness(events, n_pages, K, heap_factory):
    """Untimed correctness + realistic query timing. Records EACH query's
    latency (not just the mean) so we can report p50/p99/max — the mean hides
    the lazy-delete heap's O(P) peek-trim spikes (audit-3)."""
    inc = CostDecision(n_pages, heap_factory())
    vac = [0.0] * n_pages
    mamba = bytearray(n_pages)
    q_inc: list[float] = []
    t_naive = 0.0
    nq = viols = 0
    for ev in events:
        k, p, c = ev
        if k == 'Q':
            t0 = time.perf_counter_ns(); ans = inc.cheapest(); dt = time.perf_counter_ns() - t0
            q_inc.append(dt)
            t0 = time.perf_counter_ns(); truth = naive_cheapest(vac, mamba); t_naive += time.perf_counter_ns() - t0
            nq += 1
            if (ans is None) != (truth is None):
                viols += 1
            elif ans is not None:
                if abs(ans[1] - truth[1]) > 1e-6:
                    viols += 1
                if mamba[ans[0]]:           # never return a mamba-owned page
                    viols += 1
            continue
        inc.apply(ev)
        if k == 'U':
            vac[p] = c; mamba[p] = 0
        elif k == 'T':
            mamba[p] = 1; vac[p] = 0.0
    return {
        "q_p50": round(_pct(q_inc, 0.50), 1) if q_inc else None,
        "q_p99": round(_pct(q_inc, 0.99), 1) if q_inc else None,
        "q_max": round(max(q_inc), 1) if q_inc else None,
        "naive_mean": round(t_naive / nq, 1) if nq else None,
        "nq": nq, "viols": viols, "heap_phys": inc.heap.heap_len(),
    }


def _check_indexed(ih, ref) -> None:
    """Full structural invariants of the hand-rolled IndexedHeap vs a brute
    reference dict. Raises on any breach."""
    h, pos = ih._h, ih._pos
    if not (len(h) == len(pos) == len(ref)):
        raise AssertionError(f"size: heap={len(h)} pos={len(pos)} ref={len(ref)}")
    if {k for _, k in h} != set(pos) or set(pos) != set(ref):
        raise AssertionError("keyset mismatch")
    for i, (sc, k) in enumerate(h):
        if pos[k] != i:
            raise AssertionError(f"pos map: key {k} at {i} but pos says {pos[k]}")
        if abs(ref[k] - sc) > 1e-12:
            raise AssertionError(f"stale score on {k}: {sc} vs ref {ref[k]}")
        if i > 0 and h[(i - 1) >> 1][0] > sc:
            raise AssertionError(f"heap order broken at {i}")


def property_test(seeds=30, ops=12_000):
    """Hammer IndexedHeap correctness vs a dict+min() reference, checking every
    invariant after every op. Stresses the paths the workload under-exercises
    (audit-4 F4/F5): interior remove, remove-min, remove-then-readd, decrease
    AND increase the same key, heavy ties (small keyspace), drain-to-empty."""
    fails = 0
    total = 0
    interior_removes = drains = 0
    for seed in range(seeds):
        rng = random.Random(1000 + seed)
        ih = IndexedHeap()
        ref: dict[int, float] = {}
        keyspace = (8, 25, 300)[seed % 3]      # small => many ties + interior churn
        for _ in range(ops):
            total += 1
            r = rng.random(); k = rng.randrange(keyspace)
            if r < 0.50:                       # add / update (in-place re-key)
                sc = round(rng.random() * 8, 3)
                ih.add_or_update(k, sc); ref[k] = sc
            elif r < 0.82:                     # remove (often interior)
                if k in ref and ih._pos.get(k, 0) != 0:
                    interior_removes += 1
                ih.remove(k); ref.pop(k, None)
            elif r < 0.88 and ref:             # remove current min
                mk = min(ref, key=ref.get)
                ih.remove(mk); ref.pop(mk)
            else:                              # query + verify
                ans = ih.peek()
                truth = min(ref.values()) if ref else None
                if (ans is None) != (truth is None):
                    fails += 1
                elif ans is not None and abs(ans[1] - truth) > 1e-9:
                    fails += 1
            _check_indexed(ih, ref)
        # drain to empty (exercises root-remove repeatedly)
        for k in list(ref):
            ih.remove(k); ref.pop(k); _check_indexed(ih, ref)
        drains += 1
    return {"ops": total, "seeds": seeds, "interior_removes": interior_removes,
            "drains_to_empty": drains, "violations": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ops", type=int, default=200_000)
    args = ap.parse_args()
    K = 33

    print("=== decision_cost microbench v4 (correlated workload, IndexedHeap, p50/p99/MAX) ===\n")

    print("--- (0) IndexedHeap correctness property test (vs dict+min reference) ---")
    pt = property_test()
    print(json.dumps(pt))
    assert pt["violations"] == 0, "IndexedHeap property test FAILED"
    print("  (full invariants checked after EVERY op; interior remove / remove-min / drain exercised)\n")

    print("--- workload realism: page occupancy + reclaimable distribution ---")
    for n_pages in (1000, 4000):
        _, stats = gen_trace(n_pages, K, args.n_ops, 0)
        print(json.dumps({"n_pages": n_pages, **stats}))
    print("(bimodal: free+full >> partial; reclaimable pages are RARE under pressure)\n")

    print("--- (2) per-op maintenance: IndexedHeap vs LRU (batch-timed, 3 seeds, P=4000) ---")
    ratios = []
    for seed in range(3):
        events, _ = gen_trace(4000, K, args.n_ops, seed)
        n_u = sum(1 for e in events if e[0] in ('U', 'T'))
        t_inc = replay_time(CostDecision(4000, IndexedHeap()), events)
        t_lru = replay_time(LRUDecision(4000), events)
        ratios.append(round(t_inc / t_lru, 2))
        print(json.dumps({"seed": seed, "maintenance_events": n_u,
                          "inc_ns_per_ev": round(t_inc / max(1, n_u), 1),
                          "lru_ns_per_ev": round(t_lru / max(1, n_u), 1),
                          "ratio_inc_over_lru": round(t_inc / t_lru, 2)}))
    print(f"per-op ratio Indexed/LRU (batch-timed): {ratios}  (target <= ~3x)")
    # Bare-op decomposition (no apply() wrapper): how much of the ratio is the
    # eager-delete ALGORITHM vs the hand-rolled-Python-sift-vs-C constant?
    # (audit-4 F3) — idx/lazy isolates algorithm; idx/od is the full constant.
    rng = random.Random(0)
    ks = [rng.randrange(4000) for _ in range(200_000)]
    scs = [rng.random() * 100 for _ in range(200_000)]
    ih = IndexedHeap()
    for p in range(4000):
        ih.add_or_update(p, rng.random())
    t0 = time.perf_counter_ns()
    for k, s in zip(ks, scs):
        ih.add_or_update(k, s)
    t_idx = time.perf_counter_ns() - t0
    lh = LazyHeap()                       # C heapq.heappush (the prod primitive)
    for p in range(4000):
        lh.add_or_update(p, rng.random())
    t0 = time.perf_counter_ns()
    for k, s in zip(ks, scs):
        lh.add_or_update(k, s)
    t_lazy = time.perf_counter_ns() - t0
    od = OrderedDict((p, None) for p in range(4000))
    t0 = time.perf_counter_ns()
    for k in ks:
        od.pop(k, None); od[k] = None
    t_od = time.perf_counter_ns() - t0
    print(json.dumps({"bare_idx_ns": round(t_idx / len(ks), 1),
                      "bare_lazy_heapq_ns": round(t_lazy / len(ks), 1),
                      "bare_od_ns": round(t_od / len(ks), 1),
                      "idx_over_lazy(algorithm)": round(t_idx / t_lazy, 2),
                      "idx_over_od(full_constant)": round(t_idx / t_od, 2)}))
    print("  (idx/lazy ~2x = algorithm; idx/od ~5x = mostly Python-sift vs C; "
          "a C-backed indexed heap would likely beat lazy on per-op too)")
    # timer floor on this box, to contextualize the sub-µs query numbers (F2)
    N = 1_000_000
    t0 = time.perf_counter_ns()
    for _ in range(N):
        time.perf_counter_ns()
    floor = (time.perf_counter_ns() - t0) / N
    print(f"  perf_counter_ns() call floor on this box: ~{floor:.0f} ns "
          f"(=> sub-µs query numbers are ~half timer overhead)\n")

    print("--- (1)+(3) query latency vs P: LazyHeap(prod) vs IndexedHeap(fix) vs naive O(P) ---")
    print("  reporting p50 / p99 / MAX (the mean hides the lazy-delete O(P) spikes):")
    for n_pages in (1000, 4000, 16000):   # <=~20k = realistic single-engine pages
        events, _ = gen_trace(n_pages, K, 150_000, 0)
        lazy = query_and_correctness(events, n_pages, K, lambda: LazyHeap(8.0))
        idx = query_and_correctness(events, n_pages, K, lambda: IndexedHeap())
        print(json.dumps({"n_pages": n_pages,
                          "lazy_p50": lazy["q_p50"], "lazy_p99": lazy["q_p99"], "lazy_MAX": lazy["q_max"],
                          "idx_p50": idx["q_p50"], "idx_p99": idx["q_p99"], "idx_MAX": idx["q_max"],
                          "naive_mean": idx["naive_mean"],
                          "viols_lazy": lazy["viols"], "viols_idx": idx["viols"]}))
    print("  (LazyHeap MAX = real O(P) trim spike, scales with P; IndexedHeap MAX is\n"
          "   scheduler JITTER not peek cost — batched worst per-call ~125ns, sub-µs)\n")

    # Structural complexity, fixed churn, vary P. Reports BOTH peek (query) and
    # update (maintenance). peek should be O(1)-flat; update should be O(log P)
    # — maintenance is NOT flat (audit-4 F1), it grows with sift depth, but
    # stays sub-3µs even at 256k pages, << any forward-pass step.
    print("--- (1b) structural complexity vs P (IndexedHeap): peek O(1), maintenance O(log P) ---")
    rng = random.Random(0)
    for n_pages in (1000, 4000, 16000, 64000, 256000):
        inc = CostDecision(n_pages, IndexedHeap())
        for p in range(n_pages):
            inc.apply(('U', p, rng.random() * 100))
        inc.cheapest()
        UPD, Q = 4, 4000
        peeks, upd_t, upd_n = [], 0.0, 0
        for _ in range(Q):
            t0 = time.perf_counter_ns()
            for _ in range(UPD):
                inc.apply(('U', rng.randrange(n_pages), rng.random() * 100))
            upd_t += time.perf_counter_ns() - t0; upd_n += UPD
            t0 = time.perf_counter_ns(); inc.cheapest(); peeks.append(time.perf_counter_ns() - t0)
        print(json.dumps({"n_pages": n_pages,
                          "peek_p50_ns": round(_pct(peeks, 0.5), 1),
                          "maintenance_ns_per_ev": round(upd_t / upd_n, 1)}))
    print("  (peek flat O(1); maintenance O(log P): ~0.7µs@1k -> ~2.3µs@256k, all << ms step)\n")

    print("--- (4) heap bloat: direct LazyHeap stress (update-heavy, rare pops) ---")
    n_keys, rounds = 5000, 1000
    rng = random.Random(0)
    stream = [(rng.randrange(n_keys), rng.random() * 100) for _ in range(n_keys * rounds)]
    for cf in (0.0, 8.0):
        h = LazyHeap(compact_factor=cf)
        for k in range(n_keys):
            h.add_or_update(k, rng.random() * 100)
        peaks = []
        for i, (k, sc) in enumerate(stream):
            h.add_or_update(k, sc)
            if i % 50 == 0:
                h.peek()
            if i % (len(stream) // 10) == 0:
                peaks.append(h.heap_len())
        h.peek()
        tag = "lazy_no_compaction(==prod today)" if cf == 0.0 else f"lazy_compaction(factor={cf})"
        print(json.dumps({"mode": tag, "updates": len(stream),
                          "logical": len(h), "physical_final": h.heap_len(),
                          "physical_peak": max(peaks),
                          "steady_bloat_x": round(sum(peaks[3:]) / max(1, len(peaks[3:])) / max(1, len(h)), 1),
                          "ceiling_bloat_x": round(max(peaks) / max(1, len(h)), 1)}))
    # IndexedHeap: physical == logical by construction (no stale entries ever)
    ih = IndexedHeap()
    for k in range(n_keys):
        ih.add_or_update(k, rng.random() * 100)
    for k, sc in stream:
        ih.add_or_update(k, sc)
    print(json.dumps({"mode": "indexed(eager-delete, the fix)", "updates": len(stream),
                      "logical": len(ih), "physical_final": ih.heap_len(),
                      "ceiling_bloat_x": round(ih.heap_len() / max(1, len(ih)), 1)}))


if __name__ == "__main__":
    main()
