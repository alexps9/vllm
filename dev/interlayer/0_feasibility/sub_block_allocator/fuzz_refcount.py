"""Phase 2 (v2, post-audit) — two-level allocator with REAL vLLM semantics:
per-sub-block ref-counting + cached-block lifecycle + shared prefixes +
append-only block ids. The v1 fuzz only modelled single-owner sub-blocks; the
audit correctly flagged that the design's named net-new structures
(ref_cnt>1 via prefix sharing, decrement-to-zero, cached-but-unreferenced
eviction) were untested. This models them faithfully and fuzzes for safety.

Model (mirrors block_pool.py):
  * Sub-block id = page*K + slot — STABLE / append-only (never relocated;
    reclaim is by eviction, never by moving a block — block_pool.py:48-52).
  * ref_cnt[id]: number of live requests referencing the sub-block.
  * cached[id]: sub-block holds a prefix-cache entry (evictable when ref_cnt==0).
  * "reclaimable" = ref_cnt==0 (whether cached or not — evicting a cached one
    just costs recompute later; for SAFETY it is free to take).
  * A page is mamba-usable iff ALL K sub-blocks have ref_cnt==0 (any cached
    ones on it are evicted as part of the take).
  * Prefix sharing: a shared prefix block is touched by multiple requests →
    ref_cnt = #live referencers. free decrements; at 0 it becomes cached
    (kept for reuse), evicted on demand.

Invariants (checked every op on a short trace, periodically on long traces):
  ref_cnt>=0; ref_cnt == #live owners; no NEW-alloc of a ref_cnt>0 block;
  reclaimable set == {ref_cnt==0}; page mamba-usable iff all ref0; cached only
  when ref_cnt==0 OR live (a live block may also be cache-mapped); end-of-trace
  leak check (all ref0, all pages returned). Violations raise.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict


class Viol(Exception):
    pass


class RefAllocator:
    def __init__(self, n_pages: int, K: int):
        self.n_pages, self.K = n_pages, K
        self.N = n_pages * K
        self.ref = [0] * self.N            # ref_cnt per sub-block id (append-only id)
        self.cached: set[int] = set()      # ids with a cache entry (ref_cnt may be 0 or >0)
        self.mamba_owner: list[int | None] = [None] * n_pages
        # owner bookkeeping for invariant cross-check: id -> Counter(req->count? no, set of reqs)
        self.owners: dict[int, set[int]] = defaultdict(set)   # id -> set(req) currently referencing
        self.req_blocks: dict[int, list[int]] = defaultdict(list)  # req -> [ids it holds (with mult)]
        self.evicts = 0
        self.mamba_starve = 0

    def page_of(self, sid: int) -> int:
        return sid // self.K

    def _page_all_free(self, p: int) -> bool:
        base = p * self.K
        return all(self.ref[base + s] == 0 for s in range(self.K))

    def _free_ids_in_page(self, p: int):
        base = p * self.K
        return [base + s for s in range(self.K) if self.ref[base + s] == 0
                and self.mamba_owner[p] is None]

    # ---- attention: allocate n NEW sub-blocks (ref 0->1) ----
    def attn_alloc(self, req: int, n: int) -> int:
        got = 0
        # packing bias: prefer pages that already have live attn sub-blocks
        # (>=1 ref>0) and a free slot, before opening an all-free page.
        def page_partial(p):
            base = p * self.K
            has_live = any(self.ref[base + s] > 0 for s in range(self.K))
            has_free = any(self.ref[base + s] == 0 for s in range(self.K))
            return has_live and has_free and self.mamba_owner[p] is None
        order = [p for p in range(self.n_pages) if page_partial(p)] + \
                [p for p in range(self.n_pages)
                 if self.mamba_owner[p] is None and self._page_all_free(p)]
        for p in order:
            if got >= n:
                break
            for sid in self._free_ids_in_page(p):
                if got >= n:
                    break
                if self.ref[sid] != 0:
                    raise Viol(f"alloc NEW into ref>0 block {sid}")
                if sid in self.cached:        # reusing a cached-but-free block -> evict it
                    self.cached.discard(sid)
                    self.evicts += 1
                self.ref[sid] = 1
                self.owners[sid].add(req)
                self.req_blocks[req].append(sid)
                got += 1
        return got

    # ---- prefix sharing: req TOUCHES existing blocks (cache hit) -> ref++ ----
    def attn_touch(self, req: int, ids: list[int]) -> None:
        for sid in ids:
            if req in self.owners[sid]:
                continue                       # a req references a block once
            self.ref[sid] += 1
            self.owners[sid].add(req)
            self.req_blocks[req].append(sid)
            self.cached.add(sid)               # shared prefix blocks are cache-mapped

    # ---- free: ref-- ; at 0 -> becomes cached (kept for reuse) ----
    def attn_free(self, req: int) -> None:
        for sid in self.req_blocks.pop(req, []):
            if req not in self.owners[sid]:
                raise Viol(f"free of unowned block {sid} by {req}")
            self.owners[sid].discard(req)
            self.ref[sid] -= 1
            if self.ref[sid] < 0:
                raise Viol(f"ref_cnt<0 on {sid}")
            if self.ref[sid] == 0:
                self.cached.add(sid)           # freed-but-cached (evictable)

    # ---- mamba: needs a whole page of ref0 blocks; evict cached on it ----
    def mamba_alloc(self, req: int) -> bool:
        for p in range(self.n_pages):
            if self.mamba_owner[p] is None and self._page_all_free(p):
                base = p * self.K
                for s in range(self.K):       # evict any cached entries on the page
                    sid = base + s
                    if sid in self.cached:
                        self.cached.discard(sid)
                        self.evicts += 1
                self.mamba_owner[p] = req
                return True
        self.mamba_starve += 1
        return False

    def mamba_free(self, req: int) -> None:
        for p in range(self.n_pages):
            if self.mamba_owner[p] == req:
                self.mamba_owner[p] = None
                return

    # ---- invariants ----
    def check(self) -> None:
        for sid in range(self.N):
            if self.ref[sid] != len(self.owners[sid]):
                raise Viol(f"ref_cnt {self.ref[sid]} != #owners {len(self.owners[sid])} on {sid}")
            if self.ref[sid] < 0:
                raise Viol(f"ref<0 on {sid}")
            p = self.page_of(sid)
            if self.mamba_owner[p] is not None and self.ref[sid] != 0:
                raise Viol(f"MAMBA page {p} has attn ref on {sid}")
        # cached ids must be valid ids; a cached id may be live (shared) or free
        for sid in self.cached:
            if not (0 <= sid < self.N):
                raise Viol(f"bad cached id {sid}")
        # owner/handle consistency: every (req in owners[sid]) holds sid
        for sid, reqs in self.owners.items():
            for r in reqs:
                if self.req_blocks[r].count(sid) < 1:
                    raise Viol(f"owner {r} of {sid} missing handle")
        # mamba pages distinct
        owned = [o for o in self.mamba_owner if o is not None]
        if len(owned) != len(set(owned)):
            raise Viol("duplicate mamba page owner")

    def leak_check(self) -> None:
        # after freeing everything, nothing should be referenced
        if any(r != 0 for r in self.ref):
            raise Viol("leak: sub-block still referenced after full free")
        if any(o is not None for o in self.mamba_owner):
            raise Viol("leak: mamba page still owned after full free")


def fuzz(n_pages=400, K=33, n_ops=200_000, seed=0, check_every=1):
    rng = random.Random(seed)
    al = RefAllocator(n_pages, K)
    live_attn: list[int] = []
    live_mamba: list[int] = []
    # a small set of "hot prefix" sub-block ids that requests will SHARE (touch)
    hot_prefix: list[int] = []
    nxt = 0
    max_refcnt = 0
    shared_touches = 0
    for i in range(n_ops):
        op = rng.random()
        if op < 0.30:                         # attn alloc (new blocks)
            nxt += 1
            g = al.attn_alloc(nxt, rng.randint(1, 3 * K))
            if g:
                live_attn.append(nxt)
                # promote a few of this req's blocks to the shared "hot prefix"
                if rng.random() < 0.1 and al.req_blocks[nxt]:
                    hot_prefix.extend(al.req_blocks[nxt][:3])
                    hot_prefix[:] = hot_prefix[-200:]
        elif op < 0.55 and hot_prefix:        # SHARED-PREFIX hit: touch hot blocks
            nxt += 1
            k = rng.randint(1, min(8, len(hot_prefix)))
            ids = rng.sample(hot_prefix, k)
            al.attn_touch(nxt, ids)
            live_attn.append(nxt)
            shared_touches += 1
        elif op < 0.72 and live_attn:         # attn free
            al.attn_free(live_attn.pop(rng.randrange(len(live_attn))))
        elif op < 0.90:                       # mamba alloc
            nxt += 1
            if al.mamba_alloc(nxt):
                live_mamba.append(nxt)
        elif live_mamba:                      # mamba free
            al.mamba_free(live_mamba.pop(rng.randrange(len(live_mamba))))
        if al.ref:
            max_refcnt = max(max_refcnt, max(al.ref) if i % 50 == 0 else max_refcnt)
        if check_every and i % check_every == 0:
            al.check()
    al.check()
    # drain everything -> leak check
    for r in list(live_attn):
        al.attn_free(r)
    for r in list(live_mamba):
        al.mamba_free(r)
    al.check()
    al.leak_check()
    return {
        "n_pages": n_pages, "K": K, "n_ops": n_ops, "seed": seed,
        "check_every": check_every,
        "max_ref_cnt_seen": max(al.ref) if al.ref else 0,  # 0 after drain
        "max_ref_cnt_during": max_refcnt,                  # >1 proves sharing exercised
        "shared_touch_ops": shared_touches,
        "evicts": al.evicts,
        "mamba_starve": al.mamba_starve,
        "leak_free": True,                                 # leak_check passed (else raised)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import json
    # 1) short trace, check EVERY op (closes the transient-violation window)
    r0 = fuzz(n_pages=200, K=33, n_ops=20_000, seed=0, check_every=1)
    print("every-op check (K=33):", json.dumps(r0))
    # 2) long traces, periodic check, sweep K in {33, 66}
    for K in (33, 66):
        for s in range(args.seeds):
            r = fuzz(n_pages=400, K=K, n_ops=args.ops, seed=s, check_every=500)
            assert r["max_ref_cnt_during"] >= 2, "sharing not exercised (ref never >1)!"
            print(f"K={K} seed={s}:", json.dumps(r))
    print("\nALL PASS — zero violations; ref_cnt>1 sharing + cached eviction + leak check exercised")
