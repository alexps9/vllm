"""Phase 2 — sub_block_allocator: prove the two-level allocator can be
memory-safe.

A standalone PROTOTYPE (not the vLLM integration — that's implementation) of
the design's data structure:

  * physical pool of `n_pages` pages, each = `K` sub-blocks (K = 1056/ksize).
  * a page is in exactly one mode: FREE / ATTN (≥1 sub-block held) / MAMBA
    (whole page held by one mamba state).
  * attention allocates sub-blocks with a PACKING BIAS (fill partially-used
    ATTN pages before opening a FREE page → keeps whole pages free for mamba).
  * a page returns to FREE (mamba-usable) iff ALL its sub-blocks are free.

We fuzz it with ≥1e6 randomized ops + an adversarial max-scatter phase and
assert ZERO invariant violations (no two live owners alias a sub-block; no
use-after-free; mode/occupancy consistent; conservation holds). Mamba
starvation (no fully-free page when mamba needs one) is RECORDED, not fixed —
that is the cost-model's job (phase 3); here it must merely never corrupt
state.
"""
from __future__ import annotations

import argparse
import random


class Viol(Exception):
    pass


class TwoLevelAllocator:
    FREE, ATTN, MAMBA = 0, 1, 2

    def __init__(self, n_pages: int, K: int):
        self.n_pages = n_pages
        self.K = K
        self.mode = [self.FREE] * n_pages
        self.attn: list[dict[int, int]] = [dict() for _ in range(n_pages)]  # page -> {slot: req}
        self.mamba_owner: list[int | None] = [None] * n_pages
        self.free_pages: set[int] = set(range(n_pages))      # fully-free
        self.attn_open: set[int] = set()                     # ATTN with a free slot
        self.attn_handles: dict[int, list[tuple[int, int]]] = {}  # req -> [(page,slot)]
        self.mamba_pages: dict[int, int] = {}                # req -> page
        self.events = {"attn_oom": 0, "mamba_starve": 0}

    # ---- attention ----
    def attn_alloc(self, req: int, n: int) -> bool:
        got: list[tuple[int, int]] = []
        while n > 0:
            if self.attn_open:                       # packing bias: fill partial pages first
                p = next(iter(self.attn_open))
            elif self.free_pages:
                p = self.free_pages.pop()
                self.mode[p] = self.ATTN
            else:
                self.events["attn_oom"] += 1         # no room (real design: evict/defer)
                break
            occ = self.attn[p]
            for slot in range(self.K):
                if n == 0:
                    break
                if slot in occ:
                    continue
                if occ.get(slot) is not None:        # O(1) safety: slot must be free
                    raise Viol(f"alloc into occupied slot {p}:{slot}")
                occ[slot] = req
                got.append((p, slot))
                n -= 1
            if len(occ) >= self.K:
                self.attn_open.discard(p)
            else:
                self.attn_open.add(p)
        self.attn_handles.setdefault(req, []).extend(got)
        return n == 0

    def attn_free(self, req: int) -> None:
        for (p, slot) in self.attn_handles.pop(req, []):
            occ = self.attn[p]
            if occ.get(slot) != req:                 # use-after-free / wrong-owner guard
                raise Viol(f"free of unowned slot {p}:{slot} by req {req}")
            del occ[slot]
            if not occ:                              # page emptied -> flips to FREE
                self.mode[p] = self.FREE
                self.attn_open.discard(p)
                self.free_pages.add(p)
            else:
                self.attn_open.add(p)

    # ---- mamba ----
    def mamba_alloc(self, req: int) -> bool:
        if not self.free_pages:                      # needs a whole free page
            self.events["mamba_starve"] += 1
            return False
        p = self.free_pages.pop()
        self.mode[p] = self.MAMBA
        self.mamba_owner[p] = req
        self.mamba_pages[req] = p
        return True

    def mamba_free(self, req: int) -> None:
        p = self.mamba_pages.pop(req, None)
        if p is None:
            return
        if self.mamba_owner[p] != req:
            raise Viol(f"mamba free of unowned page {p} by {req}")
        self.mamba_owner[p] = None
        self.mode[p] = self.FREE
        self.free_pages.add(p)

    # ---- full invariant check (periodic) ----
    def check(self) -> None:
        seen: set[tuple[int, int]] = set()
        attn_occ = 0
        for p in range(self.n_pages):
            m, occ, mo = self.mode[p], self.attn[p], self.mamba_owner[p]
            if m == self.FREE:
                if occ or mo is not None:
                    raise Viol(f"FREE page {p} not empty")
                if p not in self.free_pages:
                    raise Viol(f"FREE page {p} missing from free_pages")
            elif m == self.ATTN:
                if not (1 <= len(occ) <= self.K) or mo is not None:
                    raise Viol(f"ATTN page {p} bad occupancy/mamba")
                if (len(occ) < self.K) != (p in self.attn_open):
                    raise Viol(f"attn_open membership wrong for {p}")
                if p in self.free_pages:
                    raise Viol(f"ATTN page {p} also in free_pages")
                for slot, r in occ.items():
                    if (p, slot) in seen:
                        raise Viol(f"sub-block {p}:{slot} double-owned")
                    seen.add((p, slot))
                    if (p, slot) not in self.attn_handles.get(r, []):
                        raise Viol(f"slot {p}:{slot} not in owner {r}'s handles")
                attn_occ += len(occ)
            elif m == self.MAMBA:
                if occ or mo is None:
                    raise Viol(f"MAMBA page {p} has attn slots / no owner")
                if p in self.free_pages:
                    raise Viol(f"MAMBA page {p} also in free_pages")
        # conservation
        h = sum(len(v) for v in self.attn_handles.values())
        if h != attn_occ:
            raise Viol(f"handle count {h} != occupied {attn_occ}")
        n_free = sum(1 for m in self.mode if m == self.FREE)
        n_attn = sum(1 for m in self.mode if m == self.ATTN)
        n_mamba = sum(1 for m in self.mode if m == self.MAMBA)
        if n_free + n_attn + n_mamba != self.n_pages:
            raise Viol("page-mode counts don't sum to n_pages")
        if n_free != len(self.free_pages):
            raise Viol(f"free_pages size {len(self.free_pages)} != FREE pages {n_free}")
        if n_mamba != len(self.mamba_pages):
            raise Viol(f"mamba_pages {len(self.mamba_pages)} != MAMBA pages {n_mamba}")


def fuzz(n_pages=2000, K=33, n_ops=1_000_000, seed=0):
    rng = random.Random(seed)
    al = TwoLevelAllocator(n_pages, K)
    live_attn: list[int] = []
    live_mamba: list[int] = []
    nxt = 0
    viol = 0
    for i in range(n_ops):
        op = rng.random()
        if op < 0.35:                                # attn alloc
            nxt += 1
            if al.attn_alloc(nxt, rng.randint(1, 4 * K)):
                live_attn.append(nxt)
            elif nxt in al.attn_handles:             # partial alloc still tracked
                live_attn.append(nxt)
        elif op < 0.55 and live_attn:                # attn free
            al.attn_free(live_attn.pop(rng.randrange(len(live_attn))))
        elif op < 0.80:                              # mamba alloc
            nxt += 1
            if al.mamba_alloc(nxt):
                live_mamba.append(nxt)
        elif live_mamba:                             # mamba free
            al.mamba_free(live_mamba.pop(rng.randrange(len(live_mamba))))
        if i % 2000 == 0:
            al.check()
    al.check()

    # adversarial: GENUINE max-scatter on a FRESH allocator.
    # Fill every page with K size-1 reqs (each req owns exactly 1 slot), then
    # free all-but-one per page -> every page has exactly 1/K slot used, NONE
    # fully free -> mamba must starve despite (K-1)*n_pages free sub-blocks.
    # Then free one page's last slot -> it flips FREE -> mamba recovers.
    from collections import defaultdict
    adv = TwoLevelAllocator(n_pages, K)
    rid = 0
    fill = []
    for _ in range(n_pages * K):
        rid += 1
        adv.attn_alloc(rid, 1)
        fill.append(rid)
    adv.check()
    assert len(adv.free_pages) == 0, "fill did not saturate"
    starve_when_full = adv.mamba_alloc(10**8) is False        # no free page at all
    by_page = defaultdict(list)
    for r in fill:
        by_page[adv.attn_handles[r][0][0]].append(r)
    kept = {}
    for p, rs in by_page.items():
        kept[p] = rs[0]
        for r in rs[1:]:
            adv.attn_free(r)                                  # leave exactly 1 slot/page
    adv.check()
    starve_max_scatter = adv.mamba_alloc(10**8 + 1) is False  # 1/K used everywhere -> starve
    adv.check()
    adv.attn_free(kept[next(iter(kept))])                     # empty ONE page -> flips FREE
    adv.check()
    recovered = adv.mamba_alloc(10**8 + 2)                    # now a whole page is free
    adv.check()

    return {
        "n_pages": n_pages, "K": K, "n_ops": n_ops,
        "invariant_violations": viol,               # raised as exceptions if any
        "attn_oom_events": al.events["attn_oom"],
        "mamba_starve_events": al.events["mamba_starve"],
        "adv_starve_when_full": starve_when_full,            # expect True
        "adv_starve_max_scatter": starve_max_scatter,        # expect True (1/K used, no free page)
        "adv_mamba_recovered_after_freeing_one_page": recovered,  # expect True
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=1_000_000)
    ap.add_argument("--pages", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import json
    allres = []
    for s in range(args.seeds):
        r = fuzz(n_pages=args.pages, n_ops=args.ops, seed=s)
        r["seed"] = s
        allres.append(r)
        print(f"seed {s}: OK — {json.dumps(r)}")
    print("\nALL PASS — zero invariant violations across", args.seeds, "seeds")
