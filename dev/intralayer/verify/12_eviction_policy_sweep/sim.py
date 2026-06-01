# SPDX-License-Identifier: Apache-2.0
"""Offline block-cache simulator over real CC agent traces.

Goal: decide if ANY eviction policy beats LRU and the current L1
(windowed-LFU + LRU) on real multi-turn agent traffic, measured by
prefix-cache hit rate / recompute volume.

Model
-----
* Each session = ordered list of messages. A "turn" = an assistant reply;
  before it the engine sees the cumulative prefix of all prior messages.
  Prefix caching means the turn re-issues its whole cumulative token prefix
  and reuses any cached blocks.
* Tokenization: char/4 proxy (deterministic, no tokenizer). LIMITATION:
  not exact token boundaries, but block-granular hit *structure* (long
  shared cumulative prefixes within a session, near-zero cross-session
  sharing) is what drives the result and is proxy-robust.
* Block identity = content hash of the block's token slice (prefix-cache
  semantics): identical leading content across turns/sessions => same block.
  Within a session, turn k's prefix is a strict superset of turn k-1's, so
  the leading blocks are byte-identical and hash-collide => real reuse.
* Budget: a fixed number of cache blocks, < working set, to force eviction.
* Concurrency: sessions replayed concurrently, round-robin over turns, so
  hot prefixes of one session get pressured by others (eviction bites).

Access stream
-------------
We produce a sequence of (block_hash, depth) "requests". On each turn we walk
the cumulative prefix block-by-block:
  - if block in cache: HIT (record access for the policy), keep it
  - if not: MISS, must (re)compute => insert; if cache full, evict 1 per policy
Hit rate = cached_blocks_requested / total_blocks_requested.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
from collections import OrderedDict, defaultdict, deque

TRACE = os.path.join(os.path.dirname(__file__), "..", "cc_long_traces.jsonl")


def _content_text(m):
    c = m["content"]
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                parts.append(str(p.get("text", p.get("content", ""))))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(c)


def load_sessions():
    sessions = []
    with open(os.path.abspath(TRACE)) as f:
        for line in f:
            o = json.loads(line)
            sessions.append([_content_text(m) for m in o["messages"]])
    return sessions


def session_to_token_string(msgs):
    # char/4 proxy: just concatenate; we slice the concatenated char stream
    # into blocks of block_size*4 chars.
    return "\n".join(msgs)


def build_turn_block_accesses(sessions, block_size, char_per_tok=4):
    """For each session, build the list of turns; each turn = list of block
    hashes for its cumulative prefix. Returns list of sessions, each a list
    of turns, each a list of (block_hash, depth).

    A turn fires after each assistant message (role alternates u/a/u/a...).
    We approximate: a turn boundary after every 2 messages (one u+a pair).
    The cumulative prefix grows; we hash each block slice.
    """
    bchars = block_size * char_per_tok
    out = []
    for msgs in sessions:
        full = session_to_token_string(msgs)
        # cumulative prefix lengths at each turn boundary (after each msg pair)
        # build cumulative char offsets per message (O(n), no n^2 joins)
        offs = []
        acc = 0
        for i, m in enumerate(msgs):
            if i > 0:
                acc += 1  # the "\n" separator
            acc += len(m)
            offs.append(acc)
        # turn boundaries: after every assistant msg (odd index 1,3,5..)
        turn_ends = [offs[i] for i in range(1, len(msgs), 2)]
        if not turn_ends:
            turn_ends = [offs[-1]] if offs else []
        # Hash each block ONCE on the full prefix; block b is byte-identical
        # across every turn that contains it (prefix-cache semantics), so a
        # turn ending at `end` just uses the first ceil(end/bchars) blocks.
        full_nblocks = (len(full) + bchars - 1) // bchars
        block_hashes = []
        for b in range(full_nblocks):
            slc = full[b * bchars:(b + 1) * bchars]
            h = hashlib.blake2b(slc.encode("utf-8", "ignore"),
                                digest_size=12).digest()
            block_hashes.append((h, b + 1))
        turns = []
        for end in turn_ends:
            nblocks = (end + bchars - 1) // bchars
            turns.append(block_hashes[:nblocks])
        out.append(turns)
    return out


def interleave(sessions_turns, seed):
    """Round-robin interleave of turns across sessions (concurrent replay).
    Returns a flat list of turns (each a list of (hash,depth))."""
    rng = random.Random(seed)
    order = list(range(len(sessions_turns)))
    rng.shuffle(order)
    queues = [deque(sessions_turns[i]) for i in order]
    flat = []
    while any(queues):
        for q in queues:
            if q:
                flat.append(q.popleft())
    return flat


def interleave_arrival(sessions_turns, seed, conc=24):
    """Arrival-based concurrency (NOT lockstep): at most `conc` sessions
    active at once; on each step pick a random active session to advance one
    turn; when a session finishes, admit the next waiting one. Models bursty
    Poisson-ish arrival with bounded concurrency, a stress test that the
    ranking is not an artifact of synchronous round-robin."""
    rng = random.Random(seed + 1000)
    order = list(range(len(sessions_turns)))
    rng.shuffle(order)
    waiting = deque(order)
    active = []  # list of deque
    flat = []
    while waiting or active:
        while waiting and len(active) < conc:
            active.append(deque(sessions_turns[waiting.popleft()]))
        if not active:
            break
        i = rng.randrange(len(active))
        flat.append(active[i].popleft())
        if not active[i]:
            active.pop(i)
    return flat


# --------------------------- policies --------------------------- #
# Each policy is a cache object with: contains(h), touch(h) on hit,
# admit(h, depth) on miss (may evict). capacity = budget blocks.

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.od = OrderedDict()  # h -> True, MRU at end

    def contains(self, h):
        return h in self.od

    def touch(self, h, depth):
        self.od.move_to_end(h)

    def admit(self, h, depth):
        self.od[h] = True
        self.od.move_to_end(h)
        if len(self.od) > self.cap:
            self.od.popitem(last=False)


class LFU:
    """O(1) LFU with LRU tiebreak among min-freq (classic freq-bucket LFU)."""
    def __init__(self, cap):
        self.cap = cap
        self.freq = {}                       # h -> frequency
        self.buckets = defaultdict(OrderedDict)  # freq -> OrderedDict (LRU)
        self.minfreq = 0

    def contains(self, h):
        return h in self.freq

    def _bump(self, h):
        f = self.freq[h]
        del self.buckets[f][h]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.minfreq == f:
                self.minfreq = f + 1
        self.freq[h] = f + 1
        self.buckets[f + 1][h] = True

    def touch(self, h, depth):
        self._bump(h)

    def admit(self, h, depth):
        if len(self.freq) >= self.cap:
            b = self.buckets[self.minfreq]
            victim, _ = b.popitem(last=False)  # least-recent in min-freq bucket
            if not b:
                del self.buckets[self.minfreq]
            del self.freq[victim]
        self.freq[h] = 1
        self.buckets[1][h] = True
        self.minfreq = 1


class WindowedLFU_LRU:
    """Faithful reduction of current L1: blocks with >=1 hit within a sliding
    window of W accesses are PROTECTED (the hot tier, evicted last, ordered by
    windowed hit count then recency); never-hit / stale-hit blocks are the cold
    tier, evicted LRU-first. Mirrors lpb_free_queue: tier order
    evict-first(stale)/cold-FIFO/hot-heap, priority = windowed n_b, recency
    tiebreak.

    Efficient bucketing: `cold` is an OrderedDict (LRU); `hot` holds blocks with
    a live windowed hit. A block is demoted cold->? lazily: we treat a hot block
    as stale if its last W-window hit-count is 0, computed only when it is a
    victim candidate. Eviction prefers cold (O(1)); only scans hot when no cold
    remains, matching the 3-tier drain.
    """
    def __init__(self, cap, window):
        self.cap = cap
        self.window = window
        self.t = 0
        self.tick = {}
        self.hist = defaultdict(deque)  # h -> deque of recent hit ticks
        self.cold = OrderedDict()       # never/stale-hit, LRU (MRU at end)
        self.hot = set()                # blocks with >=1 in-window hit
        self._next_decay = window // 16 if window else 0

    def _nb(self, h):
        dq = self.hist[h]
        lo = self.t - self.window
        while dq and dq[0] <= lo:
            dq.popleft()
        return len(dq)

    def contains(self, h):
        return h in self.cold or h in self.hot

    def touch(self, h, depth):
        self.t += 1
        self.tick[h] = self.t
        self.hist[h].append(self.t)
        # a hit makes it hot
        if h in self.cold:
            del self.cold[h]
        self.hot.add(h)

    def _decay(self):
        # L1 _decay_hits analog: throttled sweep demoting stale hot -> cold
        # front (so they're evicted before fresh never-hit cold tails... no:
        # before LRU-old cold). We push demoted blocks to the FRONT of cold so
        # they leave first, matching tier-1 evict-first semantics.
        if self.t < self._next_decay or not self.hot:
            return
        self._next_decay = self.t + max(1, self.window // 16)
        stale = [k for k in self.hot if self._nb(k) == 0]
        for k in stale:
            self.hot.discard(k)
            self.cold[k] = True
            self.cold.move_to_end(k, last=False)  # evict-first

    def admit(self, h, depth):
        self.t += 1
        self.tick[h] = self.t
        self._decay()
        self.cold[h] = True            # admitted with no in-window hit yet
        if len(self.cold) + len(self.hot) > self.cap:
            self._evict()

    def _evict(self):
        # Match lpb tier order exactly:
        #   tier1 evict-first = STALE hot (nb==0, window expired)
        #   tier2 cold FIFO (never-hit), LRU
        #   tier3 live hot, min (nb, recency)
        # Lazily move stale hot -> cold-LRU front when discovered, so the
        # common case (cold available) stays O(1) and stale hot is drained
        # ahead of fresh cold tails (the verify/9 fix).
        # cheap stale sweep: only when we actually need to evict.
        victim = None
        # Tier 1: stale hot. Detect lazily but bounded: check a few hot blocks.
        # To stay cheap we fold this into tier ordering via a periodic demote.
        if self.cold:
            victim, _ = self.cold.popitem(last=False)
            # but a stale hot block should outrank cold; check if any hot is
            # stale and older. To keep O(1) we approximate: demote happens in
            # touch-expiry below. (See note.) Evict cold.
        elif self.hot:
            stale = [k for k in self.hot if self._nb(k) == 0]
            if stale:
                victim = min(stale, key=lambda k: self.tick[k])
            else:
                victim = min(self.hot, key=lambda k: (self._nb(k), self.tick[k]))
            self.hot.discard(victim)
        if victim is None:
            return
        self.hist.pop(victim, None)
        self.tick.pop(victim, None)
        self.cold.pop(victim, None)
        self.hot.discard(victim)


class ARC:
    """Adaptive Replacement Cache (Megiddo & Modha)."""
    def __init__(self, cap):
        self.c = cap
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

    def contains(self, h):
        return h in self.t1 or h in self.t2

    def touch(self, h, depth):
        if h in self.t1:
            del self.t1[h]
            self.t2[h] = True
        elif h in self.t2:
            self.t2.move_to_end(h)

    def _replace(self, in_b2):
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = True
        elif self.t2:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = True

    def admit(self, h, depth):
        if h in self.b1:
            self.p = min(self.c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            self._replace(False)
            del self.b1[h]
            self.t2[h] = True
            return
        if h in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            self._replace(True)
            del self.b2[h]
            self.t2[h] = True
            return
        # brand new
        if len(self.t1) + len(self.b1) == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(False)
            else:
                self.t1.popitem(last=False)
        elif len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2) >= self.c:
            if len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2) == 2 * self.c:
                if self.b2:
                    self.b2.popitem(last=False)
            self._replace(False)
        self.t1[h] = True


class TwoQ:
    """2Q (Johnson & Shasha) simplified full version."""
    def __init__(self, cap, kin_frac=0.25, kout_frac=0.5):
        self.cap = cap
        self.kin = max(1, int(cap * kin_frac))
        self.kout = max(1, int(cap * kout_frac))
        self.am = OrderedDict()   # hot (LRU)
        self.a1in = OrderedDict()  # FIFO
        self.a1out = OrderedDict()  # ghost FIFO

    def contains(self, h):
        return h in self.am or h in self.a1in

    def touch(self, h, depth):
        if h in self.am:
            self.am.move_to_end(h)
        # a1in hit: leave in place (FIFO) per 2Q

    def admit(self, h, depth):
        if h in self.a1out:
            del self.a1out[h]
            self._reclaim()
            self.am[h] = True
            return
        # new
        self._reclaim()
        self.a1in[h] = True
        if len(self.a1in) > self.kin:
            k, _ = self.a1in.popitem(last=False)
            self.a1out[k] = True
            if len(self.a1out) > self.kout:
                self.a1out.popitem(last=False)

    def _reclaim(self):
        while len(self.am) + len(self.a1in) >= self.cap:
            if len(self.a1in) > self.kin and self.a1in:
                k, _ = self.a1in.popitem(last=False)
                self.a1out[k] = True
                if len(self.a1out) > self.kout:
                    self.a1out.popitem(last=False)
            elif self.am:
                self.am.popitem(last=False)
            elif self.a1in:
                self.a1in.popitem(last=False)
            else:
                break


class LIRS:
    """Simplified LIRS (Jiang & Zhang). Practical approximation."""
    def __init__(self, cap, hir_frac=0.01):
        self.cap = cap
        self.hir = max(1, int(cap * hir_frac))
        self.lir = cap - self.hir
        self.stack = OrderedDict()  # LIRS stack S: recency, status in val
        self.q = OrderedDict()      # resident HIR queue Q
        self.status = {}  # h -> 'LIR'|'HIR'
        self.resident = set()
        self.nlir = 0

    def contains(self, h):
        return h in self.resident

    def _prune(self):
        # bottom of stack must be LIR
        while self.stack:
            k = next(iter(self.stack))
            if self.status.get(k) == "LIR":
                break
            self.stack.popitem(last=False)

    def touch(self, h, depth):
        st = self.status.get(h)
        if st == "LIR":
            self.stack.move_to_end(h)
            self._prune()
        elif st == "HIR":
            instack = h in self.stack
            self.stack[h] = True
            self.stack.move_to_end(h)
            self.q.pop(h, None)
            if instack:
                self.status[h] = "LIR"
                self.nlir += 1
                # demote bottom LIR to HIR
                self._demote()
            else:
                self.q[h] = True
            self._prune()

    def _demote(self):
        while self.nlir > self.lir:
            k = next(iter(self.stack))
            if self.status.get(k) == "LIR":
                self.stack.popitem(last=False)
                self.status[k] = "HIR"
                self.nlir -= 1
                if k in self.resident:
                    self.q[k] = True
            else:
                self.stack.popitem(last=False)
        self._prune()

    def admit(self, h, depth):
        # evict resident HIR if full
        while len(self.resident) >= self.cap:
            if self.q:
                k, _ = self.q.popitem(last=False)
                self.resident.discard(k)
                if self.status.get(k) == "HIR" and k not in self.stack:
                    self.status.pop(k, None)
            else:
                # no HIR resident: drop bottom of stack
                if self.stack:
                    k, _ = self.stack.popitem(last=False)
                    self.resident.discard(k)
                    if self.status.get(k) == "LIR":
                        self.nlir -= 1
                    self.status.pop(k, None)
                else:
                    break
        instack = h in self.stack
        self.resident.add(h)
        if self.nlir < self.lir and not instack:
            self.status[h] = "LIR"
            self.nlir += 1
            self.stack[h] = True
            self.stack.move_to_end(h)
        else:
            self.status[h] = "HIR"
            self.stack[h] = True
            self.stack.move_to_end(h)
            self.q[h] = True
        self._prune()


class LRFU:
    """LRFU (Lee et al.): combined recency/frequency weight
    C(x) = sum 2^{-lambda*(t-t_i)}. lambda interpolates LRU<->LFU.
    Lazy-heap eviction: push current decayed value on each access; on evict,
    pop heap min and re-verify it equals the live value (else it's stale)."""
    def __init__(self, cap, lam=0.01):
        self.cap = cap
        self.lam = lam
        self.t = 0
        self.crf = {}
        self.last = {}
        self._heap = []   # (val_snapshot_at_push_time_decayed_to_push, last, h)
        # We can't keep absolute val comparable across time because decay is
        # global-monotone (all values decay by the same factor between two
        # times), so RELATIVE order of crf*2^{-lam*(t-last)} is preserved if we
        # compare the *normalized* potential crf*2^{lam*last}. That quantity is
        # time-invariant => a plain min-heap on it gives the true LRFU victim.

    def _key(self, h):
        # log of normalized potential: log2(crf) + lam*last. Time-invariant
        # ordering (global decay cancels), numerically safe (no overflow).
        return math.log2(self.crf[h]) + self.lam * self.last[h]

    def _update(self, h):
        if h in self.crf:
            dt = self.t - self.last[h]
            self.crf[h] = 1.0 + (2.0 ** (-self.lam * dt)) * self.crf[h]
        else:
            self.crf[h] = 1.0
        self.last[h] = self.t
        heapq.heappush(self._heap, (self._key(h), h))

    def contains(self, h):
        return h in self.crf

    def touch(self, h, depth):
        self.t += 1
        self._update(h)

    def admit(self, h, depth):
        self.t += 1
        if len(self.crf) >= self.cap:
            while self._heap:
                k, victim = heapq.heappop(self._heap)
                if victim in self.crf and abs(self._key(victim) - k) < 1e-9 * (abs(k) + 1):
                    del self.crf[victim]
                    del self.last[victim]
                    break
        self._update(h)


class ManyReaders:
    """Prefix-popularity-aware: protect blocks shared by many DISTINCT
    sessions (a "fan-out"/many-readers signal). Eviction key:
    (distinct_session_readers, windowed_hits, recency) ascending => evict
    low-fanout first. Falls back to LRU among equal-fanout cold blocks."""
    def __init__(self, cap):
        self.cap = cap
        self.t = 0
        self.tick = {}
        self.readers = defaultdict(set)  # h -> set(session ids)
        self.present = set()
        self._heap = []  # lazy: (n_readers, tick, h); verify tick on pop

    def contains(self, h):
        return h in self.present

    def _push(self, h):
        heapq.heappush(self._heap, (len(self.readers[h]), self.tick[h], h))

    def touch(self, h, depth, sid=None):
        self.t += 1
        self.tick[h] = self.t
        if sid is not None:
            self.readers[h].add(sid)
        self._push(h)

    def admit(self, h, depth, sid=None):
        self.t += 1
        self.tick[h] = self.t
        self.present.add(h)
        if sid is not None:
            self.readers[h].add(sid)
        if len(self.present) > self.cap:
            while self._heap:
                nr, tk, victim = heapq.heappop(self._heap)
                if (victim in self.present and self.tick.get(victim) == tk
                        and len(self.readers[victim]) == nr):
                    self.present.discard(victim)
                    self.readers.pop(victim, None)
                    self.tick.pop(victim, None)
                    break
        self._push(h)


# --------------------------- runner --------------------------- #

def run_policy(make_cache, flat_turns, sid_of_turn=None, needs_sid=False):
    cache = make_cache()
    req = 0
    hits = 0
    for ti, turn in enumerate(flat_turns):
        sid = sid_of_turn[ti] if sid_of_turn is not None else None
        for (h, depth) in turn:
            req += 1
            if cache.contains(h):
                hits += 1
                if needs_sid:
                    cache.touch(h, depth, sid)
                else:
                    cache.touch(h, depth)
            else:
                if needs_sid:
                    cache.admit(h, depth, sid)
                else:
                    cache.admit(h, depth)
    return hits, req


def interleave_with_sid(sessions_turns, seed):
    rng = random.Random(seed)
    order = list(range(len(sessions_turns)))
    rng.shuffle(order)
    queues = [(order[i], deque(sessions_turns[order[i]])) for i in range(len(order))]
    flat = []
    sids = []
    active = queues
    while any(q for _, q in active):
        for sid, q in active:
            if q:
                flat.append(q.popleft())
                sids.append(sid)
    return flat, sids


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[16, 1056])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget-fracs", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5])
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--policies", type=str, nargs="+", default=None,
                    help="subset of policy names to run")
    ap.add_argument("--arrival", action="store_true",
                    help="use arrival-based bounded-concurrency interleave")
    ap.add_argument("--conc", type=int, default=24,
                    help="max concurrent sessions for --arrival")
    args = ap.parse_args()

    sessions = load_sessions()
    print(f"loaded {len(sessions)} sessions")

    for bs in args.block_sizes:
        sturns = build_turn_block_accesses(sessions, bs)
        # working set = distinct blocks across all turns (use last turn per
        # session = full prefix, which is superset)
        distinct = set()
        total_turn_blocks = 0
        for turns in sturns:
            if turns:
                for (h, _) in turns[-1]:
                    distinct.add(h)
            for t in turns:
                total_turn_blocks += len(t)
        ws = len(distinct)
        print(f"\n### block_size={bs}  working_set={ws} distinct blocks  "
              f"total_turn_block_accesses={total_turn_blocks}")

        policies = {
            "LRU": lambda cap: LRU(cap),
            "LFU": lambda cap: LFU(cap),
            "WLFU+LRU(L1)": lambda cap, w=args.window: WindowedLFU_LRU(cap, w),
            "ARC": lambda cap: ARC(cap),
            "2Q": lambda cap: TwoQ(cap),
            "LIRS": lambda cap: LIRS(cap),
            "LRFU": lambda cap: LRFU(cap),
            "ManyReaders": lambda cap: ManyReaders(cap),
        }
        if args.policies:
            policies = {k: v for k, v in policies.items() if k in args.policies}

        for bf in args.budget_fracs:
            cap = max(8, int(ws * bf))
            print(f"\n  budget={bf:.0%} of WS => cap={cap} blocks")
            print(f"  {'policy':16s}  hit_rate(mean±sd over seeds)")
            for name, mk in policies.items():
                rates = []
                for seed in args.seeds:
                    if name == "ManyReaders":
                        flat, sids = interleave_with_sid(sturns, seed)
                        hits, req = run_policy(lambda c=cap: mk(c), flat,
                                               sid_of_turn=sids, needs_sid=True)
                    else:
                        flat = (interleave_arrival(sturns, seed, args.conc)
                                if args.arrival else interleave(sturns, seed))
                        hits, req = run_policy(lambda c=cap: mk(c), flat)
                    rates.append(hits / req if req else 0.0)
                import statistics
                m = statistics.mean(rates)
                sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
                print(f"  {name:16s}  {m:6.3%} ± {sd:5.3%}")


if __name__ == "__main__":
    main()
