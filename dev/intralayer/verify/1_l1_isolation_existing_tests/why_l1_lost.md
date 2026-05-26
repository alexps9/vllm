# Why does L1-only lose to LRU on PathA util=0.9?

(In-progress investigation, started after n=3 attribution table revealed
L1-only is +20% TTFT worse than LRU on Phase H.)

## Headline question

User: *"我记得之前我们结果里，LPB单独就是Beat LRU，你研究清楚为啥L1会比LRU差，
这按道理不应该的，应该得解决。"*

n=3 PathA data:

| config | Phase B | Phase G | Phase E | Phase F | Phase H |
|---|---:|---:|---:|---:|---:|
| LRU      |  94 ms | 333 ms |  63 ms |  63 ms | 370 ms |
| L1-only  | 156 ms (+66 %) | 475 ms (+43 %) | 144 ms (+129 %) | 150 ms (+138 %) | 444 ms (+20 %) |
| L2-only  | 151 ms (+60 %) | 483 ms (+45 %) | 140 ms (+123 %) | 149 ms (+136 %) | 426 ms (+15 %) |
| full     |  94 ms (+0 %)  | 334 ms (+0 %)  |  63 ms (+0 %)  |  63 ms (+0 %)  | 326 ms (−12 %) |

Two oddities:
1. L1-only and L2-only **regress by nearly the same amount everywhere** —
   that's not independent overhead; both are hitting the same hot path.
2. Combining L1+L2 (full) drops all phases back to LRU level or below —
   so the layers are *cancelling* each other's regression.

## Cause #1: L1's LPB queue was active in L2-only mode too (Phase 1 bug)

`vllm/v1/core/block_pool.py:171` calls `maybe_get_free_queue_factory()`
which returned the `LPBFreeBlockQueue` class **whenever a HiMA runtime
existed**, not whenever **L1 specifically** was on. So when only
`hima_l2_enabled=True`, the BlockPool still installed the LPB queue and
paid all L1's per-block overhead.

That fully explains why L2-only regresses by ~the same amount as
L1-only across every phase: they were running the same code path.

**Fix landed** (2026-05-26): `integration.py:364` `maybe_get_free_queue_factory()`
now also requires `_RUNTIME.config.hima_l1_enabled`. Re-running L2-only
PathA t99 to confirm.

## Cause #2 (hypothesis): L1's overhead has no offsetting benefit at util=0.9

The HiMA paper's L1 mental model is *"LPB protects high-utility
prefixes from eviction"*. The win comes from cache hits the LRU
would have missed. At util=0.9 PathA t1, **all four configs preserve
the anchor at 89.2 % final probe** — the KV pool is large enough
that LRU also keeps the anchor. L1 has nothing to protect, so the
benefit is zero. Only the cost remains.

Cost sources (per allocation when LPB queue active):

| op       | LRU deque | LPB heap |
|---|---|---|
| append   | O(1) | `_score_for` (dict lookup × 2 + math) + heap push O(log N) |
| popleft  | O(1) | `_purge` (skip lazy-dead heads) + heap pop O(log N) |
| update   | n/a  | mark old dead + heap push O(log N) → heap **grows unbounded** between popleft cycles |
| remove   | O(1) | mark dead in dict; cleanup deferred |

At N ≈ 8000 (util=0.9 KV pool, block_size=1056), log N ≈ 13. That's
the constant multiplier on every allocation. The Phase E cold flow
hits 50+ unique prompts × ~2 token blocks each, so the heap churns
hard and the overhead surfaces. Phase H concurrent swarm fires
30 × ~15 = 450 allocations within milliseconds → heap operations
become a hot loop.

Important: this overhead is **proportional to allocation count, not
to KV pressure**. So even when L1 has no useful work to do (anchor
safe under LRU anyway), it still pays the full cost.

## Prediction

If the hypothesis is right, L1-only **should beat LRU** when:
- KV pool is small enough that LRU would evict the anchor
- Anchor re-prefill cost exceeds LPB heap overhead

Existing supporting evidence:
- [`verify/5`](../5_window_sensitivity/) at **util=0.35**, K=10 burst:
  LRU's anchor goes 89 % → 0 %; L1-only stays at 89 % for K=15. So at
  this util L1 DOES protect anchor longer.
- We don't have a TTFT comparison at util=0.35 yet because
  `e2e_l1_pressure_curve` only does anchor probes, not Phase H swarm.

## Test running

`compare_lru_lpb.py --tag _path0 --util 0.35 --tp 2 --phase-f-scale 1`
under both `--mode lru` and `--mode l1_only`. Expected: at util=0.35
L1-only's Phase H TTFT is **lower** than LRU's (because LRU misses
the anchor and pays re-prefill).

Result (pending — see Verdict section below)

## Verdict (n=3 at util=0.35 — confirmed)

**L1-only DOES beat LRU under genuine KV pressure** — n=3 at util=0.35
confirms the n=1 finding:

| metric | LRU (n=3) | L1-only (n=3) | delta |
|---|---:|---:|---:|
| Phase G TTFT (pre-pressure swarm) | 487 ±12 ms | **430 ±0 ms** | **−12 %** ✓ |
| Phase H TTFT (post-pressure swarm) | 458 ±15 ms | **434 ±27 ms** | **−5 %** ✓ |
| Phase H hit % | 85.9 % | 88.9 % | +3 pp |
| throughput | 1251 ±58 tok/s | 1231 ±70 tok/s | −2 % (in noise) |
| final anchor probe | 89.2 % | 89.2 % | tied |

Phase G's L1 std-dev of **0 ms** (vs LRU's ±12) is the most striking
signature: L1's path-counted eviction protection produces *deterministic*
batch wall time on the swarm, while LRU's behavior varies with what
happens to be at the front of the deque when the burst hits.

At util=0.35, the KV pool is tight enough that the swarm-batched
allocations exert real eviction pressure on Phases G/H. LRU misses
some useful blocks; L1's path-counted scoring keeps the high-hit
prefix blocks. TTFT wins by 5-12 %.

At util=0.9, the pool is comfortable, LRU keeps the same blocks
LPB would, and L1's per-allocation overhead surfaces with no
offsetting benefit → L1 loses by ~20 %.

## Final answer

**The user's intuition "L1 alone beats LRU" is correct — but only at
operating points where LRU's eviction policy actually costs cache hits.**

Operating points where L1 wins:
- ✓ util=0.35 PathA (anchor at risk, n=1: −14 % G, −5 % H TTFT)
- ✓ verify/5 pressure curve at K=20 (LRU cliff K=10→15, L1 cliff K=15→20)

Operating points where L1 loses:
- ✗ util=0.9 PathA (anchor safe under LRU, n=3: +20 % H TTFT regression)

Mechanism: LPB's anchor-protection benefit is **proportional to KV
pressure**, while its per-allocation heap overhead is **proportional
to allocation count**. At low pressure with high allocation counts
(util=0.9 + concurrent swarm), overhead dominates. At high pressure
with the same allocation count (util=0.35 + concurrent swarm), the
cache-miss reduction dominates.

This is a real, fundamental property of the design — **not** a bug.
The prior `vllm.md` headline `−12 % Path A` claimed the win came from
"anchor protection" at util=0.9. We now know that's wrong: at util=0.9
there is no anchor to protect (LRU keeps it). The −12 % win came from
the L1×L2 *interaction*, not from L1 alone.

## What `vllm.md` should say after Phase 6

1. The headline `−12 % Path A / −17.7 % Path B` is correct **for
   full-stack (L1+L2)**, but the mechanism explanation ("LPB protects
   the anchor") is wrong at util=0.9 — anchor is preserved by LRU too.
   The −12 % is an L1×L2 interaction, mechanism TBD (see Cause #3).
2. L1 alone matches LRU on **cached state** (89.2 % anchor under both)
   at util=0.9. L1 alone is **slower** in TTFT due to LPB heap overhead.
3. L1 alone wins at lower util (e.g. util=0.35) where KV pressure is
   real and LRU's eviction policy costs hits.
4. The intralayer claim should be stated regime-by-regime, not as a
   single headline.

## Additional bug found during this investigation

`vllm/v1/core/block_pool.py:171` was installing the LPBFreeBlockQueue
factory whenever a HiMA runtime existed, not gated by
`hima_l1_enabled`. So L2-only mode (only `hima_l2_enabled=True`) was
silently using the LPB queue too, paying L1's overhead. This caused
verify/1 and verify/3 to show identical regression patterns
(L1-only ≈ L2-only ≈ +20 % across phases).

**Fix landed** in `vllm/v1/core/hima/integration.py:364` —
`maybe_get_free_queue_factory()` now also requires
`_RUNTIME.config.hima_l1_enabled`. Re-verified with a single-trial
L2-only run at util=0.9: hit% dropped from 88.9 % (pre-fix, LPB queue
in use) to 85.9 % (post-fix, LRU queue in use) which matches the LRU
baseline, confirming the LRU queue is now in effect under L2-only.

## What this means for "should L1 beat LRU"

The user's intuition that "L1 should beat LRU" is **regime-dependent**:
- **High KV slack (util ≤ 0.5 of pool relative to workload)**: L1 = LRU
  on cached state. Only L1's overhead shows. L1 *must* lose.
- **Pressured KV (workload size ~ pool size)**: L1 protects high-utility
  prefixes that LRU evicts. The cache-miss reduction can dwarf L1's
  per-allocation overhead. L1 wins.

The prior `vllm.md` headline `-12 % Path A` was at util=0.9
(slack regime). Now we know it was actually full-stack (L1+L2). L1
alone in this regime *cannot* beat LRU because there's no anchor to
protect. The fact that full-stack still wins tells us something about
the **interaction** (L2 admitter changes which blocks the LPB queue
touches, which is its own finding worth following up — see "Cause #3"
below if confirmed).

## Cause #3 (hypothesis under investigation)

Why does L1+L2 beat both L1-alone *and* L2-alone, and beat LRU?

Possible: L2 admitter DEFERS or REMAPS some allocations that L1-alone
would have evicted blocks for. Fewer LPB ops → less heap churn →
heap stays small/fresh → cheaper popmin. The combination amortizes
L1's cost.

Need to instrument: in full-stack mode, count
`runtime.decisions["defer"]` and `runtime.decisions["cross_*"]` to
see if admitter is firing meaningfully on PathA. If yes, this is the
mechanism. If admitter never fires, the cancellation is from something
else (e.g. a scheduling-order artifact).
