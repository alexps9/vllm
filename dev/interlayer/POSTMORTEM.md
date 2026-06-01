# interlayer — POSTMORTEM (CLOSED, not pursued)

**Verdict: the vLLM "interlayer" effort is closed. We will not build it.**
Investigation — including live measurement on Qwen3.5-35B-A3B — showed the
problem it targets is small/mislabeled, the painful part is architecturally
unfixable by this approach on hybrid, and the genuinely valuable lever
(cost-aware caching) either already exists (L1) or was tried and failed (L2).
The cost-model research has value on **sglang** (its separate-pool architecture
creates a real decision) but **not on vLLM** (its unified pool dissolves it).

`0_feasibility/` is kept as the measurement record behind this. The
forward-looking docs (`design.md`, `PLAN.md`, `1_allocator/`) and the orphaned
prototype (`vllm/v1/core/interlayer/`, `tests/v1/core/interlayer/`) were
removed; recoverable from git history (closed at commit noted in the closing
commit message).

---

## What we set out to do
Hybrid models force a uniform KV page = `max(attention, mamba)` = **1056 tokens**
on this model (mamba's fp32 SSM state is the large one). Premise: attention
allocates whole 1056-token pages but fills them fractionally → a big "page-size
bubble"; fix = a two-level sub-block allocator (attention packs 32-token
sub-blocks into 1056 pages, mamba keeps whole pages), reclaiming pages by a cost
model. The feasibility gate (`0_feasibility/`) passed every technical check
(kernel tolerates scattered sub-blocks under CUDA graphs; allocator memory-safe;
"cheapest page to vacate" decision cheap). **Feasibility was never the problem —
value was.**

## What we found (with evidence)

1. **The "42.6% bubble" is mislabeled — it is a RECOMPUTE ratio, not memory.**
   `0_feasibility/page_bubble/counterfactual_block_size.py:126-136` computes
   `Σ(L mod 1056) / Σ(new content per turn)` = the partial-block tail that is
   recomputed each turn ÷ new content. That is a compute/cache-miss quantity.

2. **The true memory bubble (attention internal fragmentation), live-measured**
   (`live_bubble_snapshot.py`, 43 resident real-agent contexts): **~5% at
   short/mid resident lengths (p50≈5k), ~0.5% at deep 100k contexts.** Small,
   and the slice the sub-block allocator could actually reclaim. (Mamba pages
   are not fragmented — the fp32 state fills the page.)

3. **The recompute tail is real but unfixable by this approach on hybrid.**
   `ttft_tail_vs_context.py`: recomputing the ~528-token tail costs **~30 ms/turn,
   flat across 16k–95k context**. And `0_feasibility/page_bubble/08_hybrid_
   architectural_blocker.md`: on hybrid you **cannot skip** the tail by finer
   *attention* caching — a mamba layer needs its SSM state at every position,
   the state is cached only at 1056 boundaries, and the circular dependency
   (mamba's skip needs attention's tail hidden states) forces full recompute.
   The binding constraint is mamba's fp32-state-forced 1056 granularity, which
   the attention sub-block fix does not touch.

4. **The cross-type (KV vs mamba) storage decision collapses to L1.** For a
   prefix-cache hit you need **both** the attention KV and the mamba snapshot —
   they are co-dependent (reconstructing either missing piece forces a near-full
   forward over the prefix, which regenerates the other; keeping one alone saves
   only its projection ~few %, at terrible value-per-byte). So there is no
   distinct "KV vs mamba" optimization — it is whole-prefix-bundle eviction =
   cost-aware eviction = **L1 (already done, already wins)**.

5. **The real gap vs sglang is prefix-cache GRANULARITY (confirmed in both
   codebases).** vLLM caches at **fixed 1056 boundaries**; the partial tail is
   never hashed (`kv_cache_utils.py:646`, `single_type_kv_cache_manager.py:299`,
   `:507`), so it cannot reuse the exact turn-end prefix — it rounds down and
   recomputes the tail. sglang's **radix tree** (`page_size=1` is *enforced* for
   mamba models, `mamba_radix_cache.py:540-542`) caches the **exact** turn-end
   prefix and snapshots mamba state at tree nodes (with a `K_BIG` density knob
   trading snapshot memory for re-scan). So sglang *can* reuse the exact
   turn-end; vLLM structurally cannot.

6. **Fixing #5 = building radix / fine-grained caching into vLLM core** — a
   from-scratch re-architecture of vLLM's block-hash KV cache. It is **outside
   the HiMA cost-model framework** (that framework is structure-agnostic
   eviction layered on top of each engine's existing cache) and **not symmetric
   with the sglang work** (on sglang the radix cache pre-existed; we only added
   LPB on top). It is an engineering port of a published technique — low
   research novelty.

## The architectural asymmetry (the core insight)

> **A cost model pays off only where the architecture leaves a hard decision.**

- **sglang** — separate attention/mamba pools + fixed split → a genuine
  "when/what to rebalance between pools" decision → a cost model has value.
  Campaign result (`dev/intralayer/sglang.md`): LPB vs baseline, **mean TTFT
  −16.2%, cache hit +68.7%** on skewed-popularity stress.
- **vLLM** — unified pool → that rebalancing decision is **dissolved by
  construction** (any page is either type; the KV↔mamba split is emergent from
  what is cached). The equivalent cost model — **L2 (admitter / budgeter /
  cross-pool planner)** — was built, measured **neutral vs LRU**, and removed
  (`dev/archive/L2/README.md`). vLLM L1 (eviction) still wins
  (Path A/B −10.7%/−12.2%).

So the cost-model lever is **architecture-dependent**: real on sglang's
separate-pool design, absent on vLLM's unified pool.

## Decisions

- **vLLM interlayer (page-size / two-level sub-block allocator): NOT pursued.**
- **Do NOT frame any deliverable as "sglang wastes less than vLLM."** The two
  engines' waste is *different kinds* (vLLM: internal frag + recompute tail;
  sglang: fixed-split misallocation) — not summable or directly comparable —
  and a cross-engine end-to-end comparison confounds kernels/schedulers/configs
  (the +26.6% L2 phantom is the precedent). Instead prove:
  1. **within-engine ablations** (our cost-model vs that engine's baseline) —
     clean, no cross-engine confound; already have sglang −16% TTFT, vLLM L1
     Path A/B;
  2. **the architectural-asymmetry insight** (why the cost model pays on sglang
     and not on vLLM) — this post-mortem is its draft.

## Research-value judgment (recorded)

- vLLM fine-grained/radix caching → engineering port, **low novelty**.
- The transferable, validated contribution → **cost-aware hybrid eviction (L1)**,
  using `c_KV(L)=αL²+βL+γ` (attention O(L²)) vs `c_M(L)=αL+β` (mamba O(L)).
- Narrow open angle → a **principled mamba-snapshot-density cost model**
  (where/how-dense to snapshot to optimize re-scan × hit-prob × memory), beyond
  sglang's crude `K_BIG`.
- Position/analysis angle → **"the unified pool dissolves the cross-pool
  rebalancing problem"** — a systems insight, this doc is the seed.

## Carried, NON-interlayer item
- **`#99` — production `LPBPriorityQueue` (L1) lazy-delete heap**: unbounded
  bloat + O(P) peek spikes under update-heavy load (found via the
  `decision_cost` IndexedHeap audit). This is a real **L1** latent issue,
  independent of interlayer — kept open.
