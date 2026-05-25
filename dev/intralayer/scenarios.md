# LPB vs LRU — shared benchmark scenarios

Engine-agnostic. Both vLLM HiMA L1 and sglang LPB should be able to
run these phases on a comparable model + workload; per-engine
specifics (how to launch, what knob enables LPB, what model/util) go
in [`vllm.md`](vllm.md) and [`sglang.md`](sglang.md). Per-engine
results tables go in the same files.

## Goal

A 1-engine-load pipeline that puts both LPB and LRU through
back-to-back regimes designed to:

1. **Establish a high-value cached prefix** ("anchor") that LPB
   should protect and LRU shouldn't have any special reason to.
2. **Apply realistic workload pressure** that, under LRU, evicts
   the anchor.
3. **Issue a workload that re-uses the anchor**, so the protection
   converts into measurable TTFT savings under LPB but not under
   LRU.
4. **Probe / measure** the anchor state to confirm the mechanism
   actually fired.

The trap that we kept falling into: if (3) is measured before (2)
has had a chance to evict the anchor under LRU, LRU and LPB both
have it cached → no measurable delta → false negative. The
ordering below avoids that.

## Phase pipeline (`A → B → G → E → F → H → C`)

| phase | role | workload | what it tests |
|---|---|---|---|
| **A** | warm-up | 500 serial probes of the same long prefix ("anchor", e.g. ~5 KB tokens) | seeds the anchor's hit-count high enough that LPB scores it well above any cc-burst block |
| **B** | average-case workload | replay N real cc-traffic sessions serially | normal multi-turn agent traffic; serves as the "no regression on average workloads" check |
| **G** | pre-pressure swarm (control) | submit M anchored requests (anchor + unique 16-token tail) **as one batched `generate()` call** | the production swarm pattern fired *before* the cache has been churned. At realistic util the anchor often survives B, so both modes hit → control: confirms G alone is not enough to expose the win |
| **E** | cold hot-path test | 50 unique 2 K-token prompts with truly random per-trial-seeded tokens | LPB's hot-path overhead (heap / scoring) vs LRU's deque — checks for regression on no-shared-prefix workloads |
| **F** | adversarial decoy | N decoys × ~30 K tokens each warmed K× then a cold-flow; total footprint scaled so the cache is genuinely under pressure | does LPB protect useless-but-hot blocks at LRU's expense? Tests for the "past hit count does not predict future utility" failure mode |
| **H** | **post-pressure swarm (headline)** | same M-batch as G, fired *after* E + F have churned the cache enough to evict the anchor under LRU | LPB still has the anchor (Phase A score keeps it), LRU lost it. Swarm reveals the difference. **This is where the workload-metric LPB win lands.** |
| **C** | final probe (diagnostic) | one serial probe of the anchor | binary "is the anchor still cached" check |

### What each metric means in each phase

- **B / D-equivalent / E / F (serial phases)**: TTFT = mean first-
  token wall over the issued requests; TPOT = mean per-decode-token
  wall; throughput = sum output tokens / sum decode wall.
- **G / H (batched phases)**: TTFT = **batch wall** for the entire
  N-request batch's first-token pass. This is the worst-case wait
  in the swarm — under LRU it's dominated by the one request that
  pays the anchor prefill (others share via the engine's prefix-
  cache merge); under LPB no one pays it. **Token throughput is
  not a meaningful comparison in G/H** because batched-decode wall
  is already dominated by the parallel decode and the saved
  prefill is amortised; report **request throughput** =
  `N_requests / total_batch_wall` instead, or just report total
  batch wall directly.
- **C (probe)**: binary `cached_tokens / anchor_len`. Expected:
  LRU 0 / anchor_len, LPB ~88-89 % / anchor_len.

### Statistical replication

N=3 independent engine loads per (mode, sweep). Aggregate
mean ± sample stddev. Use a deterministic per-trial seed for any
random content (Phase E filler).

## Sweeps

A single sweep = one (model, op-point) configuration. Run both
LRU and LPB at that sweep, n=3 trials each. Document the KV budget
and workload-to-budget ratio.

| sweep | rationale |
|---|---|
| **Path A** — production op-point, smaller model | util=0.9, base model. Most-realistic configuration. Phase H should land the LPB win here. |
| **Path B** — production op-point, larger model | util=0.9, bigger model with tighter KV-per-token. Phase H win should scale with model FLOPs (bigger anchor-prefill cost saved). |

Earlier we also ran a **Path-0** (util=0.35, artificially constrained
KV) which made Phase B itself evict the anchor, surfacing the LPB
win via Phase G alone. After adding Phase H, Path-0 became
redundant — Phase H exposes the same win at production util — so
it was retired. Documented for historical reference only.

## Expected outcomes

- **Anchor protection (Phase C)**: binary, stddev = 0 across trials
  and sweeps. LRU 0 / anchor_len, LPB ~88-89 % / anchor_len.
- **Phase H batch TTFT**: LPB faster by ~one-anchor-prefill cost
  amortised across the batch wall. On a 35 B-class model with
  ~5 K-token anchor, ~40-50 ms saving per swarm (~−12 %); on a
  120 B-class model ~100 ms saving (~−18 %).
- **All other phases (B/G/E/F)**: tied within trial noise (~±2 %).
  No regression on standard workload metrics.
- **Phase H TPOT (batched-mode formula artifact)**: the
  `(full_wall − ttft_wall) / N_TPOT_TOKENS` formula isn't
  comparable across modes in batched submission — drop from
  plots or footnote.
- **Phase F worst case (if scale=10)**: tied within noise. Need
  scale ≥ 20 (decoys ≥ 100 % of KV budget) to expose LPB's
  "protect useless hot blocks" failure mode. Still open.

## Pitfalls we found

- **The Phase D ordering bug**: original `A → B → C → D → E` had
  the Phase C probe re-warming LRU's anchor before D's serial
  rehit could measure the cold state. Solution: drop the
  intervening probe, measure cold state from the *first* rehit
  request — but that introduces its own subtlety (rehit[0]
  warms the cache for rehits 1..29, diluting the average). Cleaner:
  move the swarm to batched submission (Phase G/H), where the
  whole batch sees the same cache state.
- **`util=0.9` makes Phase B inadequate as the eviction event**:
  KV budget grows 5-8× vs `util=0.35`, and cc-burst's 6.6 M total
  prompt tokens no longer saturates it. The anchor survives B
  in both modes. Solution: Phase E + F's combined ~2.5 M tokens
  *do* create real pressure; Phase H runs after that.
- **`gpu_memory_utilization=0.05`** "to make KV scarce" is not
  the right knob — production runs at 0.85-0.95. Build pressure
  by scaling the workload instead.
- **`grep -c '"kind":"…"'`** misses rows when the key has a
  trailing space (JSON-pretty format). Always parse JSON
  properly when aggregating.
- **vLLM prefix-cache merges concurrent identical-prefix
  prefills within a batch**. So under LRU in Phase G/H, only **1**
  of the 30 requests actually pays the anchor prefill — the
  other 29 share. The LPB saving is exactly that 1 prefill cost.
  Don't expect 30× speedup; expect 1×-amortised-over-batch.

## Where this is implemented

- **vLLM**: `dev/compare_lru_lpb.py` (driver) + `dev/plot_lru_vs_lpb.py`
  (aggregator). See [`vllm.md`](vllm.md).
- **sglang**: the gate exists (`SGLANG_LPB_LRU=1`) and the
  selector is in `python/sglang/srt/mem_cache/mamba_radix_cache.py`,
  but a Phase-A-through-H driver script equivalent to
  `compare_lru_lpb.py` has not yet been written. See
  [`sglang.md`](sglang.md).
