# Finding M.14 — multi-stream makes the regression WORSE, not better

M.13 hypothesized that multi-stream (concurrent) workloads would
amortize the per-batch fixed overhead and turn partial-cache into a
net win for throughput. Test result: **the opposite happens**.

## Setup

New script `14_concurrent_workload.py`: issues batched generations
of 8 concurrent prompts per round, 21 rounds, same Qwen3-8B model,
block_size=1024, max_num_seqs=16. Both TTFT pass (max_tokens=1)
and throughput pass (max_tokens=21) are batched across the 8
concurrent prompts.

## Results

| mode | TTFT total (s) | full total (s) | tok/s | hit% |
|---|---|---|---|---|
| multi-stream baseline | 4.98 | 4.67 | 435.8 | 76.01 |
| multi-stream partial  | 3.97 | 6.64 | 306.8 | 82.27 |

| delta | value |
|---|---|
| TTFT | **-20.35%** (partial wins more on TTFT than single-stream did) |
| full_wall | **+42.05%** (way worse than single-stream's +13.28%) |
| throughput | **-29.60%** (way worse than single-stream's -11.72%) |
| hit% | 76.01% → 82.27% (smaller jump than single-stream because batched requests have less prefix-reuse opportunity) |

## Why concurrent makes it WORSE

The single-stream regression was attributed to "per-step fixed
overhead becomes a larger fraction when GPU work per step shrinks".
The same logic actually **amplifies** in multi-stream:

- **TTFT pass** (max_tokens=1): batch has 8 prompts × 1 query token
  = 8 query tokens. Same per-call cost in both modes. Partial saves
  prefill work proportional to R per prompt, so TTFT wall drops more
  in multi-stream than single-stream. (Better TTFT win.)
- **Full pass** (max_tokens=21): batch has 8 prompts × ~16 query
  tokens (partial) vs 8 × ~500 query tokens (baseline). Partial's
  batch is **14× smaller**. Per-launch fixed overhead is constant
  but useful work-per-launch dropped 14×, so wall regression amplifies.

In other words: M.13's mitigation hypothesis was backwards. Multi-
stream doesn't fix the regression — it makes the regression WORSE,
because partial cache shrinks per-prompt work, which shrinks per-
batch work, which leaves even less to amortize fixed overhead.

## What does help?

After M.13 + M.14, the actionable mitigations narrow down:

1. **Don't apply partial cache when extension is too small.** A
   request whose prefill saving would be < some threshold (e.g.
   100 tokens) shouldn't trigger partial cache. The saving is below
   per-launch overhead. Heuristic + threshold; would need tuning.

2. **Apply partial cache ONLY for the TTFT path.** The TTFT win is
   real in both single- and multi-stream. The decode/throughput
   regression is what we want to avoid. If we could distinguish
   "request whose user cares about first-token-latency" from
   "request that cares about throughput", we'd apply partial only
   to the first.

3. **Combine partial-cache CACHED tokens into the next batch's
   prefill chunk.** Instead of skipping them (which makes the batch
   tiny), include them — pay the prefill compute but maintain batch
   size. The "saving" then comes from KV cache reads being faster
   than full forward, NOT from skipping the forward. This requires
   attention kernel support for "read cached K/V at offsets 0..R but
   also compute Q for those positions and use cached K/V".

(3) is the most architecturally satisfying but requires kernel work.
(1) is the most practical and could be a small heuristic.

## Updated recommendation

The partial-cache mechanism is **not universally net positive**.
Workload analysis matrix:

| workload type | TTFT? | throughput? | net? |
|---|---|---|---|
| single-stream interactive (cc agent) | -17% | -14% | mixed — TTFT win if user-visible |
| multi-stream interactive (cc fleet) | -20% | -30% | bad — throughput collapses |
| TTFT-only benchmark | -17 to -20% | n/a | good |
| Bulk generation | n/a | -14 to -30% | bad |

For deployment: enable for TTFT-dominated interactive workloads
(single user, latency-sensitive). Disable for throughput-dominated
batched workloads. The env var stays opt-in and operators must
decide per workload.

## Validated facts so far

- Bubble elimination is real and 88% effective (M.6/M.7/M.9)
- TTFT win is robust (-16 to -20% across workload shapes)
- Throughput regression is real and scales WITH batch concurrency
- Root cause: partial cache shrinks per-batch GPU work, fixed
  per-launch overhead amortizes worse
- mitigations that DON'T work: VLLM_BATCH_QUEUE_SIZE (M.13),
  multi-stream batching (M.14)
- mitigations un-tested: extension threshold heuristic, kernel-side
  "use cached K/V but still launch Q compute"

## Files

- `dev/interlayer/14_concurrent_workload.py` — multi-stream bench
- `dev/interlayer/runs/m14/{baseline,partial_cache}.{jsonl,out}`
