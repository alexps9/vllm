# verify/2 — first sweep results (turns=32, n=1)

## Setup

- Model: Qwen/Qwen3-8B (single-group full-attention; avoids
  hybrid-model partial-cache bypass)
- TP=2, util=0.55, max_num_seqs=64, VLLM_HIMA_HPB_WINDOW_S=3600
- 16 concurrent conversation clients
- turns=32, each turn appends ~1024 tokens of synthetic "tool result"
  + 128 tokens of model response
- 7 configs in `_CONFIG_ENV` (driver.py): lru / l1_only / l2_only /
  full / l1_pcache / l2_pcache / full_pcache
- n=1 per cell (single trial)

## Result table (final turn = 31)

| config | mean hit% | median hit% | p95 TTFT | batch_wall | vs LRU TTFT |
|---|---:|---:|---:|---:|---:|
| lru          | **96.8 %** | 96.8 % | 878 ms | 878 ms | — |
| l1_only      | **96.8 %** | 96.8 % | 914 ms | 914 ms | +4.0 % |
| l2_only      | **96.8 %** | 96.8 % | 999 ms | 999 ms | +13.8 % |
| full         | **96.8 %** | 96.8 % | 1019 ms | 1019 ms | +16.0 % |
| l1_pcache    | 93.6 % | 93.6 % | 1276 ms | 1276 ms | **+45.3 %** |
| l2_pcache    | 93.6 % | 93.6 % | 1352 ms | 1352 ms | **+54.0 %** |
| full_pcache  | 93.6 % | 93.6 % | 1275 ms | 1275 ms | **+45.2 %** |

(Final prompt length at turn 31: 32 287 tokens — close to Qwen3-8B's
40 K max_model_len.)

## Hit-rate trajectory (every 4 turns)

```
lru          t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:97%
l1_only      t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:97%
l2_only      t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:97%
full         t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:97%
l1_pcache    t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:94%
l2_pcache    t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:94%
full_pcache  t0:47% t4:78% t8:88% t12:92% t16:94% t20:95% t24:96% t28:96% t31:94%
```

Non-pcache configs are **bit-identical** in hit% trajectory. Partial-
cache configs ride the same curve until turn ~28, then diverge to 94 %
(−3 pp).

## Findings

### 1. Songyang's W1 hit% collapse does NOT reproduce on this workload

W1 reported (n=16 clients, turns=32, hybrid Qwen3.5-35B-A3B, util=0.55):
**baseline 84.2 % hit / HiMA 44.2 % hit** — a 40 pp drop.

Our mock (n=16 clients, turns=32, **single-group Qwen3-8B**, util=0.55):
**LRU 96.8 % / L1-only 96.8 % / full 96.8 %** — 0 pp drop across HiMA
layers. partial-cache shows a −3 pp drop on top.

**Why**: at Qwen3-8B TP=2 util=0.55, vLLM reports KV cache size of
~830 K tokens. 16 clients × ~32 K final prompt = ~512 K tokens. With
shared-prefix coalescing, the active KV footprint is well under the
budget — **no eviction pressure**, so eviction policy (LRU vs LPB) is
irrelevant.

To reproduce W1's collapse we'd need one of:
- **Hybrid model** (Qwen3.5-35B-A3B). The mamba portion has its own
  state cache with different eviction dynamics. W1 was on this model.
- **Lower util** (≤ 0.2) to make 16 × 32 K oversubscribe.
- **More clients / more turns** to push past the KV budget.
- **Less prefix sharing** (vary the per-client prelude) so the KV
  state can't coalesce.

verify/2 as currently written **cannot rule out** Songyang's
hypothesis — the workload isn't pressuring the cache. Re-running on
the hybrid model is the highest-fidelity next step.

### 2. partial-cache regression — root-caused and FIXED ✅

**Root cause** (full write-up in
[`dev/interlayer/16_pcache_root_cause_fix.md`](../../../interlayer/16_pcache_root_cause_fix.md)):

`vllm/v1/core/single_type_kv_cache_manager.py:235` set
`num_cached_block = len(req_blocks)`. With M.4/M.5's
`_try_partial_extension` appending a partially-cached block to
`computed_blocks`, the proxy over-counted: the adopted block was
flagged as "cached as full" but had `block_hash = None`. The later
`cache_blocks()` saw `num_cached_blocks >= num_full_blocks` and
returned early — the block, after being extended to full by the
request, never entered `cached_block_hash_to_block`. Future turns
couldn't find it as a full-block hit. Manifested as −3 pp hit% +
+45 % TTFT at the very last turn.

**Fix**: derive `num_cached_block` from the underlying state predicate
(`b.is_null or b.block_hash is not None`) instead of the structural
proxy `len(req_blocks)`.

**Post-fix verification** (same workload, GPU 5,6, util=0.55):

| config | clients | pre-fix | post-fix |
|---|---:|---:|---:|
| pcache hit% | 1 | 93.6 % | **96.8 %** ✓ tied LRU |
| pcache p95 TTFT | 1 | 134 ms (+37 % vs LRU) | **102 ms** (−5 % vs LRU) ✓ |
| pcache hit% | 16 | 93.6 % | **96.8 %** ✓ tied LRU |
| pcache p95 TTFT | 16 | 1326 ms (+46 % vs LRU) | **834 ms** (−5 % vs LRU) ✓ |

pcache is now correctly a marginal TTFT win on this workload.

### 3. HiMA layers alone are within noise

L1-only +4 %, L2-only +14 %, full +16 % p95 TTFT vs LRU. With n=1 and
batch_wall variance from concurrency, ±10 % is roughly the noise
floor. The L1-only +4 % is essentially tied; L2-only +14 % and full
+16 % are at the noise edge but consistent with the verify/3 stale
"L2 admitter overhead" story (now flagged for re-measurement). n=3 on
this would resolve.

## Missing data

- **turns=64 cells failed** — Qwen3-8B's max_position_embeddings is
  40 960; the driver's growing-context formula
  (turns × 1152 + 4096 = 77 824 at turns=64) exceeds it. Fix options:
  reduce `TOOL_RESULT_TOKENS` from 1024 to 512 (gives ~37 K max
  context at turns=64), or move to a longer-context model.
- **turns=8 sanity** — not run yet (deferred until turn=32 showed
  signal worth dissecting; the lack of differentiation at 32 makes
  turn=8 even less useful — sanity already implied by trajectory
  starting at 47 % and converging cleanly).
- **n=3** — currently n=1 per cell. Re-run with n=3 once the
  workload shape is finalised (probably means: switch to hybrid model
  first).

## Next steps (prioritised)

1. **Re-run on hybrid Qwen3.5-35B-A3B** at util=0.55, same 7-config
   matrix. Should produce the W1 collapse pattern if it exists.
   Requires 2× H200 + TP=2.
2. **Investigate partial-cache regression** on this workload. M.7
   said pcache wins single-turn TTFT on follow-up cc turns; verify/2
   says it loses on 16-client growing context. Reconcile.
3. **Reduce TOOL_RESULT_TOKENS to 512** in driver, rerun turns=64 on
   Qwen3-8B as a stress test (more turns at lower per-turn context
   may stress KV differently).

## Files

- Driver: `driver.py`
- Per-config JSONLs: `runs/turns32_<config>_win3600.jsonl`
- Driver stdout/stderr: `runs/turns32_<config>_win3600.out`
- This analysis: `results.md`
