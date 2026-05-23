# dev/interlayer — vLLM-side bubble elimination

The vLLM analog of HiMA's L2 (inter-pool / cross-pool layer). vLLM
has ONE inflated KV pool (`block_size = 1056` on Qwen3.5-35B-A3B
hybrid), not two pools like sglang. Sglang's "move pages between
pools via cuMemUnmap+cuMemMap" doesn't translate. But vLLM still has
a real measurable bubble — caused by `block_size` inflation forcing
each request's last partial block to be abandoned — and this
directory documents the design, prototype, and end-to-end
validation of the mechanism that eliminates it.

## Status: PROTOTYPE VALIDATED end-to-end (non-hybrid)

The partial-block-caching mechanism works. Single-group models
(no mamba) get **bubble elimination + 43% TTFT win on follow-up
turns** at the size regime that matters (block_size = 1024,
prompts ≈ 8K tokens). See Findings M.6 and M.7 below.

## Findings (chronological)

| # | What | Status | File |
|---|---|---|---|
| M.1 | Bubble exists, exactly as predicted at block_size=1056 | done | `02_partial_cache_micro.py` + `runs/02_partial_cache_micro.jsonl` |
| M.2 | Per-group num_computed_tokens is the architectural blocker for hybrid | doc | `03_per_group_hit_length.md` |
| M.3 | TTFT win quantification: 15-50% on follow-up turns | doc | `04_savings_quantified.md` |
| M.4 | Cache-side scaffolding (BlockPool.cache_partial_block, populated by FullAttentionManager.cache_blocks) | code | commit `a80ef6d1e` |
| M.5 | Hit-side prototype (KVCacheManager._try_partial_extension) for single-group configs | code | commit `a2533c8fc` + `05_hit_side_impl.md` |
| M.6 | End-to-end validation on Qwen3-8B at block_size=256 — bubble eliminated, TTFT -24 to -45% | **validated** | `06_validation_results.md` + `runs/05_nonhybrid_{baseline,partial_cache}.{jsonl,out}` |
| M.7 | Scaled validation at block_size=1024 — **43% TTFT win at R=800** | **validated** | `07_large_results.md` + `runs/07_nonhybrid_large_{baseline,partial_cache}.{jsonl,out}` |

## Headline result (M.7)

Qwen3-8B, block_size=1024, K=8 full blocks, single H200:

```
R     base_unc base_ttft   pcache_unc pcache_ttft   TTFT delta
    0      16     18.7         16        18.8        +0.3%
   64      80     19.2         16        19.1        -0.1%
  256     272     22.8         16        19.8       -13.1%
  512     528     29.4         16        20.2       -31.2%
  800     816     35.2         16        19.8       -43.7%
 1000    1016     40.1         16        24.4       -39.1%
```

Bubble eliminated on every R > 0 (uncached drops from R+16 to 16,
i.e. -94 to -98%). Partial-cache TTFT is approximately flat ~20 ms
across R — decoupled from bubble size. Default behavior is unchanged
when `VLLM_PARTIAL_CACHE_ENABLED` is unset.

## Activation

```bash
VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u <your_script>.py
```

Both M.4 cache-side and M.5 hit-side are gated by this single env
var. Disabled by default. Only takes effect for single-group full-
attention models; hybrid models bypass (would otherwise corrupt
mamba state).

## Code locations

```
vllm/v1/core/block_pool.py
  + cached_partial_block_map: dict[partial_key, KVCacheBlock]
  + cache_partial_block() — called by FullAttentionManager.cache_blocks
  + get_cached_partial_block() — called by KVCacheManager
  + _compute_partial_key() — content-addressable key builder
  + counters + info log on first 16 insertions

vllm/v1/core/single_type_kv_cache_manager.py
  + FullAttentionManager.cache_blocks override — populates partial cache

vllm/v1/core/kv_cache_manager.py
  + KVCacheManager._try_partial_extension() — the hit-side lookup
  + Wiring in get_computed_blocks() with single-group gate
```

## Repro

```bash
# Small (M.6, block_size=256, K=2, prompts ~528-783 tokens):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py | tee dev/interlayer/runs/05_nonhybrid_baseline.out
cp dev/interlayer/runs/05_nonhybrid_micro.jsonl dev/interlayer/runs/05_nonhybrid_baseline.jsonl
CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py | tee dev/interlayer/runs/05_nonhybrid_partial_cache.out
cp dev/interlayer/runs/05_nonhybrid_micro.jsonl dev/interlayer/runs/05_nonhybrid_partial_cache.jsonl

# Large (M.7, block_size=1024, K=8, prompts ~8208-9231 tokens):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/07_nonhybrid_microbench_large.py | tee dev/interlayer/runs/07_nonhybrid_large_baseline.out
cp dev/interlayer/runs/07_nonhybrid_large.jsonl dev/interlayer/runs/07_nonhybrid_large_baseline.jsonl
CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u dev/interlayer/07_nonhybrid_microbench_large.py | tee dev/interlayer/runs/07_nonhybrid_large_partial_cache.out
cp dev/interlayer/runs/07_nonhybrid_large.jsonl dev/interlayer/runs/07_nonhybrid_large_partial_cache.jsonl
```

## What's NOT yet done

1. **Hybrid models (Qwen3.5-35B-A3B etc.)** — gated by single-group
   check; would need per-group `num_computed_tokens` lift (M.2,
   ~400 LOC, separate PR). Once landed, hybrid gets the same
   bubble elimination, and Finding D's 42.6% workload waste becomes
   recoverable.

2. **Eviction integration** — `cached_partial_block_map` grows
   monotonically; `BlockPool.evict_blocks` doesn't yet remove
   partial-cache entries. Will leak on long-running engines.

3. **Cross-request correctness at scale** — multi-tenant workloads
   where many requests share partial-cache entries simultaneously
   not yet stress-tested.

4. **Adding `VLLM_PARTIAL_CACHE_ENABLED` to vLLM's env registry**
   (`vllm/envs.py`) — the "Unknown vLLM environment variable" warning
   at runtime is cosmetic noise.

5. **End-to-end cc workload measurement** — the partial-cache fix
   should be slotted into `dev/compare_lru_lpb.py`'s LRU/LPB
   comparison pattern to produce a Finding M.8 with TTFT / TPOT /
   throughput numbers on real cc traffic.

## File map

```
dev/interlayer/
├── README.md                          # this file
├── 01_design_space.md                 # 5 candidates surveyed, A picked
├── 02_partial_cache_micro.py          # hybrid bubble baseline (Qwen3.5-35B-A3B)
├── 03_per_group_hit_length.md         # M.2 architectural blocker for hybrid
├── 04_savings_quantified.md           # M.3 TTFT win extrapolation
├── 05_nonhybrid_microbench.py         # M.6 single-group testbed (block_size=256)
├── 05_hit_side_impl.md                # M.5 hit-side design notes
├── 06_validation_results.md           # M.6 first end-to-end validation
├── 07_nonhybrid_microbench_large.py   # M.7 scaled testbed (block_size=1024)
├── 07_large_results.md                # M.7 scaled validation results
├── SESSION_HANDOFF.md                 # earlier session handoff (now superseded)
└── runs/
    ├── 02_partial_cache_micro.{jsonl,out}                — M.1 hybrid baseline
    ├── 02_partial_cache_micro_{cache_side_v2,with_cache_side}.out
    ├── 05_nonhybrid_baseline.{jsonl,out}                 — M.6 small baseline
    ├── 05_nonhybrid_partial_cache.{jsonl,out}            — M.6 small + fix
    ├── 05_nonhybrid_micro.jsonl                          — last 05_ run scratch
    ├── 07_nonhybrid_large_baseline.{jsonl,out}           — M.7 large baseline
    ├── 07_nonhybrid_large_partial_cache.{jsonl,out}      — M.7 large + fix
    └── 07_nonhybrid_large.jsonl                          — last 07_ run scratch
```
