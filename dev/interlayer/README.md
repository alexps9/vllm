# dev/interlayer — cross-engine inter-pool work

This directory holds inter-pool / cross-pool experiments from both
engines. Each engine has structurally different "inter-pool" work
because their architectures differ:

| sub-dir | engine | what it is |
|---|---|---|
| (this README + M.1–M.16 content below) | **vLLM** | bubble elimination via partial-block caching. vLLM has ONE inflated KV pool (`block_size = 1056` on hybrid models), not two pools. Sglang's "move pages between pools via cuMemUnmap+cuMemMap" doesn't translate; vLLM's bubble is in the per-request abandoned last-partial-block. |
| [`planner_validate/`](planner_validate) | **sglang** | output runs for the sglang interlayer integration tests: (1) v1 slack-harvest planner correctness on the legacy actuator, (2) v1 path-B logical actuator end-to-end smoke (`r1_v1logical/`). Drivers in sglang repo at `dev/interlayer/planner_validate/`. |

---

## vLLM-side bubble elimination

vLLM has ONE inflated KV pool (`block_size = 1056` on Qwen3.5-35B-A3B
hybrid), not two pools like sglang. Sglang's "move pages between
pools via cuMemUnmap+cuMemMap" doesn't translate. But vLLM still has
a real measurable bubble — caused by `block_size` inflation forcing
each request's last partial block to be abandoned — and this
directory documents the design, prototype, and end-to-end
validation of the mechanism that eliminates it.

## Status: SHIPPED — pcache regression root-caused + fixed

The partial-block-caching mechanism works. Single-group models
(no mamba) get **bubble elimination + 43% TTFT win on follow-up
turns** at the size regime that matters (block_size = 1024,
prompts ≈ 8K tokens). See Findings M.6 and M.7 for the win, and
M.10–M.16 for the regression hunt that closed in a one-line
``num_cached_block`` fix.

## Findings (chronological)

The findings split into two arcs:

**Arc 1 (M.1–M.9): design + validation.** Quantified the bubble,
shipped cache-side + hit-side plumbing, validated end-to-end on
single-group models, measured the real-workload TTFT/throughput
trade-off.

**Arc 2 (M.10–M.16): post-shipping root-cause chain.** M.9 surfaced
a -12 % throughput regression on the real cc workload. M.10–M.15
walked through hypotheses (in-place mutation, dispatch
fragmentation, multi-stream, threshold mitigation). M.16 finally
root-caused the regression to ``num_cached_block`` over-counting
adopted partial blocks, and fixed it.

| # | What | Status | File |
|---|---|---|---|
| M.1 | Bubble exists, exactly as predicted at block_size=1056 | done | `02_partial_cache_micro.py` + `runs/02_partial_cache_micro.jsonl` |
| M.2 | Per-group num_computed_tokens is the architectural blocker for hybrid | doc | `03_per_group_hit_length.md` |
| M.3 | TTFT win quantification: 15-50% on follow-up turns | doc | `04_savings_quantified.md` |
| M.4 | Cache-side scaffolding (BlockPool.cache_partial_block, populated by FullAttentionManager.cache_blocks) | code | commit `a80ef6d1e` |
| M.5 | Hit-side prototype (KVCacheManager._try_partial_extension) for single-group configs | code | commit `a2533c8fc` + `05_hit_side_impl.md` |
| M.6 | End-to-end validation on Qwen3-8B at block_size=256 — bubble eliminated, TTFT -24 to -45% | **validated** | `06_validation_results.md` + `runs/05_nonhybrid_{baseline,partial_cache}.{jsonl,out}` |
| M.7 | Scaled validation at block_size=1024 — **43% TTFT win at R=800** | **validated** | `07_large_results.md` + `runs/07_nonhybrid_large_{baseline,partial_cache}.{jsonl,out}` |
| M.8 | Hybrid (mamba) needs sub-block SSM state cache — beyond per-group plumbing | doc | `08_hybrid_architectural_blocker.md` |
| M.9 | Real cc workload (10 sessions, 106 turns): **TTFT -16%, bubble -88%, throughput -12%** | **measured** | `09_cc_workload_results.md` + `runs/09_cc_{baseline,partial_cache}_v5.jsonl` |
| M.10 | M.9 throughput regression first root-cause hypothesis (in-place mutation) — partial fix only | doc | `10_root_cause.md` |
| M.11 | nsys profiling of the post-M.10 residual regression — GPU-side time matches; CPU-side overhead | doc | `11_nsys_findings.md` + `runs/nsys/` |
| M.12 | py-spy localizes regression to GPU sync waits → batch fragmentation increases sync points | doc | `12_pyspy_root_cause.md` |
| M.13 | `batch_queue_size` mitigation doesn't help; dispatch fragmentation is the root cause | doc | `13_mitigation_attempts.md` + `runs/m13/` |
| M.14 | Multi-stream (concurrent) makes the regression WORSE, not better | doc | `14_multistream_results.md` + `14_concurrent_workload.py` + `runs/m14/` |
| M.15 | `VLLM_PARTIAL_CACHE_MIN_R=256` threshold heuristic recovers 24% of regression while keeping TTFT win | **shipped** | `15_threshold_mitigation.md` + `runs/m15/` |
| M.16 | Adopted partial block was never re-cached as full (num_cached_block over-count) — root cause + one-line fix | **fixed** | `16_pcache_root_cause_fix.md` + `pcache_fix_num_cached_block.patch` + verify/2 runs |

## Headline result (M.7 micro / M.9 real workload)

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

### M.9 real cc workload (Qwen3-8B, 10 sessions, 106 turns):

```
metric                  baseline   partial    delta
mean TTFT/turn          53.8 ms    45.3 ms    -15.84%   ← interactive win
hit % vs prior content  91.59%     98.99%     +8.09%    ← near-perfect
bubble tokens           51442      6147       -88.05%   ← bubble killed
mean full_wall/turn     134 ms     152 ms     +13.99%   ← decode tradeoff
throughput              157 tok/s  137.7 tok/s -12.27%
```

**Verdict**: net win for TTFT-sensitive interactive workloads
(agents, chat). Net loss for bulk-generation throughput. Opt-in via
env var so operators choose per deployment.

### Post-M.9 root-cause chain (M.10–M.16)

M.9 left two open questions: (a) why the -12 % throughput
regression at block_size=1056, and (b) whether the partial-cache
hit was actually being re-cached as a full block on subsequent
turns. The M.10–M.16 chain answered both.

| arc step | conclusion |
|---|---|
| M.10 (mutation hypothesis) | In-place mutation of the cached `block_hash` was *one* bug; fixing it recovered some throughput but the regression persisted. |
| M.11 (nsys) | GPU-side time matches between baseline and pcache → regression is CPU-side overhead, not extra GPU work. |
| M.12 (py-spy) | Localized regression to GPU sync waits in the engine main loop. Hypothesis: batch fragmentation → more sync points per token. |
| M.13 (bq mitigation) | `batch_queue_size` increase didn't close the gap; dispatch fragmentation is the real cause. |
| M.14 (multi-stream) | Multi-stream concurrency made the regression *worse* — confirmed dispatch contention. |
| M.15 (threshold) | `VLLM_PARTIAL_CACHE_MIN_R=256` heuristic cuts the throughput regression by 24 % while preserving the TTFT win. **Shipped.** |
| M.16 (verify/2 collision) | verify/2's growing-context workload exposed a different regression: adopted partial block was never re-cached as full because `num_cached_block` over-counted (`len(req_blocks)` proxy). One-line fix in `single_type_kv_cache_manager.py` recovers hit % from 93.6 → 96.8 and turns TTFT into a -5 % win vs LRU. **Fixed.** |

Net post-fix posture: TTFT win preserved on both M.7/M.9 and on the
multi-turn growing-context workload (verify/2). Throughput
regression mitigated (M.15 threshold). Hybrid path still gated on
M.2.

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
  + dedups on partial_len; skips during decode (M.9 throughput fix)

vllm/v1/core/kv_cache_manager.py
  + KVCacheManager._try_partial_extension() — the hit-side lookup
  + Wiring in get_computed_blocks() with single-group gate
  + Uses block_pool.get_partial_extensions_for() to avoid O(R) probe
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
├── 08_hybrid_architectural_blocker.md # M.8 hybrid needs sub-block SSM cache
├── 09_cc_workload_compare.py          # M.9 real cc workload bench
├── 09_cc_workload_results.md          # M.9 real-workload numbers
├── 10_root_cause.md                   # M.10 in-place mutation hypothesis (partial fix)
├── 11_nsys_findings.md                # M.11 nsys → regression is CPU-side
├── 12_pyspy_root_cause.md             # M.12 py-spy → batch fragmentation
├── 13_mitigation_attempts.md          # M.13 batch_queue_size ruled out
├── 14_concurrent_workload.py          # M.14 multi-stream driver
├── 14_multistream_results.md          # M.14 multi-stream regression numbers
├── 15_threshold_mitigation.md         # M.15 VLLM_PARTIAL_CACHE_MIN_R=256 ships
├── 16_pcache_root_cause_fix.md        # M.16 num_cached_block fix (final)
├── pcache_fix_num_cached_block.patch  # M.16 patch snapshot
├── cc_long_traces.jsonl               # M.9 dataset (10 sessions × 106 turns)
├── SESSION_HANDOFF.md                 # earlier session handoff (now superseded)
└── runs/
    ├── 02_partial_cache_micro.{jsonl,out}                — M.1 hybrid baseline
    ├── 02_partial_cache_micro_{cache_side_v2,with_cache_side}.out
    ├── 05_nonhybrid_baseline.{jsonl,out}                 — M.6 small baseline
    ├── 05_nonhybrid_partial_cache.{jsonl,out}            — M.6 small + fix
    ├── 05_nonhybrid_micro.jsonl                          — last 05_ run scratch
    ├── 07_nonhybrid_large_baseline.{jsonl,out}           — M.7 large baseline
    ├── 07_nonhybrid_large_partial_cache.{jsonl,out}      — M.7 large + fix
    ├── 07_nonhybrid_large.jsonl                          — last 07_ run scratch
    ├── 09_cc_*.{jsonl,out}                               — M.9 + M.10 diagnostic variants
    ├── m11_*.out                                         — M.11 nsys captures
    ├── nsys/                                             — M.11 nsys profile traces
    ├── m13/                                              — M.13 batch_queue sweep
    ├── m14/                                              — M.14 multi-stream sweep
    └── m15/                                              — M.15 threshold sweep
```
