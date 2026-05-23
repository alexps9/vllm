# dev/interlayer — session handoff (May 2026)

## Goal at session start

> "vLLM 是单个池子不是多池子,然而 vLLM 的浪费也是存在的,是存在 bubble 的...
> 实现消除 bubble 并且获得一些场景下 throughput, TTFT, TPOT 这些东西的性能提升"

Translation: design and ship a vLLM-side analog of HiMA's L2 (sister
sglang project's inter-pool actuator) that eliminates the
partial-block bubble and delivers measurable TTFT/throughput/TPOT
improvements.

## What this session delivered

**Design**: 5 candidates surveyed, 1 picked (partial-block cache with
COW reuse) — `01_design_space.md`. The conceptual gap between sglang
(multi-pool cuMemUnmap+cuMemMap) and vLLM (single inflated pool)
mapped to the right vLLM-side intervention. Architectural blocker
identified (per-group `num_computed_tokens` for hybrid models) and
quantified (~400 LOC PR).

**Measurement**: bubble empirically confirmed on Qwen3.5-35B-A3B
hybrid at block_size=1056 — `02_partial_cache_micro.py` /
Finding M.1. `turn2_cached = K * block_size` exactly for every R > 0;
the partial-block tail is hardcoded-floored away at
`single_type_kv_cache_manager.py:299`.

**Quantification**: extrapolation from M.1 — `04_savings_quantified.md`
/ Finding M.3. ~15-50% TTFT improvement on follow-up turns when the
fix is in place. Aligns with Finding D's earlier 42.6% compute-waste
measurement.

**Code (shipped, untested due to host issues)**:

1. **Cache-side scaffolding** (M.4, commit `a80ef6d1e`):
   - `BlockPool.cached_partial_block_map` + `cache_partial_block()` +
     `get_cached_partial_block()` + counters + insert log
   - `FullAttentionManager.cache_blocks` override that calls super,
     then `cache_partial_block` for the partial last block

2. **Hit-side prototype** (M.5, commit `a2533c8fc`):
   - `KVCacheManager.get_computed_blocks`: after coordinator's hit,
     probe partial cache for an R-token extension. Single-group only
     (no mamba) — hybrid path still blocked on per-group lift (M.2).
   - `_try_partial_extension()`: the probe loop

3. **Validation infrastructure**:
   - `dev/interlayer/02_partial_cache_micro.py` — hybrid baseline (measures bubble)
   - `dev/interlayer/05_nonhybrid_microbench.py` — single-group testbed
     (Qwen3-8B with block_size=256, would prove or disprove the
     end-to-end mechanism)

Both code paths are GATED by `VLLM_PARTIAL_CACHE_ENABLED=1`. Default
behavior on this branch is unchanged.

## What is NOT delivered

**Measured performance improvement.** The session did not produce a
clean before/after benchmark on a healthy host. Two reasons:

1. **Host CPU contention.** Throughout this session the dev host was
   under heavy CPU load from another user's CUDA kernel compilation
   pipeline (`/tmp/mysql` process at 15000% CPU + ptxas/cicc/cc1plus
   at 100% each). Python interpreter loads of vLLM/torch took 5-10
   minutes when they completed at all. The hybrid baseline
   microbench (M.1) did finish, but the cache-side validation run
   and the M.5 hit-side prototype run both got stuck at engine
   initialization for 10+ minutes and were eventually killed.

2. **Hybrid path needs deeper work.** Even on a healthy host the
   M.5 prototype only applies to single-group (full-attention)
   models. The hybrid bubble (the actually-big one, 42.6% on the
   real cc workload) needs per-group `num_computed_tokens` through
   the scheduler/manager/runner stack — ~400 LOC across 10 files,
   substantive single PR.

## Action items for next session (on a healthy host)

1. **Validate M.5 on Qwen3-8B**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py \
       | tee dev/interlayer/runs/05_nonhybrid_baseline.out
   CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 \
       .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py \
       | tee dev/interlayer/runs/05_nonhybrid_partial_cache.out
   diff dev/interlayer/runs/05_nonhybrid_baseline.out \
        dev/interlayer/runs/05_nonhybrid_partial_cache.out
   ```
   Expected: with env set, `turn2_cached = turn1_len` (full prior
   prompt cached including R partial tokens); `turn2_uncached = 16`
   (just the extension); turn2 wall approximately flat across R.
   If not, see `05_hit_side_impl.md` § "Known untested risks" for
   debugging hints (most likely: `allocate_slots` asserts on
   block-aligned `num_computed_tokens` — that comment exists at
   `kv_cache_manager.py:217-218` for a reason).

2. **Add `VLLM_PARTIAL_CACHE_ENABLED` to vLLM env registry** —
   the "Unknown vLLM environment variable detected" warning at
   `envs.py:2052` is noise. One-line addition.

3. **If M.5 validates on Qwen3-8B**, open the per-group
   `num_computed_tokens` PR for hybrid:
   - `request.py`: `num_computed_tokens` → `dict[int, int]` keyed by group
   - `kv_cache_coordinator.py`: per-group hit lengths
   - `kv_cache_manager.py`: per-group `allocate_slots`
   - `gpu_model_runner.py`: per-group `_build_attn_group_metadata`
   - Then re-run `02_partial_cache_micro.py` on Qwen3.5-35B-A3B,
     expect `turn2_cached = turn1_len` there too.
   - Re-run `dev/compare_lru_lpb.py` for end-to-end cc-workload
     TTFT/TPOT/throughput numbers (the "Finding M.6" measurement).

4. **Eviction integration** — `cached_partial_block_map` grows
   monotonically today. Hook `BlockPool.evict_blocks` to also remove
   the partial-cache entry for any block_id being evicted. ~10 LOC.

## Verdict

The work is in a clean, reviewable state: design quantified,
mechanism shipped as a flagged opt-in, and a validation testbed in
place. The only missing step is the empirical confirmation, which
needs a healthy host (1-2 hours of clean runtime). If the
single-group prototype validates, the hybrid path is just a
mechanical-but-bulky per-group lift on top — clear path to a real
end-to-end win on the cc workload.
