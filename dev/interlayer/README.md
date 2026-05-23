# dev/interlayer — vLLM-side bubble elimination

The vLLM analog of HiMA's L2 (inter-pool / cross-pool layer). vLLM has
ONE inflated KV pool (`block_size = 1056` on Qwen3.5-35B-A3B hybrid),
not two pools like sglang. Sglang's "move pages between pools via
cuMemUnmap+cuMemMap" doesn't translate. But vLLM still has a real
measurable bubble — caused by `block_size` inflation forcing each
request's last partial block to be abandoned — and this directory is
where we design, prototype, and measure mechanisms that eliminate it.

## Findings (chronological)

- **M.1** — `02_partial_cache_micro.py`: bubble baseline confirmed.
  `turn2_cached` is exactly `K * block_size` for every R > 0; the
  partial-block content is never re-cacheable.
- **M.2** — `03_per_group_hit_length.md`: the hit-side requires
  per-group `num_computed_tokens` because attention can skip the R
  cached tokens but mamba must re-prefill them (SSM state only
  cached at full-block boundaries). Architectural lift estimated at
  ~400 LOC across 10 files for hybrid models.
- **M.3** — `04_savings_quantified.md`: extrapolates 15-50% TTFT
  improvement on follow-up turns (matches Finding D's 42.6% compute
  waste at half the wall scaling).
- **M.4** — cache-side scaffolding committed: `BlockPool.cache_partial_block` +
  `get_cached_partial_block` methods, `FullAttentionManager.cache_blocks`
  override that calls them, env-var gated (`VLLM_PARTIAL_CACHE_ENABLED=1`).
- **M.5** — `05_hit_side_impl.md` + code: hit-side reader in
  `KVCacheManager.get_computed_blocks` for single-group configs
  (no mamba). After the coordinator returns its K-full-block hit,
  probes the partial cache for an R-token extension. Single-group
  only — multi-group hybrid still blocked on M.2.

## What works today

- Hybrid (Qwen3.5-35B-A3B): bubble measured and quantified, design
  complete. Hit-side blocked on the per-group lift.
- Single-group (Qwen3-8B with `block_size=256`): cache-side and
  hit-side both shipped. Theoretically a measurable TTFT win on
  follow-up turns with non-aligned partial. Not yet validated on a
  healthy host.

## Repro

Baseline microbench on hybrid (proves the bubble exists):
```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u dev/interlayer/02_partial_cache_micro.py \
    | tee dev/interlayer/runs/02_partial_cache_micro.out
```

Single-group prototype test (Qwen3-8B, requires `VLLM_PARTIAL_CACHE_ENABLED=1`):
```bash
# Baseline (env unset → should match current vLLM behavior)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py \
    | tee dev/interlayer/runs/05_nonhybrid_baseline.out

# Partial-cache enabled
CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u \
    dev/interlayer/05_nonhybrid_microbench.py \
    | tee dev/interlayer/runs/05_nonhybrid_partial_cache.out
```

Acceptance criterion: with the env var set, `turn2_cached` should equal
`turn1_len` (full prior request cached including R partial tokens),
not just `K * block_size`. TTFT for turn 2 should be approximately
constant across R values.

## What's in code (this branch)

```
vllm/v1/core/block_pool.py
  + cached_partial_block_map (dict)
  + cache_partial_block(), get_cached_partial_block()
  + _compute_partial_key()
  + 3 counters + info log on first 16 insertions
vllm/v1/core/single_type_kv_cache_manager.py
  + FullAttentionManager.cache_blocks override (calls super, then partial)
vllm/v1/core/kv_cache_manager.py
  + KVCacheManager._try_partial_extension() — the hit-side reader
  + plumbing in get_computed_blocks (single-group gate)
```

All gated by `VLLM_PARTIAL_CACHE_ENABLED=1`. Default behavior unchanged.

## What's NOT yet done

1. **Hybrid (per-group num_computed_tokens)** — the work sketched in
   `03_per_group_hit_length.md`. Estimated ~400 LOC across scheduler,
   request, kv_cache_coordinator, kv_cache_manager, gpu_model_runner.
   When complete, hybrid models get the same TTFT win.
2. **Validation on a healthy host** — the prototype hit-side ships
   syntax-clean but unsmoke-tested due to system contention on the
   dev machine. See `05_hit_side_impl.md` for the validation plan
   and debugging hints.
3. **Eviction integration** — partial cache entries are never
   evicted today; the map grows monotonically. Need to wire
   `evict_blocks` to also remove partial-cache entries pointing
   at the evicted block_id.
4. **LPB integration** — partial blocks have a different lifetime
   from full blocks. Whether LPB scoring should treat them
   differently is open.

## File map

```
dev/interlayer/
├── README.md                       # this file
├── 01_design_space.md              # full design survey (5 candidates)
├── 02_partial_cache_micro.py       # hybrid bubble baseline (Qwen3.5-35B-A3B)
├── 03_per_group_hit_length.md      # the architectural blocker for hybrid
├── 04_savings_quantified.md        # TTFT win extrapolation (15-50%)
├── 05_nonhybrid_microbench.py      # single-group testbed (Qwen3-8B)
├── 05_hit_side_impl.md             # design + risks for the hit-side
└── runs/
    ├── 02_partial_cache_micro.jsonl
    ├── 02_partial_cache_micro.out
    └── 02_partial_cache_micro_with_cache_side.out
```
