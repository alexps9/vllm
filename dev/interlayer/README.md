# dev/interlayer — vLLM-side bubble elimination

The vLLM analog of HiMA's L2 (inter-pool / cross-pool layer). vLLM has
ONE inflated KV pool (`block_size = 1056` on Qwen3.5-35B-A3B hybrid),
not two pools like sglang. Sglang's "move pages between pools via
cuMemUnmap+cuMemMap" doesn't apply. But vLLM still has a real,
measurable bubble — caused by `block_size` inflation forcing each
request's last partial block to be abandoned — and this directory is
where we design, prototype, and measure mechanisms that eliminate it.

## Status (this commit)

- **01_design_space.md** — full design survey: 5 candidates (partial-block cache with COW reuse, sub-block hashing, inflate-side reduction, HiMA VMM remap, tail compaction). Pick: **Candidate A (partial-block cache + COW)**.
- **02_partial_cache_micro.py** — baseline microbench, no code changes. Shows the bubble empirically across R ∈ {0, 32, 96, 256, 512, 800, 1000, 1055}. Run output in `runs/02_partial_cache_micro.{jsonl,out}`.

## Finding M.1 — the bubble is real and exactly as predicted

Microbench setup: 8 two-turn dialogues. Turn 1 prompts of length `K *
block_size + R` (K=2 full blocks, R = partial tail). Turn 2 issues
turn 1's prompt + 16 fresh tokens.

```
  R     turn1_len   turn2_cached  turn2_uncached   turn2_wall_ms
  0     2112        2112          16               663 (1st warmup spike)
  32    2144        2112          48               80
  96    2208        2112          112              82
  256   2368        2112          272              81
  512   2624        2112          528              93
  800   2912        2112          816              82
  1000  3112        2112          1016             77
  1055  3167        2112          1071             149
```

**Key observation:** `turn2_cached` is **exactly 2112 for every R**.
The partial last block from turn 1 (containing R real tokens) is
never reused. The (16 + R) "uncached" tokens are re-prefilled on
every follow-up turn. This is the hardcoded floor at
[`vllm/v1/core/single_type_kv_cache_manager.py:299`]:

```python
num_full_blocks = num_tokens // self.block_size
```

…paired with the docstring at
[`vllm/v1/core/kv_cache_coordinator.py:474`]:

> "Requiring this because we don't support partial block cache hit yet."

The "yet" is the design opportunity. With `block_size = 1056`, a
single partial block can carry up to 1055 wasted tokens. The 42.6%
workload-weighted partial-block waste measured on real cc traces
(see `dev/README.md` Finding D) is the per-workload accumulation of
this same effect across 50+ turns per session.

## Repro

```bash
# Baseline microbench (no code changes, ~5 min including model load):
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u dev/interlayer/02_partial_cache_micro.py \
    | tee dev/interlayer/runs/02_partial_cache_micro.out
```

Need: 2× ≥50 GB GPU (script asks for `gpu_memory_utilization=0.35` →
~50 GB each on H200). KMP_AFFINITY=disabled already set inside.

## Next steps (in this branch's roadmap)

1. **03_** Prototype Candidate A under a flag (`--enable-partial-cache`):
   - Add `PartialBlockHash` and `cached_partial_block_map` to BlockPool
   - Modify FullAttentionManager.cache_blocks to also cache the partial last block
   - Modify FullAttentionManager.find_longest_cache_hit to return an optional partial-extension length
   - Modify KVCacheManager.allocate_slots to handle non-block-aligned `num_new_computed_tokens` (this lifts the limitation called out in code at `kv_cache_manager.py:217-218`)
   - Add the COW memcpy path: when a partial hit lands, allocate a fresh block from the free queue and copy R tokens from the cached partial block into offsets `[0:R]` of the fresh block
   - Defer mamba's partial-state handling — let mamba re-prefill the R tokens via re-compute; only attention gets the cache benefit
2. **04_** Re-run the microbench with the flag on; expect `turn2_cached = 2112 + R` and `turn2_uncached = 16` for all R values.
3. **05_** Extend the LRU/LPB end-to-end cc workload comparison to also vary `enable_partial_cache`. Produce Finding M.2 with TTFT/TPOT/throughput numbers.

## File map

```
dev/interlayer/
├── README.md                       # this file
├── 01_design_space.md              # full survey of approaches
├── 02_partial_cache_micro.py       # baseline measurement (no code changes)
├── 03_partial_cache_impl.py        # (not yet) the COW prototype driver
├── 04_partial_cache_compare.py     # (not yet) on/off comparison
└── runs/
    ├── 02_partial_cache_micro.jsonl
    └── 02_partial_cache_micro.out
```
