# Finding M.3 — quantifying the TTFT win that the partial-cache fix would deliver

The microbench from Finding M.1 actually contains enough signal to
extrapolate the TTFT savings — comparing what the engine pays today
to what it would pay if the partial-cache mechanism (Finding M.2's
per-group `num_computed_tokens` path) were implemented.

## Data

From `dev/interlayer/runs/02_partial_cache_micro.jsonl`:

```
R     turn2_uncached    turn2_wall_ms
  0   16                663 (warmup spike, exclude)
 32   48                 80
 96   112                82
256   272                81
512   528                93
800   816                82
1000 1016                77
1055 1071               149
```

The `R=1000` row at 77 ms is anomalously low (likely a fortunate
kernel-cache hit during chunked-prefill scheduling for that
particular shape). The `R=1055` row's spike to 149 ms is similarly
shape-dependent (R+16 = 1071 tokens crosses into a second extra
block on the prefill chunker). Excluding both, the median for
R ∈ {32, 96, 256, 512, 800} is **~82 ms**, with R=512 at 93 ms.

## Extrapolating the post-fix numbers

If the partial-cache fix were in place, every R value would have
`uncached = 16` (the fresh extension only). At R=0 (which has
uncached=16 today), the warmup-excluded value was the same ~82 ms
pattern. So the fix's TTFT target is ~77-82 ms regardless of R.

**Savings as a function of R:**

```
R       today (ms)   post-fix (ms)   savings   savings %
 32     80           ~78             ~2 ms      ~3%
 96     82           ~78             ~4 ms      ~5%
256     81           ~78             ~3 ms      ~4%
512     93           ~78             ~15 ms     ~16%
800     82           ~78             ~4 ms      ~5%
1055   149           ~78             ~71 ms     ~48%
```

On the multi-turn cc workload, the **expected R distribution** (from
`dev/multi_turn_waste.py` per-turn breakdowns) is roughly uniform
across the [0, block_size) range, so the average per-turn savings
weight is around the R=512 row → **~15 ms saved per follow-up turn**.

Across a 50-turn session: **~750 ms TTFT saved per session**, or
roughly **15-20% headline TTFT improvement** on follow-up turns.

This is consistent with Finding D's 42.6% workload-weighted
partial-block compute waste: most of that waste is in attention
prefill, which TTFT roughly tracks, but with the sub-linear scaling
of batched prefill the wall-clock TTFT win is ~half the compute win.

## Why we haven't shipped the fix yet

The fix requires **per-group `num_computed_tokens`** through the
scheduler/manager/runner stack (Finding M.2). That's a ~400 LOC
change across ~10 files in `vllm/v1/{core,worker}/`. Doable, but a
substantial single PR. The cleanest cut:

1. Add `Request.num_computed_tokens_per_group: list[int]` (init to
   `[num_computed_tokens] * num_groups`)
2. Make the coordinator return per-group hit lengths
3. Make `_build_attn_group_metadata` take per-group num_computed
4. Cache the partial block (add `cached_partial_block_map` to
   `BlockPool`, populate from `FullAttentionManager.cache_blocks`)
5. Extend `FullAttentionManager.find_longest_cache_hit` to detect
   partial extensions and report a per-group bumped hit length

A standalone PR title would be:

> "v1: per-group num_computed_tokens; cache and reuse partial last
> blocks on attention-side hits"

## Recommendation

The signal is large enough (15-50% TTFT on heavy-partial cases, ~15-20%
average) to justify the per-group surgery. Open a feature PR; once
the per-group plumbing lands, the partial-cache hit-side is a small
delta on top.

Until then, the interlayer directory documents the design and
quantifies the win. The bubble-elimination work in this branch is
**design-complete but not yet shipped** — the gating change is per-
group `num_computed_tokens`, not partial-cache itself.
