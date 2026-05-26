# 06 — Hoist L1-only admitter + strip per-block contextlib (2026-05-26)

## Hypothesis

Journal/05's per-phase shape (Phase B/E/F regressing 60-150 % while
Phases G/H regress only 14-43 %) is a per-CALL fixed cost, not a
per-allocation cost. The LPB queue itself isn't the bottleneck
anymore — per-request HiMA hooks are.

Two suspected hot spots in `vllm/v1/core/sched/scheduler.py:444-472`
and `vllm/v1/core/hima/coordinator_hima.py:117-135`:

1. **scheduler.py:464** allocates a fresh `AdmissionDecision`
   dataclass on every scheduling iteration, even in L1-only where
   `decide_admission` short-circuits to OWN_FREE. Each call also pays
   `contextlib.suppress(Exception)` context manager setup + the
   function call frame.
2. **coordinator_hima.py:132-135** loops over every block in
   `request.kv_cache_blocks` with a per-block
   `with contextlib.suppress(Exception):` and a `hasattr(blk,
   "block_id")` check. Owned blocks (ref_cnt > 0 at this code path)
   are never in the free queue so `refresh_lpb_score` is a no-op for
   them, yet we still pay the per-block frame overhead.

## Changes

`vllm/v1/core/sched/scheduler.py:444`: gate the admitter consultation
on `_hima_pre.admitter is not None`. When L2 is off this skips the
`decide_admission` call entirely + the contextlib frame + the
AdmissionDecision allocation.

`vllm/v1/core/hima/coordinator_hima.py:117-135`: drop the per-block
`hasattr` and `with contextlib.suppress`; `KVCacheBlock` always has
`block_id` and `refresh_lpb_score` already tolerates a missing block
via `dict.get`.

## Expected effect

- L1-only mode: each scheduling iter saves ~1-2 µs (admitter hoist)
- Each cache_blocks call: saves ~600 ns × num_blocks (contextlib drop)
- For Phase E (100 cold prompts × 2 calls): savings ~1 ms
- For Phase B (100+ turns × 10 sessions × 2 calls): savings ~2-5 ms

These are bench-projected — the actual e2e numbers come from the n=3
PathA util=0.9 run in `runs/compare_l1_only_pathA_postopt_v2_t*` and
the py-spy flame graph from `runs/profile_e2e_l1only.flamegraph.svg`.

## RESULT

_(pending — bg job btcfb1xbx in flight, ETA ~30 min)_
