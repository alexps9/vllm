# Archive — HiMA L2 (admitter / budgeter / cross-pool planner)

**Status: removed from the codebase 2026-05-31. To be redesigned from
scratch.** This folder is a reference archive of the L2 investigation; the
L2 *code* no longer exists in `vllm/`.

## Why removed

L2 (the inter-pool layer: admitter + bisection budgeter + cross-pool
planner, executed via the VMM actuator) was measured **neutral** — within
noise of LRU on the verified workloads:

- Fresh same-env n=3 (verify/3, archived here under `verify_3_l2_isolation/`):
  l2_only PhaseH TTFT = **−1.3 % vs LRU** (within noise), identical hit% and
  anchor survival. l2_only uses the LRU free queue, so its eviction order is
  identical to LRU; L2 added only sub-noise admission overhead.
- The previously published **+26.6 % regression** was a stale-baseline
  phantom (compared current code against an LRU archive captured on a
  different GPU pair / load), debunked by the fresh measurement.

Since L2 delivered no measurable benefit and added significant complexity
woven into the shared `HiMARuntime`, it was deleted to keep the codebase
clean for a from-scratch redesign. **HiMA's verified value is L1 (LPB
anchor protection)**, which is untouched.

## What was deleted from `vllm/`

- `vllm/v1/core/hima/inter_pool/` — admitter.py, budgeter.py,
  cross_pool_planner.py, pressure_adapter.py
- `vllm/v1/core/hima/actuator/` — vmm_pool.py, remap.py, cuda_driver.py
- `vllm/v1/core/hima/budgeter_task.py`
- `vllm/v1/core/hima/metrics.py`, `telemetry.py` (L2 metrics/telemetry)
- L2 wiring in `integration.py` (decide_admission/planner_tick/admitter/
  budgeter/planner/actuator fields), `config.py` (hima_l2_enabled,
  hima_page_size_bytes, budget/queue/ewma knobs), `__init__.py` exports,
  `scheduler.py` (admission consult + telemetry feed), `engine/core.py`
  (budgeter task + actuator sizing), `config/cache.py` + `arg_utils.py`
  (hima_l2_enabled flag).

The full pre-removal source is recoverable from git history (the commit
that removed it, and its parent).

## Contents here

- `verify_3_l2_isolation/` — the L2-isolation verify scenario: driver-less
  README, `fresh_n3_result.md` (the −1.3 % verdict), runs, journals.

## For the redesign

Key lessons to carry forward:
- L2's job (cross-pool KV↔mamba page rebalancing under pressure) never
  bound on these workloads at util=0.9 — the pool didn't get pressured
  enough to need cross-pool moves (cf. verify/7: even 4× decoy scale didn't
  reach the eviction cliff at util=0.9). A redesign should first establish
  a workload where inter-pool pressure is real before building machinery.
- L2-only ≡ LRU on eviction because it didn't touch the free-queue order;
  any future L2 that wants to help must actually change what gets evicted
  or admitted, and be validated fresh same-env (not vs a stale baseline).
