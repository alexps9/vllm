# dev/ — HiMA development notes

Investigation logs and verification harnesses for the HiMA work on vLLM.

The active work is the **intralayer L1/L2** stack:

- **[`dev/intralayer/`](intralayer/)** — L1 (LPB intra-pool eviction +
  path counter) and L2 (admitter + budgeter + planner) investigation,
  benchmarks, and the `verify/` scenario suite.
  - [`intralayer/vllm.md`](intralayer/vllm.md) — vLLM HiMA L1/L2 design + findings
  - [`intralayer/scenarios.md`](intralayer/scenarios.md) — shared Phase A→H workload pipeline
  - [`intralayer/verify/`](intralayer/verify/) — per-scenario n=3 verifications

> A separate **interlayer partial-cache (pcache)** investigation once lived
> under `dev/interlayer/`. The feature delivered no value on HiMA's target
> models (hybrid: mamba's block-granular state caps the resume point;
> small-block: negligible ragged tail) and was removed from the codebase,
> along with its investigation tree, in 2026-05. See git history if needed.
