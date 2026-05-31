# dev/ — HiMA development notes

Investigation logs and verification harnesses for the HiMA work on vLLM.

The active work is **intralayer L1** (LPB recency-aware intra-pool eviction +
path counter):

- **[`dev/intralayer/`](intralayer/)** — L1 design, benchmarks, and the
  `verify/` scenario suite.
  - [`intralayer/vllm.md`](intralayer/vllm.md) — vLLM HiMA L1 design + findings
  - [`intralayer/sglang.md`](intralayer/sglang.md) — sglang LPB counterpart
  - [`intralayer/scenarios.md`](intralayer/scenarios.md) — shared Phase A→H workload pipeline
  - [`intralayer/verify/INDEX.md`](intralayer/verify/INDEX.md) — per-scenario n=3 verifications
- **[`dev/interlayer/`](interlayer/)** — the cross-pool / page-size **bubble**
  in hybrid models (problem-proof stage). Mirrors sglang's `dev/interlayer/`.
  - [`interlayer/design.md`](interlayer/design.md) — why vLLM's bubble is a
    *page-size* (internal-frag) bubble, not sglang's fixed-split bubble
  - [`interlayer/0_page_bubble/`](interlayer/0_page_bubble/) — proof: block_size
    inflates to 1056 → **42.6%** KV waste on 106 real CC sessions

> **Removed features** (2026-05): **L2** (admitter + budgeter + cross-pool
> planner) measured neutral (≈ LRU) and was deleted — investigation archived
> in [`archive/L2/`](archive/L2/) for a future from-scratch redesign.
> **Interlayer partial-cache (pcache)** delivered no value on HiMA's target
> models and was removed with its investigation tree. See git history.
