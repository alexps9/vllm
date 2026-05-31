# HiMA verification scenarios — index

One folder per scenario. Convention for each `<NN>_<name>/`:
`README.md` (what it tests + **repro command**) · `RESULTS.md` (verdict +
n=3 numbers) · `run.sh` (one-command reproducer) · `runs/` (raw jsonl/out).

Shared harness lives in `runs/scripts/` (server) and `dev/intralayer/`
(`compare_lru_lpb.py`, `driver.py`, `cc_long_traces.jsonl`).

| # | scenario | what | verdict | repro |
|---|---|---|---|---|
| 1 | [l1_isolation_existing_tests](1_l1_isolation_existing_tests/) | L1 (LPB) vs LRU on PathA/B + pressure curve | ✅ **L1 wins**: 35B −8.8…−10.7%, 122B −12.2% PhaseH TTFT; anchor survives 2× the cold-burst pressure | `run.sh` |
| 4 | [lpb_scoring_variants](4_lpb_scoring_variants/) | discriminate 2 suspected LPB scoring bugs (`VLLM_HIMA_LPB_SCORING`) | ⚪ **no-op**: lazy=eager=depth_tokens bit-identical; keep `lazy` | `run.sh` |
| 5 | [window_sensitivity](5_window_sensitivity/) | `VLLM_HIMA_HPB_WINDOW_S` sweep | ✅ done: decay hypothesis directionally right but small (cliff K≈20→25 across 60s→600s); pin win=3600 | `run.sh` |
| 7 | [lpb_worst_case](7_lpb_worst_case/) | LPB under heavy decoy pressure (scale≥20) | ✅ **no failure mode**: bit-identical win at scale 10/20/40; scale is the wrong lever | `run.sh` |
| 8 | [post_l2_removal_smoke](8_post_l2_removal_smoke/) | confirm L1 intact after L2 deletion | ✅ bit-identical to prior l1_only (hit% 88.87, anchor 126720) | `run.sh` |
| 9 | [swebench_w2_real](9_swebench_w2_real/) | **Songyang's real SWE-bench scenario** (W2, SWE-Bench-Lite agents) | 🔄 running — does L1 help on real agent traffic? | `run.sh` |
| 10 | [recency_fix_reverify](10_recency_fix_reverify/) | e2e re-verify of the recency-aware LPB fix | ✅ Path A win preserved (−10.8%, n=3); W2 inversion eliminated (protected-evict 37%→~4%, L1 ≈ LRU) | `README` |
| — | [repro_post_cleanup_2026-05-26](repro_post_cleanup_2026-05-26/) | post-cleanup PathA re-verification | ✅ no regression from cleanup commits | `run.sh` |

Archived (removed features, kept for reference): [`dev/archive/L2/`](../../archive/L2/)
— the L2 (admitter/budgeter/planner) investigation incl. the old verify/3.

## Net HiMA status
**L1 (LPB anchor protection) wins** on both target models (35B −10.7%, 122B
−12.2%, fresh n=3) with a characterized 2× survival window and no found
failure mode. **L2 removed** (measured neutral). **pcache removed** (no
value on hybrid). The open question scenario 9 answers: does L1's win on the
*designed* anchor pattern carry over to Songyang's *real* SWE-bench traffic?
