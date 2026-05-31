# 10. Recency-aware LPB fix — e2e re-verification

After the recency-aware LPB rewrite (single-tier→three-tier `(priority,
recency)` with window decay; commit on branch HiMA, see
[`vllm.md`](../../vllm.md)), re-verify two things end-to-end:

1. **Path A (regression guard)** — does the rewrite *preserve* L1's synthetic
   anchor-survival win? Run with the **long** window (3600 s) the anchor
   scenario needs (decay stays inert, so the win should be intact).
2. **W2 short-window (the fix's target)** — with a **short** window (60 s) so
   the decay engages, does the fix remove the pre-fix inversion where L1
   *lost* hits to LRU under pressure (verify/9: conc=256, L1 1.5% vs LRU
   3.5%)?

## How to repro

```bash
# Path A n=3 (long window, default), GPUs e.g. 6,7 — SEQUENTIAL only:
for mode in lru l1_only; do for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python -u dev/intralayer/compare_lru_lpb.py \
    --mode $mode --tag _recencyA --trial $t --util 0.9 --tp 2 --phase-f-scale 10
done; done

# W2 short-window n=3 (window=60 lets stale hits decay), GPUs 2,3:
OUT_TAG=pressure_w60 VLLM_HIMA_HPB_WINDOW_S=60 VLLM_UTIL=0.30 \
  CONCS=128,256 NINST=256 TRIALS=3 GPUS=2,3 PORT=8009 \
  bash dev/intralayer/verify/9_swebench_w2_real/run.sh
```

> **Host note:** do **not** run two GPU jobs concurrently on this host — it
> triggers a CUDA-init wedge (3 times on 2026-05-31). Run sequentially. Wedged
> `vllm serve` ignore SIGTERM; the W2 harness now SIGKILLs them.

## Results → [`RESULTS.md`](RESULTS.md)
