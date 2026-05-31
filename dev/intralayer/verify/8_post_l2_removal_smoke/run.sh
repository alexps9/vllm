#!/bin/bash
# verify/8 — confirm L1 path intact after L2 removal. Single l1_only PathA cell;
# expect PhaseH hit% 88.87 / anchor 126720 (bit-identical to prior l1_only).
set -uo pipefail
REPO=/data/yuzhou/projects/vllm-songyang
OUT=$REPO/dev/intralayer/verify/8_post_l2_removal_smoke
CUDA_VISIBLE_DEVICES=5,6 KMP_AFFINITY=disabled "$REPO/.venv/bin/python" -u \
  "$REPO/dev/intralayer/compare_lru_lpb.py" --mode l1_only --tag _postL2 --trial 1 \
  --util 0.9 --tp 2 --phase-f-scale 10 \
  > "$OUT/l1_only_postL2_t1.out" 2>&1
echo "done rc=$?"
