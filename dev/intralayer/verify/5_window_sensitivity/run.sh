#!/usr/bin/env bash
# Repro driver for verify/5 — VLLM_HIMA_HPB_WINDOW_S sweep on
# e2e_l1_pressure_curve under L1-only mode.
#
# Usage:
#   cd /data/yuzhou/projects/vllm-songyang
#   GPUS="1,2" bash dev/intralayer/verify/5_window_sensitivity/run.sh
set -euo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
OUTDIR="$REPO/dev/intralayer/verify/5_window_sensitivity/runs"
mkdir -p "$OUTDIR"
GPUS="${GPUS:-1,2}"

for win in 30 60 600 3600 86400; do
  if [ -s "$OUTDIR/e2e_pressure_l1only_win${win}.jsonl" ]; then
    echo "[verify/5] win=$win already has data, skipping"
    continue
  fi
  echo "[verify/5] e2e_l1_pressure_curve l1_only win=${win} on GPUs=${GPUS}"
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled VLLM_HIMA_HPB_WINDOW_S=$win \
    "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/e2e_l1_pressure_curve.py" \
      --mode l1_only \
      --out "$OUTDIR/e2e_pressure_l1only_win${win}.jsonl" \
    > "$OUTDIR/e2e_pressure_l1only_win${win}.out" 2>&1
done
echo "[verify/5] done"
