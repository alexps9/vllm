#!/bin/bash
# verify/7 — LPB worst-case under heavy decoy pressure (phase-f-scale=20).
set -uo pipefail
REPO=/data/yuzhou/projects/vllm-songyang
PY=$REPO/.venv/bin/python
OUT=$REPO/dev/intralayer/verify/7_lpb_worst_case/runs
INTER=$REPO/dev/intralayer/runs/vllm
mkdir -p $OUT
GPUS="${GPUS:-5,6}"
SCALE="${SCALE:-20}"
run_one() { # mode trial
  local mode=$1 trial=$2 tag="_pathA_s${SCALE}"
  echo "===== $mode s${SCALE} t$trial start $(date -Iseconds) ====="
  pkill -9 -u yuzhou -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u yuzhou -f "VllmWorker" 2>/dev/null
  sleep 8
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled \
    "$PY" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
      --mode "$mode" --tag "$tag" --trial "$trial" \
      --util 0.9 --tp 2 --phase-f-scale "$SCALE" \
    > "$OUT/${mode}_s${SCALE}_t${trial}.out" 2>&1
  echo "  $mode s${SCALE} t$trial done rc=$? at $(date -Iseconds)"
  cp "$INTER/compare_${mode}${tag}_t${trial}.jsonl" "$OUT/" 2>/dev/null || true
}
for trial in 1 2 3; do
  run_one lru "$trial"
  run_one l1_only "$trial"
done
echo "===== verify/7 s${SCALE} ALL DONE $(date -Iseconds) ====="
