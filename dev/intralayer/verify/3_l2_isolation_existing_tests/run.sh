#!/usr/bin/env bash
# Repro driver for verify/3 — L2-isolation rerun of existing intralayer tests.
# Mirror of verify/1's run.sh but with --mode l2_only.
#
# Usage:
#   cd /data/yuzhou/projects/vllm-songyang
#   GPUS="5,6" bash dev/intralayer/verify/3_l2_isolation_existing_tests/run.sh pathA
#   GPUS="1,2,3,4" bash dev/intralayer/verify/3_l2_isolation_existing_tests/run.sh pathB
#   bash dev/intralayer/verify/3_l2_isolation_existing_tests/run.sh pressure
set -euo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
OUTDIR="$REPO/dev/intralayer/verify/3_l2_isolation_existing_tests/runs"
mkdir -p "$OUTDIR"

WHAT="${1:?usage: $0 {pathA|pathB|pressure|all}}"
GPUS="${GPUS:-5,6}"

run_compare() {
  local tag="$1" tp="$2" model="$3" gpus="$4"
  for trial in 1 2 3; do
    echo "[verify/3] compare_lru_lpb l2_only ${tag} t${trial} on GPUs=${gpus}"
    CUDA_VISIBLE_DEVICES=$gpus KMP_AFFINITY=disabled \
      "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
        --mode l2_only --tag "_$tag" --trial $trial \
        --util 0.9 --tp $tp --phase-f-scale 10 \
        ${model:+--model "$model"} \
      > "$OUTDIR/compare_l2_only_${tag}_t${trial}.out" 2>&1
    cp "$REPO/dev/intralayer/runs/vllm/compare_l2_only_${tag}_t${trial}.jsonl" "$OUTDIR/"
  done
}

run_pressure() {
  echo "[verify/3] e2e_l1_pressure_curve l2_only on GPUs=${GPUS}"
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled \
    "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/e2e_l1_pressure_curve.py" \
      --mode l2_only \
      --out "$OUTDIR/e2e_l1_pressure_curve_l2_only.jsonl" \
    > "$OUTDIR/e2e_l1_pressure_curve_l2_only.out" 2>&1
}

case "$WHAT" in
  pathA)    run_compare pathA 2 "" "$GPUS" ;;
  pathB)    run_compare pathB 4 "Qwen/Qwen3.5-122B-A10B" "$GPUS" ;;
  pressure) run_pressure ;;
  all)
    run_compare pathA 2 "" "$GPUS"
    run_compare pathB 4 "Qwen/Qwen3.5-122B-A10B" "${GPUS_B:-1,2,3,4}"
    run_pressure
    ;;
  *) echo "usage: $0 {pathA|pathB|pressure|all}" >&2; exit 2 ;;
esac
echo "[verify/3] done"
