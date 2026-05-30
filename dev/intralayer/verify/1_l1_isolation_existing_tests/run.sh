#!/usr/bin/env bash
# Repro driver for verify/1 — L1-isolation rerun of existing intralayer tests.
# Self-contained: every command needed to reproduce the result is in this file.
#
# Prereqs: phase 1 sub-flags ✅, phase 3 driver --mode flags ✅,
# dev/intralayer/cc_long_traces.jsonl ✅ (canonical conversation-trace dataset).
#
# Usage:
#   cd /data/yuzhou/projects/vllm-songyang
#   GPUS="3,4" bash dev/intralayer/verify/1_l1_isolation_existing_tests/run.sh pathA
#   GPUS="1,2,3,4" bash dev/intralayer/verify/1_l1_isolation_existing_tests/run.sh pathB
#   bash dev/intralayer/verify/1_l1_isolation_existing_tests/run.sh pressure
#
# Output: runs/compare_l1_only_path{A,B}_t{1,2,3}.{jsonl,out} +
#         runs/e2e_l1_pressure_curve_l1_only.{jsonl,out}
set -euo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
OUTDIR="$REPO/dev/intralayer/verify/1_l1_isolation_existing_tests/runs"
mkdir -p "$OUTDIR"

WHAT="${1:?usage: $0 {pathA|pathB|pressure|all}}"
GPUS="${GPUS:-3,4}"

run_compare() {
  local tag="$1" tp="$2" model="$3" gpus="$4"
  for trial in 1 2 3; do
    echo "[verify/1] compare_lru_lpb l1_only ${tag} t${trial} on GPUs=${gpus}"
    CUDA_VISIBLE_DEVICES=$gpus KMP_AFFINITY=disabled \
      "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
        --mode l1_only --tag "_$tag" --trial $trial \
        --util 0.9 --tp $tp --phase-f-scale 10 \
        ${model:+--model "$model"} \
      > "$OUTDIR/compare_l1_only_${tag}_t${trial}.out" 2>&1
    cp "$REPO/dev/intralayer/runs/vllm/compare_l1_only_${tag}_t${trial}.jsonl" "$OUTDIR/"
  done
}

run_pressure() {
  echo "[verify/1] e2e_l1_pressure_curve l1_only on GPUs=${GPUS}"
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled \
    "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/e2e_l1_pressure_curve.py" \
      --mode l1_only \
      --out "$OUTDIR/e2e_l1_pressure_curve_l1_only.jsonl" \
    > "$OUTDIR/e2e_l1_pressure_curve_l1_only.out" 2>&1
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
echo "[verify/1] done"
