#!/bin/bash
# verify/3 fresh re-measure — settle whether L2-only actually regresses.
#
# The published +26.6% L2-only TTFT regression was computed vs a STALE LRU
# baseline (captured on a different GPU pair / system load) — the same
# stale-baseline artifact class that produced the L1 "phantom regression"
# (verify/6 journal/07). This reruns l2_only AND lru back-to-back, same
# environment, n=3, so the comparison is apples-to-apples.
#
# compare_lru_lpb.py PathA, util=0.9, TP=2, phase-f-scale=10, GPUs 5,6.
# (l2_only uses the LRU free queue — LPB scoring knob is irrelevant here.)
set -uo pipefail
REPO=/data/yuzhou/projects/vllm-songyang
PY=$REPO/.venv/bin/python
OUT=$REPO/dev/intralayer/verify/3_l2_isolation_existing_tests/runs
INTER=$REPO/dev/intralayer/runs/vllm
mkdir -p $OUT
GPUS="${GPUS:-5,6}"

run_one() { # mode trial
  local mode=$1 trial=$2 tag="_pathA_fresh"
  echo "===== $mode t$trial start $(date -Iseconds) ====="
  pkill -9 -u yuzhou -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u yuzhou -f "VllmWorker" 2>/dev/null
  sleep 8
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled \
    "$PY" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
      --mode "$mode" --tag "$tag" --trial "$trial" \
      --util 0.9 --tp 2 --phase-f-scale 10 \
    > "$OUT/${mode}_pathA_fresh_t${trial}.out" 2>&1
  echo "  $mode t$trial done rc=$? at $(date -Iseconds)"
  cp "$INTER/compare_${mode}${tag}_t${trial}.jsonl" "$OUT/" 2>/dev/null || true
}

for trial in 1 2 3; do
  run_one lru "$trial"
  run_one l2_only "$trial"
done
echo "===== verify/3 fresh n3 ALL DONE $(date -Iseconds) ====="
