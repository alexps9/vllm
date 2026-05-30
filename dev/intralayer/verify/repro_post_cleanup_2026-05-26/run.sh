#!/usr/bin/env bash
# Post-cleanup full re-verification (2026-05-26).
#
# Goal: after the legacy/back-compat audit + cleanup (commit 5bec64340),
# rerun the n=3 PathA verifications (verify/1 L1-only, verify/3 L2-only)
# and confirm numbers match the pre-cleanup archive (Phase 11h).
#
# Pipe A — GPUs 1,2 : PathA compare_lru_lpb.py for {lru, l1_only, l2_only},
#                     each n=3, util=0.9, phase-f-scale=10
#
# (The original Pipe B — a partial-cache/pcache canonical workload — was
# removed when pcache was deleted from the codebase.)
#
# Outputs land in this directory's runs/ subdir with the suffix _repro
# to keep the archived Phase 11h files untouched.

set -euo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
HERE="$REPO/dev/intralayer/verify/repro_post_cleanup_2026-05-26"
OUT="$HERE/runs"
mkdir -p "$OUT"

GPUS_A="${GPUS_A:-1,2}"

PY="$REPO/.venv/bin/python"

pipe_a() {
  local intermediate="$REPO/dev/intralayer/runs/vllm"
  for mode in lru l1_only l2_only; do
    for trial in 1 2 3; do
      local tag="_pathA_repro"
      echo "[pipe A] $mode pathA t$trial on GPUs=$GPUS_A"
      CUDA_VISIBLE_DEVICES=$GPUS_A KMP_AFFINITY=disabled \
        "$PY" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
          --mode "$mode" --tag "$tag" --trial "$trial" \
          --util 0.9 --tp 2 --phase-f-scale 10 \
        > "$OUT/compare_${mode}_pathA_repro_t${trial}.out" 2>&1
      cp "$intermediate/compare_${mode}${tag}_t${trial}.jsonl" "$OUT/"
    done
  done
  echo "[pipe A] done"
}

pipe_a > "$OUT/pipe_a.log" 2>&1 &
PIPE_A_PID=$!
echo "Pipe A pid=$PIPE_A_PID (GPUs=$GPUS_A)"
echo "log A: $OUT/pipe_a.log"

wait $PIPE_A_PID; A_RC=$?
echo "Pipe A exit=$A_RC"
exit $A_RC
