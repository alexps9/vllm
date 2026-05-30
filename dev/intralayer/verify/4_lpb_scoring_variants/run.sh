#!/bin/bash
# verify/4 — LPB scoring variant attribution (Phase 10b).
#
# Question (reframed post-Phase-11): the LPB heap rewrite already made
# l1_only WIN at util=0.9 (-8.8% PhaseH TTFT, verify/1 fresh n=3). Do the
# two scoring fixes behind VLLM_HIMA_LPB_SCORING improve l1_only FURTHER,
# or is the current "lazy" (LFU-on-hits) scoring already capturing the win?
#
#   lazy               current default (score stale after record_hit;
#                      c_pool fed integer depth -> ~constant -> LFU)
#   eager              refresh score on every record_hit
#   depth_tokens       feed c_pool(depth * block_size) -> real recovery cost
#   eager_depth_tokens both
#
# Test bed = compare_lru_lpb.py PathA (Phase A->H): warms one anchor 500x,
# then decoy pressure (phase-f-scale=10) tries to evict it. Anchor survival
# under decoy pressure is exactly what the two scoring bugs affect.
# l1_only x 4 variants + lru floor (scoring must be inert under LRU), n=3.
set -uo pipefail
REPO=/data/yuzhou/projects/vllm-songyang
PY=$REPO/.venv/bin/python
OUT=$REPO/dev/intralayer/verify/4_lpb_scoring_variants/runs
INTER=$REPO/dev/intralayer/runs/vllm
mkdir -p $OUT
GPUS="${GPUS:-5,6}"

run_one() { # mode variant trial
  local mode=$1 variant=$2 trial=$3 tag="_v4_${variant}"
  echo "===== $mode/$variant t$trial start $(date -Iseconds) ====="
  pkill -9 -u yuzhou -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u yuzhou -f "VllmWorker" 2>/dev/null
  sleep 8
  CUDA_VISIBLE_DEVICES=$GPUS KMP_AFFINITY=disabled VLLM_HIMA_LPB_SCORING=$variant \
    "$PY" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
      --mode "$mode" --tag "$tag" --trial "$trial" \
      --util 0.9 --tp 2 --phase-f-scale 10 \
    > "$OUT/${mode}_${variant}_t${trial}.out" 2>&1
  echo "  $mode/$variant t$trial done rc=$? at $(date -Iseconds)"
  cp "$INTER/compare_${mode}${tag}_t${trial}.jsonl" "$OUT/" 2>/dev/null || true
}

for trial in 1 2 3; do
  for variant in lazy eager depth_tokens eager_depth_tokens; do
    run_one l1_only "$variant" "$trial"
  done
  run_one lru lazy "$trial"   # floor; scoring inert under LRU
done
echo "===== verify/4 ALL DONE $(date -Iseconds) ====="
