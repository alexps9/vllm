#!/usr/bin/env bash
# Profile the EngineCore subprocess under PathA --mode l1_only.
# Strategy: launch compare_lru_lpb in background, wait for the
# "(EngineCore pid=XXX)" log line, then attach py-spy record to that
# PID and dump a flame graph after a fixed duration.
set -euo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
OUT="$REPO/dev/intralayer/verify/6_lpb_heap_perf/runs/profile_e2e_l1only.flamegraph.svg"
LOG="$REPO/dev/intralayer/verify/6_lpb_heap_perf/runs/profile_e2e_l1only.driver.out"

# Launch driver in bg; capture stderr to LOG so we can grep for the PID.
CUDA_VISIBLE_DEVICES=5,6 KMP_AFFINITY=disabled \
  "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
    --mode l1_only --tag _pathA_profile --trial 99 \
    --util 0.9 --tp 2 --phase-f-scale 1 \
  > "$LOG" 2>&1 &
DRIVER_PID=$!
echo "driver pid: $DRIVER_PID"

# Wait until EngineCore announces its PID
ENGINE_PID=""
for i in {1..120}; do
  ENGINE_PID=$(grep -oE 'EngineCore pid=([0-9]+)' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  if [ -n "$ENGINE_PID" ]; then break; fi
  sleep 2
done

if [ -z "$ENGINE_PID" ]; then
  echo "FAIL: never saw EngineCore pid in log" >&2
  kill $DRIVER_PID 2>/dev/null || true
  exit 1
fi
echo "engine pid: $ENGINE_PID — attaching py-spy"

# Wait an extra 30s so engine finishes init + Phase A warm before recording.
sleep 30

py-spy record --pid "$ENGINE_PID" --output "$OUT" \
  --duration 180 --format flamegraph --subprocesses

echo "flame graph: $OUT"

# Let driver finish (or kill it after 12 min to bound).
wait $DRIVER_PID
echo "driver done"
