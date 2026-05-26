#!/usr/bin/env bash
# Profile EngineCore subprocess under PathA L1-only.
# Strategy: launch driver bg, wait 90s for engine to be alive and past
# Phase A, then attach py-spy and record for 120s (capturing Phase B/E).
set -euo pipefail
REPO="/data/yuzhou/projects/vllm-songyang"
OUTDIR="$REPO/dev/intralayer/verify/6_lpb_heap_perf/runs"
FLAME="$OUTDIR/profile_engine_l1only.flamegraph.svg"
SPEEDSCOPE="$OUTDIR/profile_engine_l1only.speedscope.json"
LOG="$OUTDIR/profile_engine_l1only.driver.out"

CUDA_VISIBLE_DEVICES=5,6 KMP_AFFINITY=disabled \
  "$REPO/.venv/bin/python" -u "$REPO/dev/intralayer/compare_lru_lpb.py" \
    --mode l1_only --tag _pathA_profile --trial 99 \
    --util 0.9 --tp 2 --phase-f-scale 1 \
  > "$LOG" 2>&1 &
DRIVER_PID=$!
echo "driver pid: $DRIVER_PID"

# Sleep generously past engine init + Phase A (~90 s typical).
sleep 90

# Find the EngineCore PID once it has emerged in the log.
ENGINE_PID=""
for i in {1..30}; do
  ENGINE_PID=$(grep -oE 'EngineCore pid=([0-9]+)' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  if [ -n "$ENGINE_PID" ]; then break; fi
  sleep 5
done

if [ -z "$ENGINE_PID" ]; then
  echo "FAIL: no EngineCore pid in log after 90+150s" >&2
  cat "$LOG" | tail -30 >&2
  kill $DRIVER_PID 2>/dev/null || true
  exit 1
fi
echo "engine pid: $ENGINE_PID"

# Record 180s (covers all of Phase B, much of Phase E)
py-spy record --pid "$ENGINE_PID" --output "$FLAME" --duration 180 \
  --format flamegraph --subprocesses --rate 100 2>&1 | tail -20

# Also dump a speedscope JSON for searchable browsing
py-spy record --pid "$ENGINE_PID" --output "$SPEEDSCOPE" --duration 60 \
  --format speedscope --subprocesses --rate 200 2>&1 | tail -10 || true

echo "flame: $FLAME"
echo "speedscope: $SPEEDSCOPE"

# Bound the driver so we don't leak GPU memory if the test fails mid-run.
wait $DRIVER_PID || true
echo "driver done"
