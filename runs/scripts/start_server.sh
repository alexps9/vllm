#!/usr/bin/env bash
# Usage: start_server.sh <baseline|hima|hima_l1> [port=8000] [log=/tmp/vllm.log]
#   baseline: no HiMA
#   hima:     L1 LPB + L2 partial cache (full HiMA)
#   hima_l1:  L1 LPB only (intralayer; no L2 partial cache)
#
# Required env:
#   MODEL   path to the model weights directory
#   REPO    path to the vllm repo root (defaults to script's grandparent dir)
set -euo pipefail

MODE="${1:?mode = baseline | hima | hima_l1}"
PORT="${2:-8000}"
LOG="${3:-/tmp/vllm_server.log}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"
MODEL="${MODEL:?set MODEL env var to model weights path}"

UTIL="${VLLM_UTIL:-0.85}"
TP="${VLLM_TP:-2}"
MAX_LEN="${VLLM_MAX_LEN:-65536}"
MAX_SEQS="${VLLM_MAX_SEQS:-64}"

cd "$REPO"
source .venv/bin/activate

EXTRA_ENV=()
case "$MODE" in
  baseline) ;;
  hima)
    EXTRA_ENV=(
      "VLLM_PARTIAL_CACHE_ENABLED=1"
      "VLLM_PARTIAL_CACHE_MIN_R=256"
      "VLLM_HIMA_ENABLE=1"
      "VLLM_HIMA_HPB_WINDOW_S=3600"
    )
    ;;
  hima_l1)
    EXTRA_ENV=(
      "VLLM_HIMA_ENABLE=1"
      "VLLM_HIMA_HPB_WINDOW_S=3600"
    )
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
sleep 1

echo "[$MODE] vllm serve TP=$TP util=$UTIL max_len=$MAX_LEN max_seqs=$MAX_SEQS"

nohup env "${EXTRA_ENV[@]}" VLLM_LOGGING_LEVEL=INFO \
  vllm serve "$MODEL" \
    --served-model-name qwen35 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$UTIL" \
    --enable-prefix-caching \
    --language-model-only \
    --trust-remote-code \
  > "$LOG" 2>&1 &
echo $! > "${LOG}.pid"
echo "pid=$(cat ${LOG}.pid)"
