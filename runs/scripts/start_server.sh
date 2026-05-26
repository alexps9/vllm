#!/usr/bin/env bash
# Usage: start_server.sh <baseline|l1_only|l2_only|full> [port=8000] [log=/tmp/vllm.log]
#
# Modes (after HiMA sub-flag split, 2026-05-26):
#   baseline  — no HiMA
#   l1_only   — HiMA L1 (LPB intra-pool eviction) + partial-cache
#   l2_only   — HiMA L2 (admitter + budgeter + planner) + partial-cache
#   full      — both L1 and L2 + partial-cache
#
# Note: the prior ``hima`` mode (full stack) is renamed to ``full`` and
# ``hima_l1`` (L1-only attempt via the legacy master switch, which was
# actually full HiMA without partial-cache) is replaced by the cleaner
# ``l1_only`` (true L1 isolation via the new sub-flag). The legacy
# ``VLLM_HIMA_ENABLE`` env var no longer works; use the sub-flags.
#
# Required env:
#   MODEL   path to the model weights directory
#   REPO    path to the vllm repo root (defaults to script's grandparent dir)
set -euo pipefail

MODE="${1:?mode = baseline | l1_only | l2_only | full}"
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
# Partial-cache + extended LPB window are shared across all HiMA modes.
HIMA_COMMON=(
  "VLLM_PARTIAL_CACHE_ENABLED=1"
  "VLLM_PARTIAL_CACHE_MIN_R=256"
  "VLLM_HIMA_HPB_WINDOW_S=3600"
)
case "$MODE" in
  baseline) ;;
  l1_only)
    EXTRA_ENV=("${HIMA_COMMON[@]}" "VLLM_HIMA_L1_ENABLE=1")
    ;;
  l2_only)
    EXTRA_ENV=("${HIMA_COMMON[@]}" "VLLM_HIMA_L2_ENABLE=1")
    ;;
  full)
    EXTRA_ENV=(
      "${HIMA_COMMON[@]}"
      "VLLM_HIMA_L1_ENABLE=1"
      "VLLM_HIMA_L2_ENABLE=1"
    )
    ;;
  *) echo "unknown mode: $MODE (use baseline | l1_only | l2_only | full)" >&2; exit 2 ;;
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
