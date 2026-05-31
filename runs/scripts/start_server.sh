#!/usr/bin/env bash
# Usage: start_server.sh <baseline|l1_only> [port=8000] [log=/tmp/vllm.log]
#
# Modes:
#   baseline  — no HiMA
#   l1_only   — HiMA L1 (LPB intra-pool eviction)
#
# (L2 — admitter/budgeter/planner — was removed 2026-05; the old l2_only/
# full modes are gone. See dev/archive/L2/.)
#
# Required env:
#   MODEL   path to the model weights directory
#   REPO    path to the vllm repo root (defaults to script's grandparent dir)
set -euo pipefail

MODE="${1:?mode = baseline | l1_only}"
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
# LPB hit-count decay window. The recency-aware queue (2026-05) uses this as
# a real decay lever: stale hits demote to LRU order once they age out. A long
# window keeps a synthetic anchor pinned (Path A); a short one lets real agent
# traffic decay (W2). Override via VLLM_HIMA_HPB_WINDOW_S; default 3600.
HIMA_COMMON=(
  "VLLM_HIMA_HPB_WINDOW_S=${VLLM_HIMA_HPB_WINDOW_S:-3600}"
)
case "$MODE" in
  baseline) ;;
  l1_only)
    EXTRA_ENV=("${HIMA_COMMON[@]}" "VLLM_HIMA_L1_ENABLE=1")
    ;;
  *) echo "unknown mode: $MODE (use baseline | l1_only)" >&2; exit 2 ;;
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
