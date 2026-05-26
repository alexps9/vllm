#!/usr/bin/env bash
# End-to-end sweep: start server → W1 → W2 → shutdown.
# Usage: run_sweep.sh <baseline|l1_only|l2_only|full> <out_root> [w1_clients=16] [w1_turns="4 8 16 32 64"]
#
# Modes (after HiMA sub-flag split):
#   baseline  — no HiMA
#   l1_only   — HiMA L1 (LPB intra-pool eviction) + partial-cache
#   l2_only   — HiMA L2 (admitter + budgeter + planner) + partial-cache
#   full      — both L1 and L2 + partial-cache (was "hima" before split)
#
# Required env:
#   MODEL       model weights path
#   REPO        vllm repo root (default: parent of this scripts/ dir)
#
# Optional server config (defaults shown):
#   VLLM_UTIL=0.85  VLLM_MAX_SEQS=64  VLLM_MAX_LEN=65536  VLLM_TP=2
set -euo pipefail

MODE="${1:?mode = baseline | l1_only | l2_only | full}"
OUT_ROOT="${2:?out root}"
W1_CLIENTS="${3:-16}"
W1_TURNS="${4:-4 8 16 32 64}"

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPTS")}"
export MODEL="${MODEL:?set MODEL env var}"
export REPO
PORT=8000
LOG_DIR="$OUT_ROOT/$MODE"
SERVER_LOG="$LOG_DIR/server.log"

mkdir -p "$LOG_DIR"
bash "$SCRIPTS/start_server.sh" "$MODE" "$PORT" "$SERVER_LOG"

echo "waiting for server..."
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "up (${i}s)"; break; }
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { tail -30 "$SERVER_LOG"; exit 1; }

W1_OUT="$LOG_DIR/w1"
echo "=== W1 ==="
bash "$SCRIPTS/workload1.sh" "$W1_OUT" "$W1_CLIENTS" "$MODE" "$W1_TURNS"

W2_OUT="$LOG_DIR/w2"
echo "=== W2 ==="
cd "$REPO"; source .venv/bin/activate
python "$SCRIPTS/workload2.py" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --metrics-url "http://127.0.0.1:$PORT/metrics" \
  --model qwen35 \
  --out-dir "$W2_OUT" \
  --concurrencies 1,2,4,8,16,32,64,128 \
  --num-turns 16 \
  --max-tokens 256 \
  --n-instances 32 \
  --ramp-s 0.3 \
  --max-prompt-tokens 60000

PID="$(cat "$SERVER_LOG.pid" 2>/dev/null || true)"
[ -n "$PID" ] && { kill -TERM "$PID" 2>/dev/null || true; sleep 5; kill -KILL "$PID" 2>/dev/null || true; }
fuser -k -KILL "${PORT}/tcp" 2>/dev/null || true
echo "=== $MODE DONE ==="
