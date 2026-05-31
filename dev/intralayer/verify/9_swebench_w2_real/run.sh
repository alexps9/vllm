#!/usr/bin/env bash
# verify/9 — does L1 (LPB) help on Songyang's REAL SWE-bench scenario (W2)?
#
# For each mode in {baseline, l1_only} × trial {1,2,3}:
#   1. start a vLLM server (runs/scripts/start_server.sh) on Qwen3.5-35B-A3B
#   2. wait for /health
#   3. drive it with runs/scripts/workload2.py (SWE-Bench-Lite agents,
#      concurrency sweep) — this is Songyang's real W2 client
#   4. tear the server down, capture its log (incl. the [hima/lpb] evict
#      instrumentation that tells us whether L1's hot-heap engaged)
#
# Usage:
#   GPUS=2,3 bash dev/intralayer/verify/9_swebench_w2_real/run.sh
#
# Env overrides: GPUS (default 2,3), CONCS (default 4,16,32,64),
#                NTURNS (16), NINST (64), TRIALS (3).
set -uo pipefail

REPO="/data/yuzhou/projects/vllm-songyang"
# OUT_TAG nests results under runs/<tag>/ so a pressure probe (different
# util/concurrency) doesn't overwrite the normal-operating-point sweep.
OUT="$REPO/dev/intralayer/verify/9_swebench_w2_real/runs${OUT_TAG:+/$OUT_TAG}"
mkdir -p "$OUT"

export REPO="$REPO"  # start_server.sh mis-derives this from its own path
export MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
export VLLM_TP="${VLLM_TP:-2}"
export VLLM_UTIL="${VLLM_UTIL:-0.85}"
GPUS="${GPUS:-2,3}"
PORT="${PORT:-8000}"
CONCS="${CONCS:-4,16,32,64}"
NTURNS="${NTURNS:-16}"
NINST="${NINST:-64}"
TRIALS="${TRIALS:-3}"

wait_for_health() {
  local port="$1" log="$2" deadline=$(( SECONDS + 1200 ))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    # bail early if the server process died
    if [[ -f "${log}.pid" ]] && ! kill -0 "$(cat "${log}.pid")" 2>/dev/null; then
      echo "[verify/9] server pid died during startup; see $log" >&2
      return 1
    fi
    sleep 5
  done
  echo "[verify/9] server health timeout (1200s); see $log" >&2
  return 1
}

run_cell() {
  local mode="$1" trial="$2"
  local tag="${mode}_t${trial}"
  local log="$OUT/server_${tag}.log"
  local odir="$OUT/${tag}"
  mkdir -p "$odir"

  echo "==== [verify/9] $tag : starting server ($mode) on GPUs=$GPUS ===="
  CUDA_VISIBLE_DEVICES="$GPUS" KMP_AFFINITY=disabled \
    bash "$REPO/runs/scripts/start_server.sh" "$mode" "$PORT" "$log"

  if ! wait_for_health "$PORT" "$log"; then
    echo "[verify/9] $tag: server never became healthy — skipping cell" >&2
    fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
    sleep 3
    return 1
  fi
  echo "[verify/9] $tag: server healthy, launching W2 client"

  CUDA_VISIBLE_DEVICES="$GPUS" "$REPO/.venv/bin/python" -u \
    "$REPO/runs/scripts/workload2.py" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --metrics-url "http://127.0.0.1:${PORT}/metrics" \
      --model qwen35 \
      --out-dir "$odir" \
      --concurrencies "$CONCS" \
      --num-turns "$NTURNS" \
      --n-instances "$NINST" \
      --seed 42 \
    > "$odir/client.out" 2>&1
  echo "[verify/9] $tag: client done rc=$?"

  fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
  sleep 5
}

for trial in $(seq 1 "$TRIALS"); do
  for mode in baseline l1_only; do
    run_cell "$mode" "$trial"
  done
done
echo "[verify/9] all cells done"
