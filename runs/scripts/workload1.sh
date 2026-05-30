#!/usr/bin/env bash
# W1: vLLM multi-turn benchmark, sweep num_turns.
# Usage: workload1.sh <out_dir> [num_clients=4] [tag=default] [turns="4 8 16 32 64 128"]
#
# Required env:
#   REPO        vllm repo root (defaults to script's grandparent dir)
#   MODEL       model weights path (used as tokenizer)
#   MODEL_NAME  served-model-name on the API (default: qwen35)
#
# Prefix sizes are auto-selected by VLLM_UTIL:
#   0.55 → aggressive (2048 common prefix, 1200/4000 avg/max, 200-320 decode)
#   else → original   (512  common prefix, 400/1500  avg/max, 60-100  decode)
set -euo pipefail

OUT_DIR="${1:?out dir}"
NUM_CLIENTS="${2:-4}"
TAG="${3:-default}"
TURNS="${4:-4 8 16 32 64 128}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"
BENCH="$REPO/benchmarks/multi_turn/benchmark_serving_multi_turn.py"
MODEL_NAME="${MODEL_NAME:-qwen35}"
TOKENIZER="${MODEL:?set MODEL env var to model weights path}"
PORT=${PORT:-8000}

# aggressive when util is lowered
if [[ "${VLLM_UTIL:-0.85}" == "0.55" ]]; then
  COMMON_PREFIX=2048; PREFIX_AVG=1200; PREFIX_MAX=4000
  INPUT_MIN=400; INPUT_MAX=700; OUT_MIN=200; OUT_MAX=320
else
  COMMON_PREFIX=512;  PREFIX_AVG=400;  PREFIX_MAX=1500
  INPUT_MIN=200; INPUT_MAX=320; OUT_MIN=60;  OUT_MAX=100
fi

cd "$REPO"; source .venv/bin/activate 2>/dev/null || true; mkdir -p "$OUT_DIR"

snap() {
  python - "${PORT}" <<'PY'
import json, sys, urllib.request
PORT=int(sys.argv[1])
T=("vllm:prefix_cache_queries_total","vllm:prefix_cache_hits_total",
   "vllm:num_preemptions_total","vllm:prompt_tokens_total",
   "vllm:prompt_tokens_cached_total","vllm:kv_cache_usage_perc")
b={k:0.0 for k in T}
txt=urllib.request.urlopen("http://127.0.0.1:" + str(PORT) + "/metrics",timeout=10).read().decode()
for l in txt.splitlines():
    if not l or l.startswith("#"): continue
    h=l.split("{",1)[0].split(" ",1)[0]
    if h not in T: continue
    try: b[h]+=float(l.rsplit(" ",1)[1])
    except: pass
print(json.dumps(b))
PY
}

for T in $TURNS; do
  SUB="$OUT_DIR/turns_${T}"; mkdir -p "$SUB"
  NUM_CONVS=$(( NUM_CLIENTS * 2 ))
  MAX_REQ=$(( NUM_CONVS * T )); (( MAX_REQ > 512 )) && MAX_REQ=512

  cat > "$SUB/conv_spec.json" <<JSON
{
  "filetype": "generate_conversations",
  "num_conversations": ${NUM_CONVS},
  "text_files": ["${REPO}/benchmarks/multi_turn/pg1184.txt"],
  "print_stats": false,
  "prompt_input": {
    "num_turns": { "distribution": "constant", "value": ${T} },
    "common_prefix_num_tokens": { "distribution": "constant", "value": ${COMMON_PREFIX} },
    "prefix_num_tokens": { "distribution": "lognormal", "average": ${PREFIX_AVG}, "max": ${PREFIX_MAX} },
    "num_tokens": { "distribution": "uniform", "min": ${INPUT_MIN}, "max": ${INPUT_MAX} }
  },
  "prompt_output": { "num_tokens": { "distribution": "uniform", "min": ${OUT_MIN}, "max": ${OUT_MAX} } }
}
JSON

  echo "==== [W1 $TAG] turns=$T clients=$NUM_CLIENTS max_req=$MAX_REQ ===="
  snap > "$SUB/metrics_before.json"
  set +e
  python "$BENCH" \
    --model "$TOKENIZER" --served-model-name "$MODEL_NAME" \
    --url "http://127.0.0.1:$PORT" \
    --input-file "$SUB/conv_spec.json" \
    --num-clients "$NUM_CLIENTS" \
    --max-active-conversations "$NUM_CLIENTS" \
    --max-num-requests "$MAX_REQ" \
    --stats-json-output "$SUB/per_request.json" \
    > "$SUB/bench.log" 2>&1; RC=$?
  set -e
  snap > "$SUB/metrics_after.json"

  python - "$SUB" "$T" "$NUM_CLIENTS" <<'PY' > "$SUB/summary.json"
import json, os, sys, statistics as st
sub, T, nc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a=json.load(open(f"{sub}/metrics_before.json")); b=json.load(open(f"{sub}/metrics_after.json"))
dq=b["vllm:prefix_cache_queries_total"]-a["vllm:prefix_cache_queries_total"]
dh=b["vllm:prefix_cache_hits_total"]-a["vllm:prefix_cache_hits_total"]
dp=b["vllm:num_preemptions_total"]-a["vllm:num_preemptions_total"]
dpt=b["vllm:prompt_tokens_total"]-a["vllm:prompt_tokens_total"]
dpc=b["vllm:prompt_tokens_cached_total"]-a["vllm:prompt_tokens_cached_total"]
ttfts,tpots,lats,inp,outp=[],[],[],[],[]
n=0
sp=f"{sub}/per_request.json"
if os.path.exists(sp):
    for r in json.load(open(sp)):
        n+=1
        if "ttft_ms" in r: ttfts.append(float(r["ttft_ms"]))
        if "tpot_ms" in r: tpots.append(float(r["tpot_ms"]))
        if "latency_ms" in r: lats.append(float(r["latency_ms"]))
        if "input_num_tokens" in r: inp.append(int(r["input_num_tokens"]))
        if "output_num_tokens" in r: outp.append(int(r["output_num_tokens"]))
def pct(xs,p):
    if not xs: return None
    xs=sorted(xs); return xs[int(p/100*(len(xs)-1))]
print(json.dumps({
    "turns":T,"num_clients":nc,"num_requests_completed":n,
    "server_prefix_cache_hit_rate":round(dh/dq,4) if dq else 0.0,
    "server_prompt_cached_ratio":round(dpc/dpt,4) if dpt else 0.0,
    "server_total_preemptions":int(dp),
    "kv_usage_perc_end":round(b["vllm:kv_cache_usage_perc"],4),
    "client_ttft_ms_avg":round(st.mean(ttfts),2) if ttfts else None,
    "client_ttft_ms_p50":pct(ttfts,50),"client_ttft_ms_p95":pct(ttfts,95),
    "client_ttft_ms_p99":pct(ttfts,99),
    "client_tpot_ms_avg":round(st.mean(tpots),2) if tpots else None,
    "client_latency_ms_avg":round(st.mean(lats),2) if lats else None,
    "client_input_tokens_avg":round(st.mean(inp),1) if inp else None,
    "client_output_tokens_avg":round(st.mean(outp),1) if outp else None,
},indent=2))
PY
  echo "[W1 $TAG] turns=$T done (rc=$RC)"
done
