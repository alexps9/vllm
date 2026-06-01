# SPDX-License-Identifier: Apache-2.0
"""Live bubble measurement — how big is the attention-KV internal-fragmentation
bubble for a REAL long-horizon agent, measured ON the engine?

Audit dispute: an offline metric (counterfactual_block_size.py) reported "42.6%"
but that is a *recompute* ratio (filled-tail / new-content), not memory waste.
The true memory bubble = (allocated KV slots − used tokens)/allocated, which is
LENGTH-DEPENDENT (~5% @16k context, ~2% @32k, ~0.5% @104k). So the real number
depends on the resident length distribution of a live agent — measure it.

Method: load the real hybrid model (Qwen3.5-35B-A3B, TP=2, prefix-caching,
mamba align), submit a batch of REAL CC agent contexts (the 106 long-horizon
traces) at realistic mid-conversation lengths, and snapshot the engine's live
per-request block allocation (attention AND mamba groups) via
scheduler.running + coordinator.single_type_managers[g].req_to_blocks. Report:
  - attention internal fragmentation = (alloc_slots − used)/alloc_slots, live;
  - mamba allocated blocks/req (Audit-A's "mamba dominates" prong);
  - the resident length distribution that produced the number;
  - counterfactual frag at ksize=32 (what the sub-block fix would leave).
Run: CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python <this> 2>&1 | tee runs/live_bubble.out
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_VENV_BIN = os.path.dirname(sys.executable)
os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process engine -> introspectable
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import random  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: E402

MODEL = "Qwen/Qwen3.5-35B-A3B"
DATA = Path("/data/yuzhou/projects/vllm-songyang/dev/intralayer/cc_long_traces.jsonl")
MAX_LEN = 65536
KSIZE = 32                       # the sub-block fix granularity


def flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def build_contexts(tok, m: int, mode: str) -> list[list[int]]:
    """m real agent contexts as token-id lists. mode='mid' = random conversation
    progress (realistic serving snapshot); 'deep' = full conversation (capped)."""
    sessions = [json.loads(line)["messages"] for line in DATA.open()]
    rng = random.Random(0)
    rng.shuffle(sessions)
    out: list[list[int]] = []
    for msgs in sessions:
        if len(out) >= m:
            break
        ids: list[int] = []
        bounds = [0]
        for mm in msgs:
            chunk = (f"<|im_start|>{mm.get('role', 'user')}\n"
                     f"{flatten(mm.get('content'))}<|im_end|>\n")
            ids.extend(tok.encode(chunk, add_special_tokens=False))
            bounds.append(len(ids))
        if len(ids) < 200:
            continue
        if mode == "deep":
            cut = min(len(ids), MAX_LEN - 64)
        else:  # mid: a random turn boundary in [20%, 100%]
            cands = [b for b in bounds if b >= 200]
            cut = min(rng.choice(cands) if cands else len(ids), MAX_LEN - 64)
        out.append(ids[:cut])
    return out


def snapshot(llm) -> dict:
    eng = llm.llm_engine.engine_core
    sched = getattr(eng, "scheduler", None) or eng.engine_core.scheduler
    coord = sched.kv_cache_manager.coordinator
    cfg = sched.kv_cache_manager.kv_cache_config
    groups = []
    for gi, g in enumerate(cfg.kv_cache_groups):
        spec = g.kv_cache_spec
        kind = ("attention" if isinstance(spec, FullAttentionSpec)
                else "mamba" if isinstance(spec, MambaSpec) else type(spec).__name__)
        groups.append((gi, kind, spec.block_size))
    running = list(sched.running)
    per = {gi: {"alloc": 0, "used": 0, "blocks": 0, "cf32_alloc": 0} for gi, _, _ in groups}
    lengths = []
    for req in running:
        used = req.num_computed_tokens
        lengths.append(used)
        for gi, kind, bs in groups:
            mgr = coord.single_type_managers[gi]
            nb = len(mgr.req_to_blocks.get(req.request_id, []))
            per[gi]["alloc"] += nb * bs
            per[gi]["used"] += used
            per[gi]["blocks"] += nb
            per[gi]["cf32_alloc"] += ((used + KSIZE - 1) // KSIZE) * KSIZE  # ksize counterfactual
    return {"n_running": len(running), "groups": groups, "per": per, "lengths": lengths}


def report(tag: str, snap: dict) -> None:
    print(f"\n===== {tag}: n_running={snap['n_running']} =====")
    L = sorted(snap["lengths"])
    if L:
        p = lambda q: L[min(len(L) - 1, int(q * len(L)))]
        print(f"resident lengths: min={L[0]} p50={p(.5)} p95={p(.95)} max={L[-1]} "
              f"sum={sum(L)}")
    for gi, kind, bs in snap["groups"]:
        d = snap["per"][gi]
        if d["alloc"] == 0:
            continue
        frag = 100 * (d["alloc"] - d["used"]) / d["alloc"]
        cf = 100 * (d["cf32_alloc"] - d["used"]) / d["cf32_alloc"] if d["cf32_alloc"] else 0
        print(f"  group {gi} [{kind}] block_size={bs}: blocks={d['blocks']} "
              f"alloc_slots={d['alloc']:,} used={d['used']:,} "
              f"-> bubble={frag:.2f}%   (ksize=32 counterfactual: {cf:.2f}%)")
        print(json.dumps({"tag": tag, "group": gi, "kind": kind, "block_size": bs,
                          "blocks": d["blocks"], "alloc_slots": d["alloc"],
                          "used_tokens": d["used"], "bubble_pct": round(frag, 3),
                          "cf_ksize32_pct": round(cf, 3)}))


def run_mode(llm, tok, mode: str, m: int) -> None:
    prompts = [{"prompt_token_ids": ids} for ids in build_contexts(tok, m, mode)]
    print(f"\n#### mode={mode}: submitting {len(prompts)} real agent contexts "
          f"(lens: {sorted(len(p['prompt_token_ids']) for p in prompts)})")
    best = {"n_running": -1}
    holder: dict = {}

    def gen():
        holder["o"] = llm.generate(
            prompts, SamplingParams(max_tokens=24, ignore_eos=True), use_tqdm=False)

    t = threading.Thread(target=gen, daemon=True); t.start()
    for _ in range(240):
        time.sleep(0.4)
        try:
            s = snapshot(llm)
        except Exception as e:  # noqa: BLE001
            print("snapshot err:", repr(e)); break
        if s["n_running"] > best["n_running"]:
            best = s
        if not t.is_alive():
            break
    t.join()
    report(f"{mode} (peak residency)", best)


def main() -> None:
    print(f"Loading {MODEL} (TP=2, align, max_model_len={MAX_LEN}) — minutes...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(model=MODEL, tensor_parallel_size=2, trust_remote_code=True,
              dtype="bfloat16", enable_prefix_caching=True, mamba_cache_mode="align",
              max_model_len=MAX_LEN, gpu_memory_utilization=0.85, disable_log_stats=True)
    run_mode(llm, tok, "mid", m=48)     # realistic serving: agents at varied progress
    run_mode(llm, tok, "deep", m=16)    # deep long-horizon contexts (capped at MAX_LEN)


if __name__ == "__main__":
    main()
