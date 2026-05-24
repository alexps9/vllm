# SPDX-License-Identifier: Apache-2.0
"""Concurrent (multi-stream) version of 09_cc_workload_compare.py
to validate M.13's prediction that the +14% wall regression is
specific to single-stream workloads.

If true, partial cache should be neutral or net positive on a
multi-stream workload because the bigger in-flight batches amortize
the per-launch fixed overhead.

Submits all 10 cc sessions' turns concurrently (after warm-up) using
multiple SamplingParams in a single llm.generate(prompts=[...])
batch — this is the "multi-stream burst" scenario.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/14_concurrent_workload.py
  CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 \\
      .venv/bin/python -u dev/interlayer/14_concurrent_workload.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 1024
DATA = Path(__file__).resolve().parent.parent / "cc_long_traces.jsonl"
N_SESSIONS = 10
MAX_PROMPT_TOKENS = 16_000
N_TPOT_TOKENS = 20
MAX_NUM_SEQS = 16  # bigger to support concurrent requests
CONCURRENT_GROUP_SIZE = 8  # issue this many sessions' next-turn at once


def flatten_content(content):
    if content is None: return ""
    if isinstance(content, str): return content
    if not isinstance(content, list): return str(content)
    parts = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p)); continue
        t = p.get("type", "")
        if t == "text":
            parts.append(p.get("text", ""))
        elif t == "tool_use":
            inner = p.get("input", "")
            if not isinstance(inner, str):
                inner = json.dumps(inner, ensure_ascii=False)
            parts.append(f"<tool_use name={p.get('name', '')} id={p.get('id', '')}>{inner}</tool_use>")
        elif t == "tool_result":
            inner = p.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", str(x)) if isinstance(x, dict) else str(x) for x in inner
                )
            elif not isinstance(inner, str):
                inner = str(inner)
            parts.append(f"<tool_result id={p.get('tool_use_id', '')}>{inner}</tool_result>")
        else:
            parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)


def msg_to_chunk(m):
    role = m.get("role", "user")
    return f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>\n"


def main():
    mode = ("partial_cache" if os.environ.get("VLLM_PARTIAL_CACHE_ENABLED", "0") == "1"
            else "baseline")
    out_dir = Path("dev/interlayer/runs/m14")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{mode}.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sessions = [
        json.loads(l)["messages"]
        for l in DATA.read_text().splitlines() if l.strip()
    ][:N_SESSIONS]

    print(f"[{mode}] Loading {MODEL} (TP=1, util=0.6, block_size={BLOCK_SIZE}, "
          f"max_num_seqs={MAX_NUM_SEQS})")
    llm = LLM(
        model=MODEL, tensor_parallel_size=1, dtype="bfloat16",
        enable_prefix_caching=True, block_size=BLOCK_SIZE,
        max_model_len=MAX_PROMPT_TOKENS + 1024,
        gpu_memory_utilization=0.6, max_num_seqs=MAX_NUM_SEQS,
        trust_remote_code=True,
    )

    def issue_batch(prompts, max_tokens):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        t0 = time.monotonic()
        outs = llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        wall = time.monotonic() - t0
        # Collect per-request stats
        results = []
        for o in outs:
            results.append({
                "cached": o.num_cached_tokens or 0,
                "prompt_len": len(o.prompt_token_ids) if o.prompt_token_ids else 0,
                "output_tokens": len(o.outputs[0].token_ids) if o.outputs else 0,
            })
        return wall, results

    # Build per-turn prompts for all sessions
    session_state = []  # [{"msgs": ..., "running_ids": ..., "turn_prompts": [...]}]
    for s_idx, msgs in enumerate(sessions):
        running_ids = []
        turn_prompts = []
        for m in msgs:
            piece = tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
            running_ids.extend(piece)
            if m.get("role") != "assistant":
                continue
            if len(running_ids) > MAX_PROMPT_TOKENS:
                break
            turn_prompts.append(list(running_ids))
        session_state.append(turn_prompts)

    # Round-robin: for each "round" up to MAX_TURNS, gather one prompt from each
    # session (if it has one), issue them as a single batch.
    max_turns = max(len(s) for s in session_state)
    t_start = time.monotonic()
    log(kind="meta", mode=mode, model=MODEL, n_sessions=N_SESSIONS,
        concurrent_batch_size=CONCURRENT_GROUP_SIZE, total_max_turns=max_turns)

    print(f"\n[{mode}] Multi-stream burst — up to {max_turns} rounds × "
          f"{CONCURRENT_GROUP_SIZE} concurrent sessions per round")

    for round_idx in range(max_turns):
        round_prompts = []
        round_sess = []
        for s_idx, prompts in enumerate(session_state):
            if round_idx < len(prompts) and len(round_prompts) < CONCURRENT_GROUP_SIZE:
                round_prompts.append(prompts[round_idx])
                round_sess.append(s_idx)
        if not round_prompts:
            break
        # TTFT pass (max_tokens=1, batched)
        ttft_wall, ttft_res = issue_batch(round_prompts, max_tokens=1)
        # Throughput pass (max_tokens=N_TPOT_TOKENS+1, batched)
        full_wall, full_res = issue_batch(round_prompts, max_tokens=N_TPOT_TOKENS + 1)
        for sidx, p, tr, fr in zip(round_sess, round_prompts, ttft_res, full_res):
            log(kind="cc_turn", session_idx=sidx, round=round_idx,
                prompt_len=len(p),
                ttft_cached=tr["cached"], ttft_wall_s=ttft_wall,  # NB: shared wall for batch
                full_cached=fr["cached"], full_wall_s=full_wall,
                output_tokens=fr["output_tokens"],
                elapsed_s=time.monotonic() - t_start)
        print(f"  round {round_idx}: {len(round_prompts)} streams, "
              f"ttft_batch_wall={ttft_wall*1000:.1f}ms full_batch_wall={full_wall*1000:.1f}ms")

    fout.close()
    total = time.monotonic() - t_start
    print(f"\n[{mode}] Done in {total:.1f}s. Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
