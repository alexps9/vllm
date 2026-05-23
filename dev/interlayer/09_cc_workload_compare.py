# SPDX-License-Identifier: Apache-2.0
"""End-to-end cc workload comparison: baseline vs partial-cache (Finding M.9).

Replays N real Claude Code sessions through Qwen3-8B (non-hybrid full
attention, so the M.5 partial-cache hit-side actually fires). Each
turn issued twice (max_tokens=1 for TTFT, max_tokens=21 for
throughput). Per-turn metrics logged to jsonl. Run twice (with/without
VLLM_PARTIAL_CACHE_ENABLED=1) to produce a baseline-vs-fix comparison.

This is the "real workload" measurement that turns the M.7 microbench
win (43% TTFT on synthetic bubble) into a workload-weighted number on
actual cc traffic.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py
  CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 \\
      .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py
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
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")  # quieter logs

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 1024
DATA = Path(__file__).resolve().parent.parent / "cc_long_traces.jsonl"
N_SESSIONS = 10
MAX_PROMPT_TOKENS = 16_000
N_TPOT_TOKENS = 20


def flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p))
            continue
        t = p.get("type", "")
        if t == "text":
            parts.append(p.get("text", ""))
        elif t == "tool_use":
            inner = p.get("input", "")
            if not isinstance(inner, str):
                inner = json.dumps(inner, ensure_ascii=False)
            parts.append(
                f"<tool_use name={p.get('name', '')} "
                f"id={p.get('id', '')}>{inner}</tool_use>"
            )
        elif t == "tool_result":
            inner = p.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", str(x)) if isinstance(x, dict) else str(x)
                    for x in inner
                )
            elif not isinstance(inner, str):
                inner = str(inner)
            parts.append(
                f"<tool_result id={p.get('tool_use_id', '')}>"
                f"{inner}</tool_result>"
            )
        else:
            parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)


def msg_to_chunk(m: dict) -> str:
    role = m.get("role", "user")
    return f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>\n"


def main() -> None:
    mode = ("partial_cache" if os.environ.get("VLLM_PARTIAL_CACHE_ENABLED", "0") == "1"
            else "baseline")
    out_dir = Path("dev/interlayer/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"09_cc_{mode}.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sessions = [
        json.loads(l)["messages"]
        for l in DATA.read_text().splitlines() if l.strip()
    ][:N_SESSIONS]

    print(f"[{mode}] Loading {MODEL} (TP=1, util=0.6, block_size={BLOCK_SIZE}) …")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_PROMPT_TOKENS + 1024,
        gpu_memory_utilization=0.6,
        max_num_seqs=4,
        trust_remote_code=True,
    )

    def issue(token_ids: list[int], max_tokens: int) -> dict:
        # ignore_eos: force the model to always generate exactly max_tokens
        # so that wall-time comparisons aren't polluted by variance in
        # output length (bf16 non-determinism can shift EOS timing).
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        t0 = time.monotonic()
        outs = llm.generate(prompts=[token_ids], sampling_params=sp, use_tqdm=False)
        wall = time.monotonic() - t0
        out = outs[0]
        n_out = len(out.outputs[0].token_ids) if out.outputs else 0
        return {
            "wall_s": wall,
            "cached": out.num_cached_tokens or 0,
            "prompt_len": len(token_ids),
            "output_tokens": n_out,
        }

    t_start = time.monotonic()
    log(kind="meta", mode=mode, model=MODEL, block_size=BLOCK_SIZE,
        n_sessions=N_SESSIONS)

    print(f"\n[{mode}] cc burst over {N_SESSIONS} sessions, "
          f"max_tokens=1 (TTFT) then max_tokens={N_TPOT_TOKENS+1} (TPOT).")
    for s_idx in range(N_SESSIONS):
        msgs = sessions[s_idx]
        running_ids: list[int] = []
        prev_len = 0
        turn = 0
        for m in msgs:
            piece = tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
            running_ids.extend(piece)
            if m.get("role") != "assistant":
                continue
            if len(running_ids) > MAX_PROMPT_TOKENS:
                break
            r_t = issue(running_ids, max_tokens=1)
            r_d = issue(running_ids, max_tokens=N_TPOT_TOKENS + 1)
            new_content = len(running_ids) - prev_len
            log(
                kind="cc_turn", session_idx=s_idx, turn=turn,
                prompt_len=len(running_ids),
                ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
                full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
                output_tokens=r_d["output_tokens"],
                new_content_tokens=new_content,
                elapsed_s=time.monotonic() - t_start,
            )
            prev_len = len(running_ids)
            turn += 1
        print(f"  session {s_idx:>2}: {turn} turns done "
              f"(elapsed {time.monotonic() - t_start:.0f}s)")

    fout.close()
    print(f"\n[{mode}] Done. Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
