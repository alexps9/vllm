# SPDX-License-Identifier: Apache-2.0
"""L1 pressure curve: anchor survival vs cold-burst size.

Generalizes Finding E.2 from a binary "29 sessions evicts everything"
result to a CURVE: at what cold-burst pressure does the anchor break?

For each pressure level K in PRESSURE_LEVELS, we:
  1) Warm anchor 5x (resets its position to MRU in the free queue).
  2) Replay K silent sessions from a fresh pool (cold burst, no probes).
  3) Probe anchor once → record num_cached_tokens.

The K cold-burst sessions are drawn from disjoint pools across pressure
levels (sessions 1..5 for K=5, sessions 6..15 for K=10, etc.), so
different pressure levels see different content. Each phase's warm
re-introduces the anchor to the tail of the free queue.

Result: a curve showing anchor cached % vs K. Should drop sharply once
cumulative new blocks exceeds KV budget.

Run:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python -u dev/e2e_l1_pressure_curve.py | tee dev/e2e_l1_pressure_curve.out
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-35B-A3B"
DATA = Path("/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl")
BLOCK_SIZE = 1056
MAX_PROMPT_TOKENS = 60_000
N_ANCHOR_WARM = 5
# Cold-burst session counts to test. Disjoint pools => need session pool
# of size sum(PRESSURE_LEVELS). 0+5+10+15+20+25+30 = 105 → fits in 106.
PRESSURE_LEVELS = [0, 5, 10, 15, 20, 25, 30]

# Maps --mode → kwargs forwarded to LLM(). Sub-flags are independent.
_MODE_KWARGS: dict[str, dict[str, bool]] = {
    "lru":     {},
    "l1_only": {"hima_l1_enabled": True},
    "l2_only": {"hima_l2_enabled": True},
    "full":    {"hima_l1_enabled": True, "hima_l2_enabled": True},
}


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=list(_MODE_KWARGS),
        default="lru",
        help="HiMA layer config. lru=baseline; l1_only / l2_only "
        "isolate one layer; full=both layers on.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL output path; defaults to "
        "dev/intralayer/e2e_l1_pressure_curve_<mode>.jsonl",
    )
    args = ap.parse_args()
    mode = args.mode
    mode_kwargs = _MODE_KWARGS[mode]
    out_jsonl = (
        args.out
        if args.out is not None
        else Path(f"dev/intralayer/e2e_l1_pressure_curve_{mode}.jsonl")
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sessions: list[list[dict]] = [
        json.loads(l)["messages"] for l in DATA.read_text().splitlines() if l.strip()
    ]
    needed = sum(PRESSURE_LEVELS) + 1  # +1 for session-0 (anchor)
    assert len(sessions) >= needed, f"need {needed} sessions, have {len(sessions)}"
    print(f"Loaded {len(sessions)} sessions; using {needed}")

    # Anchor = session 0's first user message.
    first_user = next(m for m in sessions[0] if m["role"] == "user")
    anchor_ids = tokenizer.encode(msg_to_chunk(first_user), add_special_tokens=False)
    anchor_len = len(anchor_ids)
    print(f"Anchor: {anchor_len} tokens (~{anchor_len / BLOCK_SIZE:.1f} blocks)")

    print(f"Loading {MODEL} (TP=2, util=0.35, mamba_cache_mode=align, "
          f"mode={mode}, mode_kwargs={mode_kwargs})...")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=MAX_PROMPT_TOKENS + 1024,
        gpu_memory_utilization=0.35,
        max_num_seqs=64,
        trust_remote_code=True,
        **mode_kwargs,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def issue(token_ids: list[int]) -> tuple[int, float]:
        t0 = time.monotonic()
        outs = llm.generate(prompts=[token_ids], sampling_params=sp, use_tqdm=False)
        return (outs[0].num_cached_tokens or 0, time.monotonic() - t0)

    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    t_start = time.monotonic()

    # Walk through the disjoint session pools for each pressure level.
    next_offset = 1  # skip session 0 (used as anchor)
    results: list[dict] = []

    for K in PRESSURE_LEVELS:
        print(f"\n=========== Phase: pressure K={K} ===========")
        # Warm anchor 5×.
        print(f"  Warming anchor {N_ANCHOR_WARM}× ...")
        for i in range(N_ANCHOR_WARM):
            cached, _ = issue(anchor_ids)
        # Burst K sessions silently.
        burst_session_indices = list(range(next_offset, next_offset + K))
        next_offset += K
        cum_new = 0
        cum_blocks_lb = 0
        n_requests = 0
        if K > 0:
            print(f"  Replaying {K} sessions (indices {burst_session_indices[0]}..."
                  f"{burst_session_indices[-1]}) with no probes...")
        for s_idx in burst_session_indices:
            msgs = sessions[s_idx]
            running_ids: list[int] = []
            prev_len = 0
            for m in msgs:
                running_ids.extend(
                    tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
                )
                if m.get("role") != "assistant":
                    continue
                if len(running_ids) > MAX_PROMPT_TOKENS:
                    break
                cached, _ = issue(running_ids)
                new_content = len(running_ids) - prev_len
                cum_new += new_content
                cum_blocks_lb += new_content // BLOCK_SIZE
                n_requests += 1
                prev_len = len(running_ids)
        # Probe anchor once.
        cached, wall = issue(anchor_ids)
        pct = 100 * cached / anchor_len
        row = {
            "K": K,
            "anchor_cached": cached,
            "anchor_len": anchor_len,
            "anchor_pct": pct,
            "cum_new_tokens": cum_new,
            "cum_new_blocks_lb": cum_blocks_lb,
            "n_burst_requests": n_requests,
            "elapsed_s": time.monotonic() - t_start,
        }
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        results.append(row)
        print(
            f"  → anchor probe: {cached}/{anchor_len} ({pct:.1f}%)  "
            f"cum_new={cum_new:,}  cum_blocks(lb)={cum_blocks_lb}"
        )

    fout.close()

    # --------------------- summary --------------------- #
    print("\n" + "=" * 70)
    print("L1 PRESSURE CURVE — anchor cached % vs cold-burst session count")
    print("=" * 70)
    print(f"  {'K':>3}  {'cum_new':>10}  {'cum_blocks(lb)':>14}  "
          f"{'anchor_cached':>13}  {'anchor_pct':>10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['K']:>3}  {r['cum_new_tokens']:>10,}  "
              f"{r['cum_new_blocks_lb']:>14}  "
              f"{r['anchor_cached']:>10}/{r['anchor_len']}  "
              f"{r['anchor_pct']:>9.1f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
