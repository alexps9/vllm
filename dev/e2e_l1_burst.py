# SPDX-License-Identifier: Apache-2.0
"""Focused L1 anchor-eviction test on real cc traces — *no intermediate probes*.

The original ``dev/e2e_replay.py`` issues an anchor probe between every
session. That probe HITS the anchor blocks and pushes them back to the
tail of vLLM's free-block LRU queue — i.e., the measurement itself
prevents eviction. Result: anchor survival reported as 89.2% (i.e.
floor(4737/1056)*1056 / 4737 = "fully cached, modulo the partial-block
remainder") regardless of how much cold-burst traffic flowed through.

Here we remove the probe artifact:

  1) Warm anchor 5x.
  2) Probe BASELINE (one probe).
  3) Replay sessions 1..N IN FULL with **no anchor probes** in between.
  4) Probe FINAL (one probe).

The L1 claim predicts: with enough cold-burst pressure (>= KV_budget_blocks -
anchor_blocks new cached blocks), the anchor's blocks slip from tail to
head of the free queue and get evicted, so the final probe sees few
or zero cached anchor tokens — even though every preceding warm hit
made it "recently used".

Setup: same vLLM config as e2e_replay (TP=2, mamba_cache_mode=align,
util=0.35, max_num_seqs=64). The previous run measured KV budget at
~1022 blocks and the workload as ~1488 new blocks, so eviction is
*expected* without probes.

Run:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python -u dev/e2e_l1_burst.py | tee dev/e2e_l1_burst.out
"""

from __future__ import annotations

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
N_SESSIONS = 30  # incl. session 0 (used as anchor), so 29 cold-burst sessions
MAX_PROMPT_TOKENS = 60_000
N_ANCHOR_WARM = 5
OUT_JSONL = Path("dev/e2e_l1_burst.jsonl")


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sessions: list[list[dict]] = [
        json.loads(l)["messages"] for l in DATA.read_text().splitlines() if l.strip()
    ][:N_SESSIONS]

    # Anchor = session 0's first user message.
    first_user = next(m for m in sessions[0] if m["role"] == "user")
    anchor_ids = tokenizer.encode(
        msg_to_chunk(first_user), add_special_tokens=False
    )
    anchor_len = len(anchor_ids)
    print(f"Anchor: {anchor_len} tokens (~{anchor_len / BLOCK_SIZE:.1f} blocks)")

    print(f"Loading {MODEL} (TP=2, util=0.35, mamba_cache_mode=align)...")
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
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def issue(token_ids: list[int]) -> tuple[int, float]:
        t0 = time.monotonic()
        outs = llm.generate(prompts=[token_ids], sampling_params=sp, use_tqdm=False)
        return (outs[0].num_cached_tokens or 0, time.monotonic() - t0)

    OUT_JSONL.unlink(missing_ok=True)
    fout = OUT_JSONL.open("w")
    t_start = time.monotonic()

    # Phase A: warm anchor 5 times → its blocks settle at the tail of the
    # free queue with a fresh "most-recently used" stamp.
    print(f"\nPhase A: warming anchor with {N_ANCHOR_WARM} hits.")
    for i in range(N_ANCHOR_WARM):
        cached, wall = issue(anchor_ids)
        fout.write(json.dumps({
            "kind": "warm", "iter": i, "cached": cached, "wall_s": wall,
        }) + "\n")
        print(f"  warm[{i}] cached={cached}/{anchor_len}")
    fout.flush()

    # Baseline anchor probe.
    cached, wall = issue(anchor_ids)
    print(f"\nBASELINE anchor probe (post-warm): cached={cached}/{anchor_len} "
          f"({100 * cached / anchor_len:.1f}%)")
    fout.write(json.dumps({
        "kind": "probe", "label": "baseline",
        "after_session_idx": -1, "cached": cached, "wall_s": wall,
        "elapsed_s": time.monotonic() - t_start,
    }) + "\n")
    fout.flush()

    # Phase B: cold-burst workload — sessions 1..29, full content, no probes.
    print(f"\nPhase B: replaying sessions 1..{N_SESSIONS - 1} with NO probes "
          "in between (anchor cannot be refreshed).")
    cum_new_tokens = 0
    cum_new_blocks_lb = 0  # lower-bound on new cached blocks
    for s_idx in range(1, N_SESSIONS):
        msgs = sessions[s_idx]
        running_ids: list[int] = []
        prev_len = 0
        turn_count = 0
        for m_idx, m in enumerate(msgs):
            piece = tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
            running_ids.extend(piece)
            if m.get("role") != "assistant":
                continue
            if len(running_ids) > MAX_PROMPT_TOKENS:
                break
            ids = running_ids
            cached, wall = issue(ids)
            prompt_len = len(ids)
            new_content = prompt_len - prev_len
            partial_waste = max(prev_len - cached, 0) if prev_len > 0 else 0
            cum_new_tokens += new_content
            # Lower bound: at least floor(new_content / 1056) full blocks
            # got committed to cache. Some may be shared with prior session's
            # cached blocks (cache hit on prefix), but for distinct sessions
            # there's basically no overlap.
            cum_new_blocks_lb += new_content // BLOCK_SIZE
            fout.write(json.dumps({
                "kind": "session_turn",
                "session_idx": s_idx, "turn": turn_count,
                "prompt_len": prompt_len, "cached": cached,
                "new_content": new_content, "partial_waste": partial_waste,
                "wall_s": wall,
                "elapsed_s": time.monotonic() - t_start,
            }) + "\n")
            prev_len = prompt_len
            turn_count += 1
        fout.flush()
        print(f"  session {s_idx:>2}: {turn_count} turns, "
              f"final prompt_len={prev_len}, "
              f"cum_new_tokens={cum_new_tokens:,}, "
              f"cum_new_blocks(lb)={cum_new_blocks_lb}")

    # Phase C: FINAL anchor probe.
    cached, wall = issue(anchor_ids)
    pct = 100 * cached / anchor_len
    print(f"\nFINAL anchor probe (after {N_SESSIONS - 1} sessions of cold burst):")
    print(f"  anchor_cached = {cached}/{anchor_len} ({pct:.1f}%)")
    print(f"  cum_new_tokens during workload = {cum_new_tokens:,}")
    print(f"  cum_new_blocks (lower bound)   = {cum_new_blocks_lb}")
    print(f"  KV budget (blocks)             = 1022 (vLLM reported)")
    print()
    if cached >= (anchor_len // BLOCK_SIZE) * BLOCK_SIZE:
        print("VERDICT: anchor SURVIVED → L1 claim not reproduced "
              "(insufficient pressure for given KV budget).")
    elif cached > 0:
        print(f"VERDICT: anchor PARTIALLY EVICTED → L1 claim reproduced "
              f"(only {cached}/{anchor_len} survives).")
    else:
        print("VERDICT: anchor FULLY EVICTED → L1 claim reproduced "
              "(LRU dropped a high-value heavily-hit block under pressure).")

    fout.write(json.dumps({
        "kind": "probe", "label": "final",
        "after_session_idx": N_SESSIONS - 1, "cached": cached, "wall_s": wall,
        "cum_new_tokens": cum_new_tokens,
        "cum_new_blocks_lb": cum_new_blocks_lb,
        "elapsed_s": time.monotonic() - t_start,
    }) + "\n")
    fout.close()
    print(f"\nLog written to {OUT_JSONL}")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
