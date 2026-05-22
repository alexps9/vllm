# SPDX-License-Identifier: Apache-2.0
"""LRU vs LPB end-to-end comparison on real cc workload.

Runs the SAME workload twice — once with the default LRU free-block queue
(`hima_enabled=False`) and once with HiMA L1's LPB-scored queue
(`hima_enabled=True`) — and records per-request metrics so we can compare:

  * L1 outcome: anchor cache survival after cold-burst pressure.
  * L2 outcome on cc traffic:
      - aggregate cache hit rate (Σnum_cached / Σprompt_len)
      - mean per-request wall time (with max_tokens=10)
      - first-token (TTFT) proxy (with max_tokens=1, separate pass)
      - per-output-token (TPOT) proxy = (wall_N20 - wall_N1) / 19
      - aggregate throughput = Σoutput_tokens / Σwall

Invocation:
  .venv/bin/python -u dev/compare_lru_lpb.py --mode lru | tee dev/compare_lru.out
  .venv/bin/python -u dev/compare_lru_lpb.py --mode lpb | tee dev/compare_lpb.out

The two runs must use separate Python processes because HiMA enables a
process-global runtime singleton.
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
# Intel OpenMP (which torch+vllm pull in via MKL) auto-pins the process to a
# single CPU based on GPU NUMA topology. On hosts with hot CPU contention
# this starves the python interpreter to ~0.2% CPU. Disable BEFORE any
# torch import.
os.environ.setdefault("KMP_AFFINITY", "disabled")
# Extend HiMA's path-counted-hit window beyond the default 60s — our run
# takes several minutes and we don't want anchor's hits to expire from the
# counter between warm and final probe.
os.environ.setdefault("VLLM_HIMA_HPB_WINDOW_S", "3600")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-35B-A3B"
# Bundled into dev/ so the experiment is self-contained on any host.
DATA = Path(__file__).resolve().parent / "cc_long_traces.jsonl"
BLOCK_SIZE = 1056
MAX_PROMPT_TOKENS = 60_000
N_ANCHOR_WARM = 500  # high enough that anchor's n_b dominates cc-session hits
                     # (each cc turn issues 2 requests → ~100 hits for top blocks
                     # of a 50-turn session; we want anchor 5x above that)
N_BURST_SESSIONS = 10  # cc sessions to replay as cold burst between warm & probe
N_TPOT_TOKENS = 20  # how many decode tokens for the throughput pass


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
    ap.add_argument("--mode", choices=["lru", "lpb"], required=True)
    args = ap.parse_args()
    mode = args.mode
    hima_on = mode == "lpb"

    out_jsonl = Path(f"dev/compare_{mode}.jsonl")
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sessions: list[list[dict]] = [
        json.loads(l)["messages"]
        for l in DATA.read_text().splitlines() if l.strip()
    ][:N_BURST_SESSIONS + 1]  # +1 for session 0 (anchor)

    # Anchor = session 0's first user message
    first_user = next(m for m in sessions[0] if m["role"] == "user")
    anchor_ids = tokenizer.encode(
        msg_to_chunk(first_user), add_special_tokens=False
    )
    anchor_len = len(anchor_ids)
    print(f"[{mode}] Anchor: {anchor_len} tokens "
          f"(~{anchor_len / BLOCK_SIZE:.1f} blocks)")

    print(f"[{mode}] Loading {MODEL} (TP=2, util=0.35, "
          f"hima_enabled={hima_on})...")
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
        hima_enabled=hima_on,
    )

    def issue(token_ids, max_tokens: int) -> dict:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
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

    # ----- Phase A: warm anchor N_ANCHOR_WARM times -----
    print(f"\n[{mode}] Phase A: warming anchor {N_ANCHOR_WARM}x ...")
    for i in range(N_ANCHOR_WARM):
        r = issue(anchor_ids, max_tokens=1)
        if i in (0, 1, N_ANCHOR_WARM // 2, N_ANCHOR_WARM - 1):
            print(f"  warm[{i:>3}] cached={r['cached']}/{anchor_len}  "
                  f"wall={r['wall_s']*1000:.0f}ms")
    log(kind="phase", phase="A_done", elapsed_s=time.monotonic() - t_start)

    # Baseline probe (anchor should be fully cached)
    r = issue(anchor_ids, max_tokens=1)
    print(f"\n[{mode}] BASELINE anchor probe: cached={r['cached']}/{anchor_len} "
          f"({100*r['cached']/anchor_len:.1f}%)")
    log(kind="anchor_probe", label="baseline",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    # ----- Phase B: cc cold burst with TTFT + TPOT passes per turn -----
    print(f"\n[{mode}] Phase B: cc burst over {N_BURST_SESSIONS} sessions; "
          f"each turn issued with max_tokens=1 (TTFT) then max_tokens={N_TPOT_TOKENS+1} "
          "(TPOT/throughput).")
    for s_idx in range(1, 1 + N_BURST_SESSIONS):
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
            # Pass 1: TTFT (max_tokens=1)
            r_t = issue(running_ids, max_tokens=1)
            # Pass 2: throughput (max_tokens=N_TPOT_TOKENS+1)
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
        print(f"  [{mode}] session {s_idx:>2}: {turn} turns done "
              f"(elapsed {time.monotonic() - t_start:.0f}s)")
    log(kind="phase", phase="B_done", elapsed_s=time.monotonic() - t_start)

    # ----- Phase C: final anchor probe (LRU vs LPB headline) -----
    r = issue(anchor_ids, max_tokens=1)
    pct = 100 * r["cached"] / anchor_len
    print(f"\n[{mode}] FINAL anchor probe: cached={r['cached']}/{anchor_len} "
          f"({pct:.1f}%)")
    log(kind="anchor_probe", label="final",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    # ----- Phase D: anchor-rehit workload (LPB BEST CASE) ----- #
    # Downstream traffic that *does* re-issue anchored prompts. Each request
    # = anchor + small unique tail. Under LPB, the anchor (now still cached)
    # gives near-full prefix hit; under LRU (anchor was evicted in Phase B/C)
    # every such request pays a fresh prefill. This is the headline win the
    # paper claims for shared system-prompt agent fleets.
    print(f"\n[{mode}] Phase D: anchor-rehit workload (30 anchored requests).")
    N_REHIT = 30
    for j in range(N_REHIT):
        # Append a unique 16-token tail so the prompt is anchor + small new
        # content (no inter-rehit caching of the tail).
        tail_text = f"\n<|im_start|>user\n[rehit-{j:03d}] continue\n<|im_end|>\n"
        tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
        prompt = anchor_ids + tail_ids
        r_t = issue(prompt, max_tokens=1)
        r_d = issue(prompt, max_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="rehit_turn", j=j,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if j in (0, 1, N_REHIT // 2, N_REHIT - 1):
            print(f"  rehit[{j:>2}] ttft_cached={r_t['cached']:>5} "
                  f"ttft_wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase E: no-shared-prefix cold flow (LPB WORST CASE) ----- #
    # Many short *unique* prompts that share NO prefix with anchor or with
    # each other. There's no anchor to protect; LPB has nothing useful to
    # do. This isolates the hot-path overhead of HiMA's path counter +
    # depth tracking + heap-based queue vs the simple LRU deque.
    print(f"\n[{mode}] Phase E: no-shared-prefix cold flow "
          "(50 unique 2k-token prompts).")
    N_COLD = 50
    PROMPT_LEN_COLD = 2048
    # Long filler so each prompt is distinct. Need N_COLD × PROMPT_LEN_COLD
    # = 50 × 2048 = ~102K tokens; multiply the base sentence enough times.
    filler_text = (
        "The quick brown fox jumps over the lazy dog. " * 30000
    )
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    assert len(filler_ids) >= N_COLD * PROMPT_LEN_COLD, (
        f"filler too short: {len(filler_ids)} < {N_COLD * PROMPT_LEN_COLD}"
    )
    for k in range(N_COLD):
        prompt = filler_ids[k * PROMPT_LEN_COLD: (k + 1) * PROMPT_LEN_COLD]
        # Each prompt is a unique slice → no shared prefix across cold queries
        r_t = issue(prompt, max_tokens=1)
        r_d = issue(prompt, max_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="cold_turn", k=k,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if k in (0, N_COLD // 2, N_COLD - 1):
            print(f"  cold[{k:>2}] cached={r_t['cached']:>3} "
                  f"wall={r_t['wall_s']*1000:.0f}ms")

    fout.close()
    print(f"\n[{mode}] Done. {time.monotonic() - t_start:.0f}s total. "
          f"Log: {out_jsonl}")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
