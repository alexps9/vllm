# SPDX-License-Identifier: Apache-2.0
"""Convert the partial-block bubble's wasted tokens into wall-clock seconds.

Finding D measured ~4.71M wasted tokens across 106 cc sessions. This
script measures the actual prefill throughput on Qwen3.5-35B-A3B + H200
+ TP=2, then computes the total wall-clock TTFT cost of the bubble.

Approach:
  1) Spin up vLLM with the same config used everywhere else.
  2) Issue requests at lengths {512, 1k, 2k, 4k, 8k, 16k, 32k} (purely
     fresh, no cache hits — use unique prefixes per request).
  3) Measure wall-clock per request to get prefill latency vs length.
  4) Fit a per-token cost model (linear-ish: prefill_latency_us ≈ a + b·L).
  5) Apply to the 4.71M wasted tokens, broken down by length bucket from
     dev/e2e_replay.jsonl, to compute total TTFT cost.

Run:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python -u dev/ttft_cost_of_bubble.py | tee dev/ttft_cost_of_bubble.out
"""

from __future__ import annotations

import gc
import json
import os
import statistics
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
PROMPT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
N_TRIALS_PER_LEN = 3
OUT = Path("dev/ttft_cost_of_bubble.json")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Build a large filler token pool of distinct content.
    base = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. "
    all_ids = tok.encode(base * 5000, add_special_tokens=False)
    assert len(all_ids) >= 50000, f"filler too short ({len(all_ids)})"

    print(f"Loading {MODEL} (TP=2, util=0.45, mamba_cache_mode=align)...")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=33_000,
        gpu_memory_utilization=0.45,
        max_num_seqs=64,
        trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup
    print("Warmup...")
    for _ in range(3):
        llm.generate(prompts=[all_ids[:512]], sampling_params=sp, use_tqdm=False)

    # Measure prefill latency for each target length.
    results: dict[int, dict] = {}
    nonce_base = int(time.time()) & 0xFFFF
    nonce_offset = 0

    for L in PROMPT_LENGTHS:
        trials = []
        for trial in range(N_TRIALS_PER_LEN):
            # Make each prompt unique to avoid any prefix-cache hits.
            nonce = f" [trial-{nonce_base + nonce_offset:05d}] "
            nonce_offset += 1
            nonce_ids = tok.encode(nonce, add_special_tokens=False)
            # Build prompt of EXACTLY L tokens.
            body_len = L - len(nonce_ids)
            ids = nonce_ids + all_ids[trial * L: trial * L + body_len]
            assert len(ids) == L, f"bad len: {len(ids)} vs {L}"

            t0 = time.monotonic()
            outs = llm.generate(prompts=[ids], sampling_params=sp, use_tqdm=False)
            wall = time.monotonic() - t0
            cached = outs[0].num_cached_tokens or 0
            trials.append({"trial": trial, "wall_s": wall, "cached_tokens": cached})
            print(f"  L={L:>5} trial {trial}: {wall*1000:.0f} ms, cached={cached}")

        # Use median of trials to reject any one-shot JIT spike.
        med = statistics.median(t["wall_s"] for t in trials)
        results[L] = {
            "prompt_len": L,
            "trials": trials,
            "median_wall_s": med,
            "tps": L / med if med > 0 else 0,
        }
        print(f"  → L={L:>5}: median {med * 1000:.0f} ms, tps≈{L / med:,.0f}")

    # Linear fit: wall = a + b * L
    n = len(results)
    xs = [L for L in results]
    ys = [results[L]["median_wall_s"] for L in xs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(xs, ys))
    den = sum((xi - mean_x) ** 2 for xi in xs)
    b = num / den
    a = mean_y - b * mean_x
    print(f"\nLinear fit: prefill_wall_s ≈ {a*1000:.1f} ms + L × {b*1e6:.2f} µs/token")

    # ---- Apply to bubble waste ----
    # Per-session real-traffic waste from real_session_waste.py output:
    # total waste = 4,713,841 tokens across all 106 sessions.
    # Better: read e2e_replay.jsonl turn data for length-bucketed waste.
    replay_path = Path("dev/e2e_replay.jsonl")
    if replay_path.exists():
        turns = [
            json.loads(l) for l in replay_path.read_text().splitlines()
            if l.strip() and json.loads(l).get("kind") == "session_turn"
        ]
        # Each turn's waste is at the previous turn's prompt length (the
        # last block of the previous turn isn't cached, so the difference
        # is re-prefilled at the START of this turn's prompt -- which
        # involves a prefill of (waste + new_content) tokens).
        # Simpler model: waste tokens become extra prefill work that the
        # engine has to do at total prompt length (cached + waste + new).
        # Per-token marginal cost: b (slope) seconds/token.
        total_waste_tokens = sum(t.get("partial_block_waste", 0) for t in turns)
        per_token_marginal = b  # seconds per extra prefill token
        ttft_cost_s = total_waste_tokens * per_token_marginal
        print()
        print(f"From dev/e2e_replay.jsonl ({len(turns)} session_turn rows):")
        print(f"  total partial-block waste = {total_waste_tokens:,} tokens")
        print(f"  prefill marginal cost     = {per_token_marginal*1e6:.2f} µs/token")
        print(f"  → TTFT cost of bubble     = {ttft_cost_s:.1f} seconds total")
        print(f"  ÷ {len(turns)} requests     = {ttft_cost_s/len(turns)*1000:.1f} ms per request avg")

        # Scale to FULL 106-session corpus from Finding D (4,713,841 tokens):
        FULL_WASTE = 4_713_841
        full_cost_s = FULL_WASTE * per_token_marginal
        print()
        print(f"Scaling to FULL 106-session cc corpus (Finding D):")
        print(f"  total waste = {FULL_WASTE:,} tokens")
        print(f"  → TTFT cost = {full_cost_s:.1f} seconds = {full_cost_s/60:.1f} minutes")

    OUT.write_text(json.dumps({
        "prefill_latency": results,
        "fit": {"intercept_s": a, "slope_s_per_token": b},
    }, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
