# SPDX-License-Identifier: Apache-2.0
"""Multi-turn agent: validate per-turn partial-block waste on Qwen3.5-35B-A3B.

Claim being tested:
  * After each turn ends, vLLM commits ``floor(L / 1056) × 1056`` tokens
    to the prefix cache (the last partial block is freed without being
    hashed).
  * The next turn must re-prefill those last 0..1055 tokens.

For each scenario:
  Issue a fresh request at every turn (so cache hit must come from prior
  turns' commits — not from kept-alive KV). Record ``num_cached_tokens``
  on each turn and compare against the partial-block-aware prediction.

Reproduction:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python dev/multi_turn_waste.py | tee dev/multi_turn_waste.out
"""

from __future__ import annotations

import gc
import os
import sys

# Expose venv's binaries (ninja) to worker procs.
_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
# INFO logs so we capture "Setting attention block size to 1056 tokens"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 1056  # observed inflate value (see dev/trace_inflate.py)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Build a single long deterministic token sequence; prefix-slicing it
    # guarantees that any (turn_N+1) prompt is a *strict extension* of
    # turn_N's prompt at the token level — no tokenizer reshuffle.
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "She sells seashells down by the seashore. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    all_ids = tokenizer.encode(base * 6000, add_special_tokens=False)
    if len(all_ids) < 200000:
        raise RuntimeError(f"filler too short ({len(all_ids)} tokens)")

    print(f"Loading {MODEL} (TP=2, mamba_cache_mode=align, max_model_len=200000)"
          "... a few min.")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=200000,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def issue(token_ids: list[int]) -> int:
        # New vLLM API: a list[int] passed as the prompt is interpreted as
        # token ids; wrapping as a single-element list with the outer list
        # makes it a batch of size 1.
        outs = llm.generate(
            prompts=[token_ids], sampling_params=sp, use_tqdm=False,
        )
        return outs[0].num_cached_tokens or 0

    # --------------------- experiment scenarios ----------------------- #

    scenarios = [
        # (name, base_len, per_turn_increment, num_turns)
        ("micro_turn  (50 tok/turn)",    1500,    50, 20),
        ("short_turn  (150 tok/turn)",   2000,   150, 20),
        ("medium_turn (500 tok/turn)",   3000,   500, 20),
        ("long_turn   (3000 tok/turn)",  2000,  3000, 20),
    ]

    summary_rows = []
    for name, base_len, inc, n_turns in scenarios:
        # Each scenario uses a *unique* prefix (its own [length-XXX] header)
        # so cache state is fresh — different scenarios don't share blocks.
        header = (
            f"[scenario-{name.replace(' ', '_').replace('(', '').replace(')', '')}] "
        )
        # Use a small unique nonce that's at least 1 token, but doesn't shift
        # downstream tokenization much.
        nonce_ids = tokenizer.encode(header, add_special_tokens=False)
        # Shifted base for this scenario:
        my_ids = nonce_ids + all_ids
        assert len(my_ids) >= base_len + inc * (n_turns - 1)

        print()
        print("=" * 78)
        print(f"Scenario: {name}")
        print(f"  base_len = {base_len}, per-turn increment = {inc}, "
              f"turns = {n_turns}")
        print("=" * 78)
        header_cols = (
            "turn", "prompt_len", "cached", "expected", "diff",
            "new_tokens_to_prefill", "partial_block_waste",
        )
        print("  {:>4} | {:>10} | {:>7} | {:>8} | {:>5} | {:>20} | {:>12}".format(
            *header_cols
        ))
        print("  " + "-" * 78)

        prev_len = 0
        scenario_total_waste = 0
        for turn in range(n_turns):
            cur_len = base_len + turn * inc
            ids = my_ids[:cur_len]
            cached = issue(ids)

            # Expected: floor((prev_len - 1) / BLOCK_SIZE) * BLOCK_SIZE
            # The -1 accounts for the fact that the last token of the prior
            # request is the one being sampled, so its block is still open.
            if prev_len <= 0:
                expected = 0
            else:
                expected = ((prev_len - 1) // BLOCK_SIZE) * BLOCK_SIZE
            diff = cached - expected
            new_to_prefill = cur_len - cached
            # Of those, how many "should" have been cached but weren't
            # (= prior turn's partial last block):
            ideal_cached = prev_len if turn > 0 else 0
            partial_block_waste = max(ideal_cached - cached, 0)
            scenario_total_waste += partial_block_waste

            print("  {:>4} | {:>10} | {:>7} | {:>8} | {:>+5} | {:>20} | {:>12}".format(
                turn, cur_len, cached, expected, diff, new_to_prefill,
                partial_block_waste,
            ))
            prev_len = cur_len

        # Totals
        gross_new_input = base_len + (n_turns - 1) * inc  # ignoring turn 0
        avg_waste_per_turn = scenario_total_waste / max(n_turns - 1, 1)
        # Fraction of "new prefill that was actually waste from prior turn"
        new_tokens_added_over_run = (n_turns - 1) * inc
        waste_frac_of_new = (
            (scenario_total_waste / new_tokens_added_over_run) * 100
            if new_tokens_added_over_run > 0
            else 0.0
        )
        print(f"\n  total partial-block waste (turns 1..N): "
              f"{scenario_total_waste}")
        print(f"  avg waste / turn (after turn 0):        "
              f"{avg_waste_per_turn:.1f} tokens")
        print(f"  waste / new tokens added across run:    "
              f"{waste_frac_of_new:.1f}%")

        summary_rows.append((
            name, base_len, inc, n_turns, scenario_total_waste,
            avg_waste_per_turn, waste_frac_of_new,
        ))

    # ------------------------------ summary ----------------------------- #
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  {:<32} | {:>10} | {:>10} | {:>13}".format(
        "scenario", "avg waste", "waste/run", "waste/new_tok",
    ))
    print("  " + "-" * 78)
    for name, _, _, _, _, avg_w, frac in summary_rows:
        print(f"  {name:<32} | {avg_w:>10.1f} | {'':>10} | {frac:>12.1f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
