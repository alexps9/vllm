# SPDX-License-Identifier: Apache-2.0
"""Empirically verify prefix-cache hit granularity on Qwen3-Next-80B-A3B.

For each shared-prefix length L in TEST_LENGTHS:
  1) Build a deterministic prompt of exactly L tokens.
  2) Run pass 1: generate to prime the prefix cache.
  3) Run pass 2 (separate call): re-issue, read ``num_cached_tokens``.

If our analysis of ``lcm_block_size = 544`` is correct, we expect
  ``num_cached_tokens == floor((L - 1) / 544) * 544``.
(L-1 because the last token's KV is the one being generated.)

Run via: .venv/bin/python dev/hit_rate_microbench.py
"""

from __future__ import annotations

import gc
import os
import sys

# Workers spawn with a clean PATH; expose the venv's `ninja` so backend
# auto-compilation can find it.
_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Qwen3.5-35B-A3B is the model the paper used. It's a VL model in HF
# (Qwen3_5MoeForConditionalGeneration), but text-only prompts work fine.
MODEL = "Qwen/Qwen3.5-35B-A3B"

# Lengths to probe.
#   < 544          → predicted 0 cached
#   around boundaries (543, 544, 545, 1087, 1088)
#   multi-block    → predicted floor((L-1)/544)*544
TEST_LENGTHS = [100, 300, 500, 543, 544, 545, 800, 1087, 1088, 1500, 2000, 5000, 10000]

PREDICTED_LCM = 1056  # = 16 × cdiv(2146304, 16 × 1024); see dev/trace_inflate.py


def build_prompt_of_length(tokenizer, target_len: int) -> tuple[str, list[int]]:
    """Build a prompt that tokenizes to exactly ``target_len`` tokens."""
    # Use a long deterministic filler so different lengths have *different*
    # token prefixes (otherwise L=100 and L=300 would share a 100-token prefix
    # and produce a confounded cache state).
    base = (
        f"[length-{target_len:05d}] "
        "The quick brown fox jumps over the lazy dog. "
    )
    s = base * (target_len // 8 + 8)
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) < target_len:
        raise RuntimeError(f"filler too short: {len(ids)} < {target_len}")
    return tokenizer.decode(ids[:target_len]), ids[:target_len]


def main() -> None:
    # INFO so we capture "Setting attention block size to X tokens" and
    # "Padding mamba page size by X%" logged by _align_hybrid_block_size.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"Loading {MODEL} (TP=2, mamba_cache_mode=align)... takes a few min.")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",  # required for mamba prefix-cache hits
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    # Introspect engine to find the *actual* per-group block_size after KV cache setup.
    cfg = llm.llm_engine.vllm_config  # type: ignore[attr-defined]
    print(f"\nengine.cache_config.block_size       = {cfg.cache_config.block_size}")
    print(f"engine.cache_config.mamba_block_size = {cfg.cache_config.mamba_block_size}")
    print(f"engine.cache_config.mamba_page_size_padded = "
          f"{cfg.cache_config.mamba_page_size_padded}")
    print(f"engine.cache_config.mamba_cache_mode = {cfg.cache_config.mamba_cache_mode}")

    # NOTE: ``llm.collective_rpc("get_kv_cache_spec")`` would dump per-layer
    # specs, but the RPC encoder doesn't serialize ``torch.dtype`` and hangs
    # the engine. The authoritative ``block_size`` shows up in the engine's
    # INFO log line "Setting attention block size to N tokens" — see stderr.
    print()

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    prompts: dict[int, str] = {}
    for L in TEST_LENGTHS:
        prompts[L], ids = build_prompt_of_length(tokenizer, L)
        assert len(ids) == L

    # Pass 1: prime cache.
    print("Pass 1: priming prefix cache...")
    _ = llm.generate(list(prompts.values()), sp, use_tqdm=False)

    # Pass 2: measure cache hits one at a time.
    print("Pass 2: measuring num_cached_tokens on second issue:\n")
    print(f"  predicted lcm_block_size = {PREDICTED_LCM}")
    print(f"  formula: predicted_hit = floor((L - 1) / {PREDICTED_LCM}) "
          f"× {PREDICTED_LCM}\n")
    print(f"  {'L (tokens)':>10} | {'predicted':>10} | "
          f"{'actual':>10} | {'match?':>7}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*7}")

    results = []
    for L in TEST_LENGTHS:
        outs = llm.generate([prompts[L]], sp, use_tqdm=False)
        actual = outs[0].num_cached_tokens or 0
        predicted = (max(L - 1, 0) // PREDICTED_LCM) * PREDICTED_LCM
        ok = "yes" if actual == predicted else "NO"
        print(f"  {L:>10} | {predicted:>10} | {actual:>10} | {ok:>7}")
        results.append((L, predicted, actual))

    print("\nSummary:")
    misses = [r for r in results if r[1] != r[2]]
    if not misses:
        print(f"  All actuals match floor((L-1)/{PREDICTED_LCM}) × {PREDICTED_LCM}.")
    else:
        print(f"  {len(misses)} mismatch(es):")
        for L, p, a in misses:
            print(f"    L={L}: predicted={p}, actual={a}")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
