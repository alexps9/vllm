# SPDX-License-Identifier: Apache-2.0
"""Microbench: partial-block bubble on a NON-HYBRID model with a forced
moderate block_size.

The hybrid version (02) measured the bubble at block_size=1056 on
Qwen3.5-35B-A3B, but fixing the bubble there requires per-group
num_computed_tokens surgery (Finding M.2). A non-hybrid full-attention
model has only ONE KV cache group, so the partial-cache fix is much
smaller — no per-group plumbing. The bubble per turn is bounded by
`block_size - 1` and the win is proportional. We force a larger-than-
default block_size to get a measurable bubble.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/05_nonhybrid_microbench.py
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
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-8B"           # full attention, no mamba
BLOCK_SIZE = 256                  # forced — larger than the typical 16
K_FULL_BLOCKS = 2                 # base prompt has K * 256 = 512 tokens
EXTENSION = 16
R_VALUES = [0, 16, 64, 128, 200, 255]  # partial tail lengths to sweep


def main() -> None:
    out_dir = Path("dev/interlayer/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "05_nonhybrid_micro.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"Loading {MODEL} (TP=1, util=0.5, block_size={BLOCK_SIZE}) …")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_model_len=8192,
        gpu_memory_utilization=0.5,
        max_num_seqs=4,
        trust_remote_code=True,
    )

    filler_text = "The quick brown fox jumps over the lazy dog. " * 8000
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)

    def issue(token_ids: list[int], max_tokens: int) -> dict:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        t0 = time.monotonic()
        outs = llm.generate(prompts=[token_ids], sampling_params=sp, use_tqdm=False)
        wall = time.monotonic() - t0
        out = outs[0]
        return {
            "wall_s": wall,
            "cached": out.num_cached_tokens or 0,
            "prompt_len": len(token_ids),
        }

    base_len = K_FULL_BLOCKS * BLOCK_SIZE  # 512
    print(f"\nSweeping R ∈ {R_VALUES} with K={K_FULL_BLOCKS} full blocks "
          f"(base_len={base_len}, extension={EXTENSION}). "
          f"VLLM_PARTIAL_CACHE_ENABLED="
          f"{os.environ.get('VLLM_PARTIAL_CACHE_ENABLED', '0')}")

    for trial_idx, R in enumerate(R_VALUES):
        offset = trial_idx * 4096
        turn1_len = base_len + R
        turn1_ids = filler_ids[offset:offset + turn1_len]
        turn2_ids = filler_ids[offset:offset + turn1_len + EXTENSION]

        r1 = issue(turn1_ids, max_tokens=1)
        r2 = issue(turn2_ids, max_tokens=1)

        log(
            kind="row",
            R=R,
            turn1_len=turn1_len,
            turn2_len=turn1_len + EXTENSION,
            turn1_cached=r1["cached"],
            turn1_wall_s=r1["wall_s"],
            turn2_cached=r2["cached"],
            turn2_wall_s=r2["wall_s"],
            turn2_uncached=turn1_len + EXTENSION - r2["cached"],
        )
        eff = 100 * r2["cached"] / (turn1_len + EXTENSION)
        print(f"  R={R:>4}  turn1[len={turn1_len}, wall={r1['wall_s']*1000:>5.0f}ms, "
              f"cached={r1['cached']:>4}]  "
              f"turn2[len={turn1_len+EXTENSION}, wall={r2['wall_s']*1000:>5.0f}ms, "
              f"cached={r2['cached']:>4} ({eff:.1f}%), "
              f"uncached={turn1_len+EXTENSION-r2['cached']}]")

    fout.close()
    print(f"\nWrote {out_jsonl}")


if __name__ == "__main__":
    main()
