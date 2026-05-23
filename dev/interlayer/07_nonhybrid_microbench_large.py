# SPDX-License-Identifier: Apache-2.0
"""Scaled-up version of 05_nonhybrid_microbench.py.

Same idea (single-group full-attention, partial-cache prototype
validation) but with K=8 full blocks and block_size=1024 to mimic
the hybrid case's inflated block size (1056). Win should be much more
visible than the 20ms-walls of 05_.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u dev/interlayer/07_nonhybrid_microbench_large.py
  CUDA_VISIBLE_DEVICES=0 VLLM_PARTIAL_CACHE_ENABLED=1 .venv/bin/python -u dev/interlayer/07_nonhybrid_microbench_large.py
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

MODEL = "Qwen/Qwen3-8B"
BLOCK_SIZE = 1024                  # close to the hybrid inflated case
K_FULL_BLOCKS = 8                  # base prompt 8 * 1024 = 8192 tokens
EXTENSION = 16
R_VALUES = [0, 64, 256, 512, 800, 1000, 1023]


def main() -> None:
    out_dir = Path("dev/interlayer/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "07_nonhybrid_large.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"Loading {MODEL} (TP=1, util=0.6, block_size={BLOCK_SIZE}) …")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_model_len=16384,
        gpu_memory_utilization=0.6,
        max_num_seqs=4,
        trust_remote_code=True,
    )

    filler_text = "The quick brown fox jumps over the lazy dog. " * 30000
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

    base_len = K_FULL_BLOCKS * BLOCK_SIZE  # 8192
    print(f"\nSweeping R ∈ {R_VALUES} with K={K_FULL_BLOCKS} full blocks "
          f"(base_len={base_len}, extension={EXTENSION}). "
          f"VLLM_PARTIAL_CACHE_ENABLED="
          f"{os.environ.get('VLLM_PARTIAL_CACHE_ENABLED', '0')}")

    for trial_idx, R in enumerate(R_VALUES):
        # Use a different non-overlapping slice for each R
        offset = trial_idx * 16384
        turn1_len = base_len + R
        turn1_ids = filler_ids[offset:offset + turn1_len]
        turn2_ids = filler_ids[offset:offset + turn1_len + EXTENSION]

        # Run each pair 3 times — first is warmup, average of the last 2
        wts1, wts2, c1, c2 = [], [], None, None
        for rep in range(3):
            r1 = issue(turn1_ids, max_tokens=1)
            r2 = issue(turn2_ids, max_tokens=1)
            wts1.append(r1["wall_s"])
            wts2.append(r2["wall_s"])
            c1 = r1["cached"]
            c2 = r2["cached"]
        # Use last 2 reps (skip warmup) for wall time
        wall1 = sum(wts1[-2:]) / 2
        wall2 = sum(wts2[-2:]) / 2

        log(
            kind="row",
            R=R,
            turn1_len=turn1_len,
            turn2_len=turn1_len + EXTENSION,
            turn1_cached=c1,
            turn1_wall_s=wall1,
            turn2_cached=c2,
            turn2_wall_s=wall2,
            turn2_uncached=turn1_len + EXTENSION - c2,
            reps=3,
        )
        eff = 100 * c2 / (turn1_len + EXTENSION)
        print(f"  R={R:>4}  turn1[len={turn1_len}, wall={wall1*1000:>5.0f}ms, "
              f"cached={c1:>5}]  "
              f"turn2[len={turn1_len+EXTENSION}, wall={wall2*1000:>5.0f}ms, "
              f"cached={c2:>5} ({eff:.1f}%), "
              f"uncached={turn1_len+EXTENSION-c2}]")

    fout.close()
    print(f"\nWrote {out_jsonl}")


if __name__ == "__main__":
    main()
