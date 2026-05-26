# SPDX-License-Identifier: Apache-2.0
"""Microbench: quantify the partial-block bubble's cost as a function of tail length R.

Issues a sequence of 2-turn dialogues where the first turn ends with K
full blocks + R tokens of a partial last block. The second turn re-issues
the same prefix + 16 fresh tokens. With block_size = 1056 (inflated on
Qwen3.5-35B-A3B hybrid) and vLLM's current behavior, the partial-block
content is never cached, so the second turn pays full prefill on (R + 16)
tokens regardless of R.

Goal: show TTFT(turn 2) growing linearly with R as ground-truth evidence
of the bubble. This is the BASELINE measurement for the partial-cache
prototype (file 03) — after the fix, TTFT(turn 2) should be flat in R.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u dev/interlayer/02_partial_cache_micro.py
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

import torch  # noqa: E402  (env vars must precede)
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 1056  # inflated by _align_hybrid_block_size; see dev/README.md Finding A
K_FULL_BLOCKS = 2  # turn 1 ends with K full blocks + R partial tokens
EXTENSION = 16     # turn 2 appends this many fresh tokens
R_VALUES = [0, 32, 96, 256, 512, 800, 1000, 1055]  # partial tail lengths to sweep


def main() -> None:
    out_dir = Path("dev/interlayer/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "02_partial_cache_micro.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"Loading {MODEL} (TP=2, util=0.35) …")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=8192,
        gpu_memory_utilization=0.35,
        max_num_seqs=4,
        trust_remote_code=True,
    )

    # Build a long filler the tokenizer will encode densely so we can take
    # disjoint slices of exact lengths.
    filler_text = "The quick brown fox jumps over the lazy dog. " * 6000
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

    base_len = K_FULL_BLOCKS * BLOCK_SIZE  # 2112
    print(f"\nSweeping R ∈ {R_VALUES} with K={K_FULL_BLOCKS} full blocks "
          f"(base_len={base_len}, extension={EXTENSION})")

    for trial_idx, R in enumerate(R_VALUES):
        # Each R uses a DIFFERENT slice of the filler so prefixes don't
        # accidentally collide across R values (which would let later R's
        # see earlier R's cache hits).
        offset = trial_idx * 4096
        turn1_len = base_len + R
        turn1_ids = filler_ids[offset:offset + turn1_len]
        turn2_ids = filler_ids[offset:offset + turn1_len + EXTENSION]

        # Turn 1: prime the cache with turn1_len tokens.
        r1 = issue(turn1_ids, max_tokens=1)
        # Turn 2: re-issue the same prefix + EXTENSION fresh tokens.
        r2 = issue(turn2_ids, max_tokens=1)

        # Expected cached without bubble fix: K * block_size = 2112
        # (the partial last R tokens of turn 1 are NOT in the prefix cache).
        # With bubble fix: K * block_size + R = 2112 + R.
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
