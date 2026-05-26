"""Phase E in isolation — for py-spy attach.

Boots the engine, prints its EngineCore PID, then loops doing
cold-prompt issue() calls forever (or until --n). Lets you attach
py-spy from outside while the steady state is running.

Usage:
    .venv/bin/python -u phase_e_only.py --mode l1_only --n 100
    # in another shell, run py-spy attach --pid <ENGINE_PID>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("VLLM_HIMA_HPB_WINDOW_S", "3600")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3.5-35B-A3B"
PROMPT_LEN = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lru", "l1_only"), required=True)
    ap.add_argument("--n", type=int, default=100, help="number of cold prompts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[mode={args.mode}] loading {MODEL} ...", flush=True)
    kwargs: dict = {}
    if args.mode == "l1_only":
        kwargs["hima_l1_enabled"] = True

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=PROMPT_LEN + 1024,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        trust_remote_code=True,
        **kwargs,
    )

    print("PHASE_E_READY", flush=True)
    # Sleep so the operator has time to attach py-spy.
    time.sleep(5)

    rng = random.Random(args.seed)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    t_total = 0.0
    for k in range(args.n):
        # Each prompt is unique random ints — no shared prefix.
        prompt = [rng.randint(10, 50_000) for _ in range(PROMPT_LEN)]
        t0 = time.monotonic()
        out = llm.generate(prompts=[prompt], sampling_params=sp, use_tqdm=False)
        dt = time.monotonic() - t0
        t_total += dt
        if k < 5 or k % 20 == 0 or k == args.n - 1:
            print(f"  prompt[{k:>3}] wall={dt * 1000:.0f}ms", flush=True)
    print(
        f"DONE mode={args.mode} n={args.n} mean={t_total / args.n * 1000:.0f}ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
