"""Phase 2 smoke driver: boot a real vLLM engine under one L1/L2 config.

Since vLLM v1 spawns ``EngineCore`` as a subprocess, the HiMA runtime
singleton lives there — ``get_runtime()`` in this process is always None.
Verification therefore happens via log-marker grep against the captured
EngineCore stderr; that's done by ``smoke_l1l2_verify.py`` on the .out
file this script's caller produced.

Usage (caller captures stderr):
    .venv/bin/python -u smoke_l1l2.py --config l1_only \\
        2>&1 | tee runs/smoke_l1_only.out
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

MODEL = "Qwen/Qwen3-0.6B"


def _set_env(config: str) -> None:
    for k in ("VLLM_HIMA_L1_ENABLE", "VLLM_HIMA_L2_ENABLE"):
        os.environ.pop(k, None)
    if config == "lru":
        return
    if config == "l1_only":
        os.environ["VLLM_HIMA_L1_ENABLE"] = "1"
    elif config == "l2_only":
        os.environ["VLLM_HIMA_L2_ENABLE"] = "1"
    elif config == "full":
        os.environ["VLLM_HIMA_L1_ENABLE"] = "1"
        os.environ["VLLM_HIMA_L2_ENABLE"] = "1"
    else:
        raise ValueError(f"unknown config: {config}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        choices=("lru", "l1_only", "l2_only", "full"),
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.30)
    args = parser.parse_args()

    _set_env(args.config)

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    print(f"SMOKE_CONFIG_TAG {args.config}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        max_model_len=512,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        enforce_eager=True,
    )
    boot_s = time.time() - t0
    print(f"SMOKE_BOOT_DONE config={args.config} boot_s={boot_s:.2f}", flush=True)

    # One short prompt to exercise scheduler hot path (so we'd see admitter
    # decisions log lines fire if admitter were called).
    out = llm.generate(["Hello, world."], SamplingParams(max_tokens=args.max_tokens))
    gen_ok = bool(out and out[0].outputs)
    print(f"SMOKE_GEN_DONE config={args.config} gen_ok={gen_ok}", flush=True)

    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
