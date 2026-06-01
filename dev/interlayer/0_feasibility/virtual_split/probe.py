"""Phase 1 — virtual_split: confirm the attention kernel runs at
kernel_block_size << manager block_size (virtual block splitting is active),
and that generation is correct/deterministic at that layout.

This validates the design's central premise: "the kernel is already
fine-grained; the 1056 lives only in the allocator." If kernel_block_size
== manager_block_size, the premise is FALSE and the design needs rework.

TP=1 (single process) so the monkeypatch on select_common_block_size — which
runs in the model-runner — fires in-process and we can capture its real
inputs/outputs.

Run: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/1_virtual_split/probe.py
"""
from __future__ import annotations

import json
import os

OUT = "dev/interlayer/0_feasibility/virtual_split/runs"
os.makedirs(OUT, exist_ok=True)

# --- monkeypatch: capture (manager_block_size -> kernel_block_size) ---------
import vllm.v1.worker.utils as wu  # noqa: E402

_orig_select = wu.select_common_block_size
_captured: list[dict] = []


def _patched_select(kv_manager_block_size, backends):
    r = _orig_select(kv_manager_block_size, backends)
    _captured.append(
        {
            "manager_block_size": int(kv_manager_block_size),
            "kernel_block_size": int(r),
            "ratio": int(kv_manager_block_size) // int(r) if r else None,
            "backends": [getattr(b, "__name__", str(b)) for b in backends],
        }
    )
    return r


wu.select_common_block_size = _patched_select

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3.5-35B-A3B"


def main() -> None:
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=True,  # skip cudagraph capture; faster boot, phase 6 covers graphs
    )

    cfg = llm.llm_engine.vllm_config
    manager_bs = cfg.cache_config.block_size

    # determinism / correctness sanity: same prompt twice, greedy -> identical
    prompts = [
        "Explain in one sentence why the sky is blue.",
        "Write a haiku about GPUs.",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    out1 = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    out2 = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    deterministic = out1 == out2

    result = {
        "model": MODEL,
        "cache_config.block_size": manager_bs,
        "captured_select_common_block_size": _captured,
        "deterministic_greedy": deterministic,
    }
    # the headline judgement
    ksizes = {c["kernel_block_size"] for c in _captured}
    mbs = {c["manager_block_size"] for c in _captured}
    result["kernel_block_sizes_seen"] = sorted(ksizes)
    result["manager_block_sizes_seen"] = sorted(mbs)
    result["splitting_active"] = any(
        c["kernel_block_size"] < c["manager_block_size"] for c in _captured
    )

    with open(f"{OUT}/probe_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
