"""Phase 1 (real test) — kernel-block-size invariance.

The design's premise is that the attention kernel reads the SAME KV correctly
regardless of how the 1056-page is sub-divided into kernel blocks. We test it
falsifiably: force kernel_block_size to two legal values (16 and 32 — both in
flash-attn's supported set for this fp32-SSM hybrid, and both factors of 1056)
and assert the greedy outputs + first-token logprobs are bit-identical.

If 16 and 32 diverge, the kernel is NOT granularity-transparent and the
"no kernel change" premise is FALSE.

Run once per ksize (separate processes; one LLM per process):
  PATH=$PWD/.venv/bin:$PATH CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    .venv/bin/python probe_invariance.py --ksize 16
  ... --ksize 32
then: .venv/bin/python probe_invariance.py --compare
"""
from __future__ import annotations

import argparse
import json
import os

OUT = "dev/interlayer/0_feasibility/virtual_split/runs"
MODEL = "Qwen/Qwen3.5-35B-A3B"
PROMPTS = [
    "Explain in one sentence why the sky is blue.",
    "List the first 10 prime numbers.",
    "Write a short paragraph about the history of GPUs and why they matter for ML.",
    "Translate to French: The quick brown fox jumps over the lazy dog. Then explain the idiom.",
]


def run(ksize: int) -> None:
    import vllm.v1.worker.utils as wu

    _orig = wu.select_common_block_size

    def _force(kv_manager_block_size, backends):
        # force our target kernel block size when this is the 1056 attn group;
        # fall back to the real selection otherwise.
        if kv_manager_block_size % ksize == 0:
            return ksize
        return _orig(kv_manager_block_size, backends)

    wu.select_common_block_size = _force

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=128, logprobs=1)
    outs = llm.generate(PROMPTS, sp)
    rec = {"ksize": ksize, "per_prompt": []}
    for o in outs:
        c = o.outputs[0]
        # first-token top logprob (value) for a tighter-than-tokens check
        first_lp = None
        if c.logprobs and c.logprobs[0]:
            first_lp = max(v.logprob for v in c.logprobs[0].values())
        rec["per_prompt"].append(
            {"token_ids": list(c.token_ids), "n": len(c.token_ids),
             "first_logprob": first_lp}
        )
    with open(f"{OUT}/inv_{ksize}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {OUT}/inv_{ksize}.json  (ksize={ksize})")


def compare() -> None:
    a = json.load(open(f"{OUT}/inv_16.json"))
    b = json.load(open(f"{OUT}/inv_32.json"))
    ok = True
    details = []
    for i, (pa, pb) in enumerate(zip(a["per_prompt"], b["per_prompt"])):
        toks_eq = pa["token_ids"] == pb["token_ids"]
        lp_eq = (pa["first_logprob"] is None or pb["first_logprob"] is None
                 or abs(pa["first_logprob"] - pb["first_logprob"]) == 0.0)
        ok = ok and toks_eq
        details.append({"prompt": i, "n16": pa["n"], "n32": pb["n"],
                        "tokens_bit_identical": toks_eq,
                        "first_logprob_equal": lp_eq})
    verdict = {"kernel_block_size_invariant_16_vs_32": ok, "per_prompt": details}
    with open(f"{OUT}/inv_compare.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    else:
        run(args.ksize)
