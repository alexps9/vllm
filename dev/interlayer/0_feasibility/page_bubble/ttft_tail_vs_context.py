# SPDX-License-Identifier: Apache-2.0
"""Verifiable downside of the big (1056) page at LONG context: the prefix-cache
recompute TAIL, and how its TTFT cost GROWS with context length.

At any context length, a re-issue's cache hit rounds DOWN to floor(L/1056)*1056,
so the last partial block (up to ~1055 tokens) is recomputed every turn. The
token COUNT is ~constant (~528 avg), but those tail tokens attend to the FULL L
cached context, so the recompute PREFILL COST grows with L. This is the harm
the "42.6%" actually pointed at — real, and (per 08_hybrid_architectural_blocker)
NOT fixable on hybrid by finer attention caching, because mamba's SSM state is
only cached at 1056 boundaries and forces recompute of the tail through the
whole stack.

Controlled measurement (isolates the tail): at each scale, compare a context
that is a MULTIPLE of 1056 (tail = 0; only the increment recomputed) vs the same
scale + 528 (tail = 528). Re-issue both; the TTFT delta = the tail-recompute
cost. Sweep scale to show it grows with context.

Run: CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python <this> 2>&1 | tee runs/ttft_tail.out
"""

from __future__ import annotations

import os
import sys
import time

_VENV_BIN = os.path.dirname(sys.executable)
os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import random  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3.5-35B-A3B"
B = 1056
MAX_LEN = 131072
INC = 8                  # small agent increment (new content this turn)


def main() -> None:
    print(f"Loading {MODEL} (TP=2, align, max_model_len={MAX_LEN})...")
    llm = LLM(model=MODEL, tensor_parallel_size=2, trust_remote_code=True,
              dtype="bfloat16", enable_prefix_caching=True, mamba_cache_mode="align",
              max_model_len=MAX_LEN, gpu_memory_utilization=0.85, disable_log_stats=True)
    rng = random.Random(0)
    vocab = 100000

    def ttft_reissue(ctx_len: int) -> tuple[float, int]:
        """Build a fresh ctx of ctx_len random tokens; prefill+finish (turn1,
        commits floor/1056 to cache); re-issue ctx+INC (turn2); return
        (turn2_wall_s, num_cached_on_turn2)."""
        ctx = [rng.randrange(vocab) for _ in range(ctx_len)]
        sp1 = SamplingParams(max_tokens=1, temperature=0)
        llm.generate([{"prompt_token_ids": ctx}], sp1, use_tqdm=False)  # turn1
        ext = ctx + [rng.randrange(vocab) for _ in range(INC)]
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ext}], sp1, use_tqdm=False)  # turn2
        dt = time.perf_counter() - t0
        nc = out[0].num_cached_tokens
        return dt, nc

    print(f"{'scale':>8} {'aligned_ms':>11} {'+528_ms':>9} {'tail_harm_ms':>13} "
          f"{'aligned_cached':>15} {'+528_cached':>12}")
    results = []
    for k in (8, 16, 32, 48, 64, 90):       # k blocks of 1056 -> ~8k..95k tokens
        base = k * B
        a_ms, a_nc = ttft_reissue(base)         # tail = 0
        t_ms, t_nc = ttft_reissue(base + 528)   # tail = 528
        harm = (t_ms - a_ms) * 1000
        results.append((base, a_ms * 1000, t_ms * 1000, harm, a_nc, t_nc))
        print(f"{base:>8} {a_ms*1000:>11.1f} {t_ms*1000:>9.1f} {harm:>13.1f} "
              f"{a_nc:>15} {t_nc:>12}")
    print("\nInterpretation: 'tail_harm_ms' = extra TTFT to recompute the 528-token")
    print("partial-block tail on each re-issue (what the big page costs vs a 0-tail).")
    print("If it grows with scale, the big-page recompute harm worsens with context.")
    print("NB (08_hybrid_architectural_blocker): on hybrid this tail is NOT")
    print("removable by finer attention caching — mamba forces recompute. So this")
    print("is a real big-page downside the interlayer sub-block fix does NOT capture.")


if __name__ == "__main__":
    main()
