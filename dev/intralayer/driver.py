"""verify/2 — Songyang SWE-bench W1 regression repro.

Mocks 16 concurrent multi-turn conversations on Qwen3-8B with growing
context per turn (1k tokens of synthetic "tool history" appended per
turn). Measures per-turn TTFT distribution and prefix-cache hit % so we
can compare HiMA layers against LRU on the same workload shape that
showed +193% p95 TTFT and -47pp hit rate on Songyang's actual W1.

config matrix:

    --config lru          # no HiMA
    --config l1_only      # VLLM_HIMA_L1_ENABLE=1
    --config l2_only      # VLLM_HIMA_L2_ENABLE=1
    --config full         # L1 + L2

Window pinned to `VLLM_HIMA_HPB_WINDOW_S=3600` per verify/5 finding.

Usage:
    .venv/bin/python -u driver.py --config <name> --turns 64 \\
        [--n-clients 16] [--out runs/<filename>.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
# Disable Intel OpenMP CPU pinning before any torch import.
os.environ.setdefault("KMP_AFFINITY", "disabled")
# Pin path-counter window per verify/5 conclusion.
os.environ.setdefault("VLLM_HIMA_HPB_WINDOW_S", "3600")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")


MODEL = "Qwen/Qwen3-8B"  # default; --model overrides for hybrid repro
# Per turn: ~1 KB tool result + ~1 KB assistant response = ~2 KB context growth
TOOL_RESULT_TOKENS = 1024
N_CLIENTS_DEFAULT = 16
# Hybrid models (Qwen3.5-*) need mamba_cache_mode set so the engine
# resolves a per-pool block_size for the mamba state cache. The
# attention groups still benefit from prefix-caching identically.
_HYBRID_MODEL_PREFIXES = ("Qwen/Qwen3.5-",)


_CONFIG_ENV = {
    "lru": {},
    "l1_only": {"VLLM_HIMA_L1_ENABLE": "1"},
    "l2_only": {"VLLM_HIMA_L2_ENABLE": "1"},
    "full": {"VLLM_HIMA_L1_ENABLE": "1", "VLLM_HIMA_L2_ENABLE": "1"},
}


def apply_env(config: str) -> None:
    """Apply config-specific env vars before vLLM import."""
    for k in ("VLLM_HIMA_L1_ENABLE", "VLLM_HIMA_L2_ENABLE"):
        os.environ.pop(k, None)
    for k, v in _CONFIG_ENV[config].items():
        os.environ[k] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(_CONFIG_ENV))
    ap.add_argument("--turns", type=int, required=True, help="rounds per client")
    ap.add_argument(
        "--n-clients", type=int, default=N_CLIENTS_DEFAULT,
        help="concurrent conversation count (Songyang W1: 16)",
    )
    ap.add_argument(
        "--util", type=float, default=0.55,
        help="gpu_memory_utilization. Songyang W1 ran 0.55.",
    )
    ap.add_argument(
        "--tp", type=int, default=2, help="tensor_parallel_size",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="per-turn JSONL path; defaults to runs/turns<N>_<config>_win3600.jsonl",
    )
    ap.add_argument(
        "--model", default=MODEL,
        help="HF model id. Default Qwen3-8B (single-group). For hybrid W1 "
        "repro, pass Qwen/Qwen3.5-35B-A3B.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for deterministic synthetic tool-result content",
    )
    args = ap.parse_args()

    apply_env(args.config)

    out_path = args.out
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "runs" / (
            f"turns{args.turns}_{args.config}_win3600.jsonl"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Import vLLM AFTER env is set so config is picked up at runtime
    # construction (per Phase 2 smoke convention).
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    # Each client gets a unique system prefix so the 16 conversations don't
    # share a common prompt prefix — that's the shape SWE-bench has too
    # (each task gets its own context).
    rng = random.Random(args.seed)
    n = args.n_clients

    # Vocabulary range: stay clear of special tokens and roughly within
    # Qwen3-8B's text-token zone (vocab ~152k for Qwen3). 10..50000 is safe.
    def random_token_block(n_tokens: int, seed: int) -> list[int]:
        r = random.Random(seed)
        return [r.randint(10, 50_000) for _ in range(n_tokens)]

    # Per-client conversation state: histories[i] is a list of token ids.
    # Initial system prompt is ~256 tokens of client-specific random
    # content + a fixed ~256-token shared prelude (mocks the SWE-bench
    # system/task framing that all 16 clients share).
    shared_prelude = random_token_block(256, args.seed)
    histories: list[list[int]] = []
    for client_idx in range(n):
        client_seed = args.seed + 100_000 + client_idx
        per_client_intro = random_token_block(256, client_seed)
        histories.append(shared_prelude + per_client_intro)

    # Boot the engine.
    is_hybrid = any(args.model.startswith(p) for p in _HYBRID_MODEL_PREFIXES)
    llm_kwargs: dict = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=args.util,
        max_num_seqs=64,
        max_model_len=args.turns * (TOOL_RESULT_TOKENS + 128) + 4096,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if is_hybrid:
        # Mamba groups need an explicit cache mode; "align" is what the
        # other intralayer drivers use and matches Songyang's W1 setup.
        llm_kwargs["mamba_cache_mode"] = "align"
    print(
        f"[config={args.config}] loading {args.model} TP={args.tp} "
        f"util={args.util} hybrid={is_hybrid}",
        flush=True,
    )
    llm = LLM(**llm_kwargs)
    print(f"[config={args.config}] engine ready", flush=True)

    # Write meta line first.
    fout = out_path.open("w")
    fout.write(json.dumps({
        "kind": "meta",
        "config": args.config,
        "turns": args.turns,
        "n_clients": n,
        "util": args.util,
        "tp": args.tp,
        "model": args.model,
        "window_s": int(os.environ.get("VLLM_HIMA_HPB_WINDOW_S", "0")),
        "tool_result_tokens": TOOL_RESULT_TOKENS,
        "env": {k: os.environ.get(k, "") for k in (
            "VLLM_HIMA_L1_ENABLE", "VLLM_HIMA_L2_ENABLE",
            "VLLM_HIMA_HPB_WINDOW_S",
        )},
    }) + "\n")
    fout.flush()

    t_run_start = time.monotonic()
    for turn in range(args.turns):
        # All 16 clients submit their next-turn prompt at the same time.
        prompts = list(histories)
        prompt_lens = [len(p) for p in prompts]
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        t0 = time.monotonic()
        outs = llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        batch_wall = time.monotonic() - t0

        ttfts: list[float] = []
        hits_pct: list[float] = []
        for i, out in enumerate(outs):
            cached = out.num_cached_tokens or 0
            hit_pct = cached / prompt_lens[i] * 100 if prompt_lens[i] else 0.0
            hits_pct.append(hit_pct)
            # Per-request first_token_latency is in vLLM v1's
            # RequestStateStats; fall back to batch_wall when unavailable.
            ttft_s = batch_wall
            metrics = getattr(out, "metrics", None)
            if metrics is not None:
                ftl = getattr(metrics, "first_token_latency", None)
                if ftl is not None and ftl > 0:
                    ttft_s = ftl
            ttfts.append(ttft_s)

        ttft_ms = [t * 1000 for t in ttfts]
        ttft_sorted = sorted(ttft_ms)
        p95_idx = max(0, int(len(ttft_sorted) * 0.95) - 1)

        turn_row = {
            "kind": "turn",
            "turn": turn,
            "batch_wall_ms": batch_wall * 1000,
            "mean_prompt_len": statistics.mean(prompt_lens),
            "mean_hit_pct": statistics.mean(hits_pct),
            "median_hit_pct": statistics.median(hits_pct),
            "min_hit_pct": min(hits_pct),
            "max_hit_pct": max(hits_pct),
            "mean_ttft_ms": statistics.mean(ttft_ms),
            "median_ttft_ms": statistics.median(ttft_ms),
            "p95_ttft_ms": ttft_sorted[p95_idx],
            "max_ttft_ms": max(ttft_ms),
            "elapsed_s": time.monotonic() - t_run_start,
        }
        fout.write(json.dumps(turn_row) + "\n")
        fout.flush()
        if turn % 8 == 0 or turn == args.turns - 1:
            print(
                f"  turn[{turn:>3}] batch_wall={batch_wall * 1000:.0f}ms  "
                f"hit%={turn_row['mean_hit_pct']:.1f}  "
                f"p95_ttft={turn_row['p95_ttft_ms']:.0f}ms  "
                f"mean_prompt_len={turn_row['mean_prompt_len']:.0f}",
                flush=True,
            )

        # Append the model's 1-token response + 1024 random tool-result tokens
        # to each client's history. Deterministic per (turn, client).
        for client_idx in range(n):
            resp_token_ids = list(outs[client_idx].outputs[0].token_ids)
            tool_seed = args.seed + 200_000 + client_idx * 10_000 + turn
            tool_chunk = random_token_block(TOOL_RESULT_TOKENS, tool_seed)
            histories[client_idx] = (
                histories[client_idx] + resp_token_ids + tool_chunk
            )

    # Summary line.
    fout.write(json.dumps({
        "kind": "summary",
        "total_wall_s": time.monotonic() - t_run_start,
    }) + "\n")
    fout.close()
    print(f"DONE wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
