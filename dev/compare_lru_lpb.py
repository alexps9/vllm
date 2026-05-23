# SPDX-License-Identifier: Apache-2.0
"""LRU vs LPB end-to-end comparison on real cc workload.

Runs the SAME workload twice — once with the default LRU free-block queue
(`hima_enabled=False`) and once with HiMA L1's LPB-scored queue
(`hima_enabled=True`) — and records per-request metrics so we can compare:

  * L1 outcome: anchor cache survival after cold-burst pressure.
  * L2 outcome on cc traffic:
      - aggregate cache hit rate (Σnum_cached / Σprompt_len)
      - mean per-request wall time (with max_tokens=10)
      - first-token (TTFT) proxy (with max_tokens=1, separate pass)
      - per-output-token (TPOT) proxy = (wall_N20 - wall_N1) / 19
      - aggregate throughput = Σoutput_tokens / Σwall

Invocation:
  .venv/bin/python -u dev/compare_lru_lpb.py --mode lru | tee dev/compare_lru.out
  .venv/bin/python -u dev/compare_lru_lpb.py --mode lpb | tee dev/compare_lpb.out

The two runs must use separate Python processes because HiMA enables a
process-global runtime singleton.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
# Intel OpenMP (which torch+vllm pull in via MKL) auto-pins the process to a
# single CPU based on GPU NUMA topology. On hosts with hot CPU contention
# this starves the python interpreter to ~0.2% CPU. Disable BEFORE any
# torch import.
os.environ.setdefault("KMP_AFFINITY", "disabled")
# Extend HiMA's path-counted-hit window beyond the default 60s — our run
# takes several minutes and we don't want anchor's hits to expire from the
# counter between warm and final probe.
os.environ.setdefault("VLLM_HIMA_HPB_WINDOW_S", "3600")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-35B-A3B"
# Bundled into dev/ so the experiment is self-contained on any host.
DATA = Path(__file__).resolve().parent / "cc_long_traces.jsonl"
BLOCK_SIZE = 1056
MAX_PROMPT_TOKENS = 60_000
N_ANCHOR_WARM = 500  # high enough that anchor's n_b dominates cc-session hits
                     # (each cc turn issues 2 requests → ~100 hits for top blocks
                     # of a 50-turn session; we want anchor 5x above that)
N_BURST_SESSIONS = 10  # cc sessions to replay as cold burst between warm & probe
N_TPOT_TOKENS = 20  # how many decode tokens for the throughput pass


def flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p))
            continue
        t = p.get("type", "")
        if t == "text":
            parts.append(p.get("text", ""))
        elif t == "tool_use":
            inner = p.get("input", "")
            if not isinstance(inner, str):
                inner = json.dumps(inner, ensure_ascii=False)
            parts.append(
                f"<tool_use name={p.get('name', '')} "
                f"id={p.get('id', '')}>{inner}</tool_use>"
            )
        elif t == "tool_result":
            inner = p.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", str(x)) if isinstance(x, dict) else str(x)
                    for x in inner
                )
            elif not isinstance(inner, str):
                inner = str(inner)
            parts.append(
                f"<tool_result id={p.get('tool_use_id', '')}>"
                f"{inner}</tool_result>"
            )
        else:
            parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)


def msg_to_chunk(m: dict) -> str:
    role = m.get("role", "user")
    return f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lru", "lpb"], required=True)
    ap.add_argument("--trial", type=int, default=1,
                    help="Trial index; writes dev/compare_{mode}{tag}_t{trial}.jsonl. "
                         "Use 1,2,3,… to capture noise via independent engine loads.")
    ap.add_argument("--util", type=float, default=0.35,
                    help="gpu_memory_utilization for the engine. Path-0 ran "
                         "at 0.35 (KV ≈ 1.08M tokens). Path-A uses 0.9 for "
                         "realistic operating point.")
    ap.add_argument("--tp", type=int, default=2,
                    help="tensor_parallel_size. Bigger models need TP=4 or 8.")
    ap.add_argument("--model", default=MODEL,
                    help="HF model id. Default keeps Qwen3.5-35B-A3B; bigger "
                         "models (e.g. Qwen3.5-122B-A10B) feed Path B.")
    ap.add_argument("--tag", default="",
                    help="Suffix tag baked into the output filename (e.g. "
                         "'_pathA' / '_pathB') so different sweeps don't "
                         "clobber each other.")
    ap.add_argument("--phase-f-scale", type=int, default=1,
                    help="Multiplier for Phase F's adversarial size. "
                         "scale=1 = 5 decoys × 10K (baseline, ~5% of "
                         "util-0.35 KV budget). scale=10 = 50 decoys × 30K "
                         "(~54% of util-0.9 KV budget) — designed to "
                         "actually expose LPB's worst case.")
    args = ap.parse_args()
    mode = args.mode
    hima_on = mode == "lpb"
    trial = args.trial
    model_id = args.model
    tag = args.tag
    util = args.util
    tp = args.tp
    pf_scale = args.phase_f_scale
    # Deterministic per (mode, trial) so Phase E/F's random content is
    # comparable between LRU and LPB on the same trial index.
    rng = random.Random(1000 + trial)

    out_jsonl = Path(f"dev/compare_{mode}{tag}_t{trial}.jsonl")
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731
    log(kind="meta", mode=mode, trial=trial, model=model_id, tag=tag,
        util=util, tp=tp, phase_f_scale=pf_scale)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    sessions: list[list[dict]] = [
        json.loads(l)["messages"]
        for l in DATA.read_text().splitlines() if l.strip()
    ][:N_BURST_SESSIONS + 1]  # +1 for session 0 (anchor)

    # Anchor = session 0's first user message
    first_user = next(m for m in sessions[0] if m["role"] == "user")
    anchor_ids = tokenizer.encode(
        msg_to_chunk(first_user), add_special_tokens=False
    )
    anchor_len = len(anchor_ids)
    print(f"[{mode}] Anchor: {anchor_len} tokens "
          f"(~{anchor_len / BLOCK_SIZE:.1f} blocks)")

    print(f"[{mode}] Loading {model_id} (TP={tp}, util={util}, "
          f"hima_enabled={hima_on}, phase_f_scale={pf_scale})...")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=MAX_PROMPT_TOKENS + 1024,
        gpu_memory_utilization=util,
        max_num_seqs=64,
        trust_remote_code=True,
        hima_enabled=hima_on,
    )

    def issue(token_ids, max_tokens: int) -> dict:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        t0 = time.monotonic()
        outs = llm.generate(prompts=[token_ids], sampling_params=sp, use_tqdm=False)
        wall = time.monotonic() - t0
        out = outs[0]
        n_out = len(out.outputs[0].token_ids) if out.outputs else 0
        return {
            "wall_s": wall,
            "cached": out.num_cached_tokens or 0,
            "prompt_len": len(token_ids),
            "output_tokens": n_out,
        }

    t_start = time.monotonic()

    # ----- Phase A: warm anchor N_ANCHOR_WARM times -----
    print(f"\n[{mode}] Phase A: warming anchor {N_ANCHOR_WARM}x ...")
    for i in range(N_ANCHOR_WARM):
        r = issue(anchor_ids, max_tokens=1)
        if i in (0, 1, N_ANCHOR_WARM // 2, N_ANCHOR_WARM - 1):
            print(f"  warm[{i:>3}] cached={r['cached']}/{anchor_len}  "
                  f"wall={r['wall_s']*1000:.0f}ms")
    log(kind="phase", phase="A_done", elapsed_s=time.monotonic() - t_start)

    # Baseline probe (anchor should be fully cached)
    r = issue(anchor_ids, max_tokens=1)
    print(f"\n[{mode}] BASELINE anchor probe: cached={r['cached']}/{anchor_len} "
          f"({100*r['cached']/anchor_len:.1f}%)")
    log(kind="anchor_probe", label="baseline",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    # ----- Phase B: cc cold burst with TTFT + TPOT passes per turn -----
    print(f"\n[{mode}] Phase B: cc burst over {N_BURST_SESSIONS} sessions; "
          f"each turn issued with max_tokens=1 (TTFT) then max_tokens={N_TPOT_TOKENS+1} "
          "(TPOT/throughput).")
    for s_idx in range(1, 1 + N_BURST_SESSIONS):
        msgs = sessions[s_idx]
        running_ids: list[int] = []
        prev_len = 0
        turn = 0
        for m in msgs:
            piece = tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
            running_ids.extend(piece)
            if m.get("role") != "assistant":
                continue
            if len(running_ids) > MAX_PROMPT_TOKENS:
                break
            # Pass 1: TTFT (max_tokens=1)
            r_t = issue(running_ids, max_tokens=1)
            # Pass 2: throughput (max_tokens=N_TPOT_TOKENS+1)
            r_d = issue(running_ids, max_tokens=N_TPOT_TOKENS + 1)
            new_content = len(running_ids) - prev_len
            log(
                kind="cc_turn", session_idx=s_idx, turn=turn,
                prompt_len=len(running_ids),
                ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
                full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
                output_tokens=r_d["output_tokens"],
                new_content_tokens=new_content,
                elapsed_s=time.monotonic() - t_start,
            )
            prev_len = len(running_ids)
            turn += 1
        print(f"  [{mode}] session {s_idx:>2}: {turn} turns done "
              f"(elapsed {time.monotonic() - t_start:.0f}s)")
    log(kind="phase", phase="B_done", elapsed_s=time.monotonic() - t_start)

    # ----- Phase D: anchor-rehit workload (LPB BEST CASE) ----- #
    # MOVED here from after Phase C. Reason: the Phase C probe is itself a
    # real anchor request whose prefill side-effect re-populates the anchor
    # blocks at LRU's MRU end — by the time we hit Phase D, LRU's cache
    # holds the anchor again and we cannot observe the predicted gap. By
    # running D *before* C, rehit[0]'s ttft_cached truthfully reports the
    # post-burst anchor-survival state (the very first rehit hasn't yet
    # mutated the cache itself).
    print(f"\n[{mode}] Phase D: anchor-rehit workload (30 anchored requests).")
    N_REHIT = 30
    for j in range(N_REHIT):
        # Append a unique 16-token tail so the prompt is anchor + small new
        # content (no inter-rehit caching of the tail).
        tail_text = f"\n<|im_start|>user\n[rehit-{j:03d}] continue\n<|im_end|>\n"
        tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
        prompt = anchor_ids + tail_ids
        r_t = issue(prompt, max_tokens=1)
        r_d = issue(prompt, max_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="rehit_turn", j=j,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if j in (0, 1, N_REHIT // 2, N_REHIT - 1):
            print(f"  rehit[{j:>2}] ttft_cached={r_t['cached']:>5} "
                  f"ttft_wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase E: no-shared-prefix cold flow (LPB hot-path overhead) ----- #
    # 50 short unique 2K-token prompts with TRULY random tokens (per-trial
    # deterministic seed). Replaces the prior "the quick brown fox" filler
    # which caused ~50% block-level aliasing between adjacent unique slices,
    # contaminating the no-shared-prefix claim.
    N_COLD = 50
    # Phase F's cold flow scales with pf_scale to actually saturate KV at
    # high gpu_memory_utilization; Phase E stays small (it measures hot-path
    # overhead, not pressure).
    N_DECOY = 50 * pf_scale
    PROMPT_LEN_COLD = 2048
    print(f"\n[{mode}] Phase E: no-shared-prefix cold flow "
          f"({N_COLD} unique 2K-token random prompts; "
          f"trial seed = {1000 + trial}).")
    # Sample random ints from a safe vocab range; Qwen3.5's vocab is ~152K,
    # we cap at 50K to stay clear of any special-token regions.
    total_tokens = (N_COLD + N_DECOY) * PROMPT_LEN_COLD
    cold_ids = [rng.randint(10, 50_000) for _ in range(total_tokens)]
    for k in range(N_COLD):
        prompt = cold_ids[k * PROMPT_LEN_COLD: (k + 1) * PROMPT_LEN_COLD]
        r_t = issue(prompt, max_tokens=1)
        r_d = issue(prompt, max_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="cold_turn", k=k,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if k in (0, N_COLD // 2, N_COLD - 1):
            print(f"  cold[{k:>2}] cached={r_t['cached']:>3} "
                  f"wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase F: decoy-warming (LPB ADVERSARIAL WORST CASE) ----- #
    # Warm 5 large decoy prefixes 100 hits each — each decoy sized to ~10K
    # tokens (~10 KV blocks). After warming, the 5 decoys collectively hold
    # ~50 blocks (~5% of KV budget) at high LPB scores, but we never re-hit
    # them again. Then a cold-unique flow needs those blocks: under LRU the
    # decoys are evicted by the cold flow's pressure; under LPB they're
    # protected by their hit-count score, forcing the cold flow to evict
    # other (potentially useful) blocks instead.
    #
    # This is the "past hit count does NOT predict future utility" failure
    # mode that the LPB heuristic is structurally vulnerable to.
    # Phase F sizing scales with pf_scale. The LPB-_HIT_SCORE_OFFSET = 1e12
    # means any block with a single hit outranks cold blocks, so
    # N_DECOY_WARM=5 is plenty — keeping it small lets us afford many more
    # decoys without exploding wall time.
    N_DECOYS = 5 * pf_scale
    DECOY_LEN_TARGET = 10_000 + 20_000 * (pf_scale > 1)
    N_DECOY_WARM = 100 if pf_scale == 1 else 5
    print(f"\n[{mode}] Phase F: decoy-warming adversarial worst case "
          f"({N_DECOYS} decoys × {DECOY_LEN_TARGET}-tok × {N_DECOY_WARM} hits, "
          f"then {N_DECOY} cold-unique prompts).")
    # Build distinct decoys by tokenizing varied prose; each decoy is unique
    # so LPB scores them independently from the real anchor.
    base_phrases = [
        "Alpha vector indices traversal: ",
        "Beta sentinel coordinator dispatch: ",
        "Gamma reduction pipeline metadata: ",
        "Delta consensus quorum tracker: ",
        "Epsilon backpressure buffer manifold: ",
        "Zeta hyperscale ingress shuttle: ",
        "Eta stochastic gradient resonance: ",
        "Theta meridian arbitration loop: ",
        "Iota holographic dispatch fabric: ",
        "Kappa coalesced retrieval pipeline: ",
    ]
    decoys_ids: list[list[int]] = []
    for d_idx in range(N_DECOYS):
        phrase = base_phrases[d_idx % len(base_phrases)]
        decoy_text = phrase + (
            f"decoy-{d_idx}-payload word{rng.randint(0, 999_999)} "
            * max(1500, DECOY_LEN_TARGET // 6 + 100)
        )
        d_ids = tokenizer.encode(decoy_text, add_special_tokens=False)
        # Truncate to DECOY_LEN_TARGET so each decoy is the same size
        d_ids = d_ids[:DECOY_LEN_TARGET]
        decoys_ids.append(d_ids)
        if d_idx in (0, N_DECOYS // 2, N_DECOYS - 1):
            print(f"  decoy[{d_idx}]: {len(d_ids)} tokens "
                  f"(~{len(d_ids) / BLOCK_SIZE:.1f} blocks)")
    print(f"  total decoy footprint: "
          f"{N_DECOYS * DECOY_LEN_TARGET} tokens "
          f"(~{N_DECOYS * DECOY_LEN_TARGET / BLOCK_SIZE:.0f} blocks)")
    # Warm all decoys round-robin (interleave so none is too recent at end)
    for hit in range(N_DECOY_WARM):
        for d_idx in range(N_DECOYS):
            r = issue(decoys_ids[d_idx], max_tokens=1)
        if hit in (0, 1, N_DECOY_WARM // 2, N_DECOY_WARM - 1):
            print(f"  decoy_warm[{hit:>3}] last cached={r['cached']}/"
                  f"{len(decoys_ids[-1])} wall={r['wall_s']*1000:.0f}ms")
    log(kind="phase", phase="F_warm_done", elapsed_s=time.monotonic() - t_start)
    # Cold-unique flow (uses the second half of cold_ids so it doesn't alias
    # Phase E's slices)
    for k in range(N_DECOY):
        prompt = cold_ids[
            (N_COLD + k) * PROMPT_LEN_COLD: (N_COLD + k + 1) * PROMPT_LEN_COLD
        ]
        r_t = issue(prompt, max_tokens=1)
        r_d = issue(prompt, max_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="decoy_turn", k=k,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if k in (0, N_DECOY // 2, N_DECOY - 1):
            print(f"  decoy[{k:>2}] cached={r_t['cached']:>3} "
                  f"wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase C: final anchor probe (DIAGNOSTIC ONLY) -----
    # Now meaningless as a "post-burst survival" signal because Phase D
    # rehit[0] already covered that, and Phases D/E/F have churned the
    # cache. Kept for diagnostic completeness — under LPB this should
    # still show anchor cached (high LPB score from Phase A), under LRU
    # it will likely show anchor evicted by the Phase E/F cold flows.
    r = issue(anchor_ids, max_tokens=1)
    pct = 100 * r["cached"] / anchor_len
    print(f"\n[{mode}] FINAL anchor probe (diagnostic, post-D/E/F): "
          f"cached={r['cached']}/{anchor_len} ({pct:.1f}%)")
    log(kind="anchor_probe", label="final",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    fout.close()
    print(f"\n[{mode}] Done (trial {trial}). "
          f"{time.monotonic() - t_start:.0f}s total. "
          f"Log: {out_jsonl}")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
