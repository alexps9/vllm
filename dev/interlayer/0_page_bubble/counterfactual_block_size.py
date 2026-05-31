# SPDX-License-Identifier: Apache-2.0
"""Counterfactual: what if vLLM's block_size weren't inflated to 1056?

Findings A-D established that on Qwen3.5-35B-A3B, vLLM's hybrid-padding
inflate forces block_size from the default 16 to 1056 (×66). Finding D
showed real cc traffic loses 42.62% of new content to partial-block
waste at block_size=1056.

This script applies the *same* formula on the *same* 106 cc sessions at
a sweep of block sizes:

  block_size ∈ {16, 32, 64, 128, 256, 512, 1056, 2112}

`floor((L_prev - 1) / B) × B` is the cache hit length on every turn
when block_size = B. Workload-weighted waste % = (Σ partial-block
waste) / (Σ new content tokens) summed across all sessions.

This isolates the *block_size* contribution to the bubble — everything
else (tokenization, real cc traffic, lcm-rounded prefix cache) is held
constant. Result: a clean curve showing how much of vLLM's 42.6% bubble
disappears when block_size shrinks.

Run via:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python dev/counterfactual_block_size.py | tee dev/counterfactual_block_size.out
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

DATA = Path("/data/yuzhou/projects/vllm-songyang/dev/intralayer/cc_long_traces.jsonl")
MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZES = [16, 32, 64, 128, 256, 512, 1056, 2112]
FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


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
                f"<tool_result id={p.get('tool_use_id', '')}>{inner}</tool_result>"
            )
        else:
            parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)


def main() -> None:
    print(f"Loading tokenizer {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sessions: list[list[dict]] = []
    with DATA.open() as f:
        for line in f:
            sessions.append(json.loads(line)["messages"])
    print(f"Loaded {len(sessions)} sessions")

    # Pre-tokenize each session into a list of cumulative-after-each-message
    # token lengths. Same approach as dev/real_session_waste.py.
    print("Tokenizing all sessions (incremental per-message)...")
    all_cumlens: list[list[int]] = []
    for s_idx, msgs in enumerate(sessions):
        running = 0
        cumlen: list[int] = []
        for m in msgs:
            chunk = f"<|im_start|>{m.get('role', 'user')}\n{flatten_content(m.get('content'))}<|im_end|>\n"
            running += len(tok.encode(chunk, add_special_tokens=False))
            cumlen.append(running)
        all_cumlens.append(cumlen)
        if s_idx % 25 == 0:
            print(f"  [{s_idx + 1}/{len(sessions)}] final cumlen={running}")

    # For each block_size, compute waste stats.
    results: dict[int, dict] = {}

    for B in BLOCK_SIZES:
        per_sess_waste_pct: list[float] = []
        total_waste = 0
        total_new = 0

        for msgs, cumlen in zip(sessions, all_cumlens):
            # turn ends = after each assistant message
            turn_ends = [cumlen[i] for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if len(turn_ends) < 2:
                continue
            s_waste = 0
            s_new = 0
            for n in range(1, len(turn_ends)):
                prev = turn_ends[n - 1]
                expected = ((prev - 1) // B) * B
                s_waste += max(prev - expected, 0)
                s_new += turn_ends[n] - turn_ends[n - 1]
            if s_new > 0:
                per_sess_waste_pct.append(100 * s_waste / s_new)
            total_waste += s_waste
            total_new += s_new

        workload_pct = 100 * total_waste / total_new if total_new else 0
        results[B] = {
            "block_size": B,
            "workload_weighted_waste_pct": workload_pct,
            "total_waste_tokens": total_waste,
            "total_new_tokens": total_new,
            "median_session_waste_pct": (
                statistics.median(per_sess_waste_pct) if per_sess_waste_pct else 0
            ),
            "p95_session_waste_pct": (
                statistics.quantiles(per_sess_waste_pct, n=100)[94]
                if len(per_sess_waste_pct) >= 100 else max(per_sess_waste_pct, default=0)
            ),
        }
        print(f"\nblock_size={B:>5}:")
        print(f"  workload-weighted waste = {workload_pct:>5.2f}%")
        print(f"  total waste tokens      = {total_waste:>10,}")
        print(f"  median session waste %  = {results[B]['median_session_waste_pct']:>5.1f}%")
        print(f"  p95 session waste %     = {results[B]['p95_session_waste_pct']:>5.1f}%")

    # Save raw results
    out_path = Path("dev/interlayer/0_page_bubble/runs/counterfactual_block_size.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # ----- Plot 1: workload-weighted waste vs block_size -----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = [r["block_size"] for r in results.values()]
    ys_workload = [r["workload_weighted_waste_pct"] for r in results.values()]
    ys_p50 = [r["median_session_waste_pct"] for r in results.values()]
    ys_p95 = [r["p95_session_waste_pct"] for r in results.values()]
    ax.plot(xs, ys_workload, marker="o", lw=2, label="workload-weighted (Σwaste/Σnew)", color="C3")
    ax.plot(xs, ys_p50, marker="s", lw=2, label="per-session median", color="C0")
    ax.plot(xs, ys_p95, marker="^", lw=2, label="per-session p95", color="C2")
    ax.axvline(1056, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(1056, ax.get_ylim()[1] * 0.95,
            "  vLLM's inflated\n  block_size on this model = 1056",
            fontsize=9, va="top", color="gray")
    ax.axvline(16, color="gray", ls=":", lw=1, alpha=0.4)
    ax.text(16, ax.get_ylim()[1] * 0.4,
            "  vLLM default\n  block_size = 16",
            fontsize=9, va="bottom", color="gray")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("hypothetical block_size (tokens)")
    ax.set_ylabel("partial-block waste %")
    ax.set_title(
        "Counterfactual: workload-weighted partial-block waste vs block_size\n"
        "(106 real Claude Code sessions, same workload, only block_size varies)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGDIR / "fig_block_size_counterfactual.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # ----- Summary line for README -----
    print("\nReadme line:")
    for B in [16, 32, 64, 128, 256, 512, 1056]:
        if B in results:
            print(f"  block_size={B:>5}  → {results[B]['workload_weighted_waste_pct']:>5.2f}% waste")


if __name__ == "__main__":
    main()
