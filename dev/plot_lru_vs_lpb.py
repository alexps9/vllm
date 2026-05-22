# SPDX-License-Identifier: Apache-2.0
"""Aggregate dev/compare_{lru,lpb}.jsonl and produce LRU vs LPB figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSONL_LRU = Path("dev/compare_lru.jsonl")
JSONL_LPB = Path("dev/compare_lpb.jsonl")
FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    turns = [r for r in rows if r.get("kind") == "cc_turn"]
    probes = [r for r in rows if r.get("kind") == "anchor_probe"]
    return turns, probes


def summarize(turns: list[dict], probes: list[dict]) -> dict:
    if not turns:
        return {}
    sum_prompt = sum(t["prompt_len"] for t in turns)
    sum_cached_ttft = sum(t["ttft_cached"] for t in turns)
    sum_cached_full = sum(t["full_cached"] for t in turns)
    sum_ttft_wall = sum(t["ttft_wall_s"] for t in turns)
    sum_full_wall = sum(t["full_wall_s"] for t in turns)
    sum_out = sum(t["output_tokens"] for t in turns)
    n = len(turns)
    # TPOT proxy: full_wall / output_tokens for the throughput pass. The
    # throughput pass benefits from the TTFT pass's cache priming so prefill
    # is ~free; full_wall is dominated by decode-per-token cost.
    # (We avoid (full - ttft)/(out-1) because cache priming inverts the sign.)
    tpots = [
        t["full_wall_s"] / t["output_tokens"]
        for t in turns if t["output_tokens"] > 0
    ]
    mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    base = next(
        (p for p in probes if p.get("label") == "baseline"), None
    )
    fin = next(
        (p for p in probes if p.get("label") == "final"), None
    )
    return {
        "n_requests": n,
        "total_prompt_tokens": sum_prompt,
        "cache_hit_pct_ttft": 100 * sum_cached_ttft / sum_prompt if sum_prompt else 0,
        "cache_hit_pct_full": 100 * sum_cached_full / sum_prompt if sum_prompt else 0,
        "mean_ttft_ms": 1000 * sum_ttft_wall / n,
        "mean_full_wall_ms": 1000 * sum_full_wall / n,
        "mean_tpot_ms": 1000 * mean_tpot,
        "throughput_tok_per_s": sum_out / sum_full_wall if sum_full_wall else 0,
        "total_wall_s": sum_full_wall + sum_ttft_wall,
        "total_output_tokens": sum_out,
        "baseline_anchor_cached": base["cached"] if base else None,
        "final_anchor_cached": fin["cached"] if fin else None,
        "anchor_len": base["anchor_len"] if base else 4737,
    }


def main() -> None:
    lru_t, lru_p = load(JSONL_LRU)
    lpb_t, lpb_p = load(JSONL_LPB)
    s_lru = summarize(lru_t, lru_p)
    s_lpb = summarize(lpb_t, lpb_p)

    print("\n" + "=" * 78)
    print("LRU vs LPB on Qwen3.5-35B-A3B, 10 cc sessions cold-burst")
    print("=" * 78)
    print(f"  {'metric':<28} {'LRU (baseline)':>16} {'LPB (HiMA L1)':>16} "
          f"{'Δ':>10}")
    print("  " + "-" * 76)

    def row(label: str, k: str, fmt: str, better: str = "lower") -> None:
        v1 = s_lru.get(k)
        v2 = s_lpb.get(k)
        if v1 is None or v2 is None:
            return
        delta = v2 - v1
        # for "lower better" metrics, report as Δ%; same for "higher better"
        if v1 != 0:
            pct = (delta / v1) * 100
        else:
            pct = float("inf")
        print(f"  {label:<28} {fmt.format(v1):>16} {fmt.format(v2):>16} "
              f"{pct:>+8.1f}%")

    row("requests issued",              "n_requests",            "{:.0f}")
    row("total prompt tokens",          "total_prompt_tokens",   "{:.0f}")
    row("cache hit % (TTFT pass)",      "cache_hit_pct_ttft",    "{:.2f}%")
    row("cache hit % (decode pass)",    "cache_hit_pct_full",    "{:.2f}%")
    row("mean TTFT (ms)",               "mean_ttft_ms",          "{:.1f}")
    row("mean full-wall (ms)",          "mean_full_wall_ms",     "{:.1f}")
    row("mean TPOT (ms/tok)",           "mean_tpot_ms",          "{:.2f}")
    row("throughput (out_tok/sec)",     "throughput_tok_per_s",  "{:.1f}")
    row("total wall (s)",               "total_wall_s",          "{:.1f}")
    row("BASELINE anchor cached",       "baseline_anchor_cached", "{:.0f}")
    row("FINAL anchor cached",          "final_anchor_cached",    "{:.0f}")

    # --- Figure 1: anchor survival comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    anchor_len = s_lru.get("anchor_len", 4737)
    modes = ["LRU\n(vLLM default)", "LPB\n(HiMA L1)"]
    baseline = [s_lru.get("baseline_anchor_cached", 0),
                s_lpb.get("baseline_anchor_cached", 0)]
    final = [s_lru.get("final_anchor_cached", 0),
             s_lpb.get("final_anchor_cached", 0)]
    x = range(len(modes))
    w = 0.35
    ax.bar([i - w/2 for i in x], baseline, w, label="BASELINE probe",
           color="#7ca8c8")
    ax.bar([i + w/2 for i in x], final, w,
           label="FINAL probe (after 10 cc sessions cold burst)",
           color="#c74848")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel(f"anchor cached tokens (out of {anchor_len})")
    ax.set_title("L1: anchor survival under cold burst — LRU vs LPB")
    ax.axhline(anchor_len, color="gray", ls="--", lw=0.7, alpha=0.5,
               label=f"full anchor ({anchor_len} tokens)")
    for i, (b, f_) in enumerate(zip(baseline, final)):
        ax.annotate(f"{b}\n({100*b/anchor_len:.1f}%)",
                    xy=(i - w/2, b), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{f_}\n({100*f_/anchor_len:.1f}%)",
                    xy=(i + w/2, f_), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGDIR / "fig_lru_vs_lpb_anchor.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out}")

    # --- Figure 2: side-by-side metrics ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [
        ("cache hit %\n(decode pass)", "cache_hit_pct_full", "higher", "%"),
        ("mean TTFT (ms)", "mean_ttft_ms", "lower", "ms"),
        ("mean TPOT (ms/tok)", "mean_tpot_ms", "lower", "ms"),
        ("throughput (tok/s)", "throughput_tok_per_s", "higher", "tok/s"),
    ]
    for ax, (label, key, direction, unit) in zip(axes, metrics):
        vs = [s_lru.get(key, 0), s_lpb.get(key, 0)]
        colors = ["#7ca8c8", "#3d8540"] if direction == "higher" else ["#c74848", "#3d8540"]
        # Make LPB green if it wins, otherwise show LRU green.
        winner_idx = 1 if (
            (direction == "higher" and vs[1] > vs[0])
            or (direction == "lower" and vs[1] < vs[0])
        ) else 0
        colors = ["#a8b0b8", "#a8b0b8"]
        colors[winner_idx] = "#3d8540"
        ax.bar(["LRU", "LPB"], vs, color=colors)
        for i, v in enumerate(vs):
            ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("L1 wiring: LRU vs LPB workload metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = FIGDIR / "fig_lru_vs_lpb_metrics.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"Wrote {out2}")

    # Dump aggregated stats
    out_json = Path("dev/compare_summary.json")
    out_json.write_text(json.dumps({"lru": s_lru, "lpb": s_lpb}, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
