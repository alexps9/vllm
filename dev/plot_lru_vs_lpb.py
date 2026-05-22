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


def load(path: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    turns = [r for r in rows if r.get("kind") == "cc_turn"]
    probes = [r for r in rows if r.get("kind") == "anchor_probe"]
    rehits = [r for r in rows if r.get("kind") == "rehit_turn"]
    colds = [r for r in rows if r.get("kind") == "cold_turn"]
    return turns, probes, rehits, colds


def _aggregate(rows: list[dict]) -> dict:
    """Generic aggregator for rows with prompt_len/ttft_cached/ttft_wall_s/
    full_cached/full_wall_s/output_tokens."""
    if not rows:
        return {}
    sp = sum(r["prompt_len"] for r in rows)
    sc_t = sum(r["ttft_cached"] for r in rows)
    sc_f = sum(r["full_cached"] for r in rows)
    swt = sum(r["ttft_wall_s"] for r in rows)
    swf = sum(r["full_wall_s"] for r in rows)
    so = sum(r["output_tokens"] for r in rows)
    n = len(rows)
    tpots = [
        r["full_wall_s"] / r["output_tokens"]
        for r in rows if r["output_tokens"] > 0
    ]
    return {
        "n_requests": n,
        "total_prompt_tokens": sp,
        "cache_hit_pct_ttft": 100 * sc_t / sp if sp else 0,
        "cache_hit_pct_full": 100 * sc_f / sp if sp else 0,
        "mean_ttft_ms": 1000 * swt / n,
        "mean_full_wall_ms": 1000 * swf / n,
        "mean_tpot_ms": 1000 * sum(tpots) / len(tpots) if tpots else 0.0,
        "throughput_tok_per_s": so / swf if swf else 0,
        "total_wall_s": swf + swt,
        "total_output_tokens": so,
    }


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
    lru_t, lru_p, lru_re, lru_co = load(JSONL_LRU)
    lpb_t, lpb_p, lpb_re, lpb_co = load(JSONL_LPB)
    s_lru = summarize(lru_t, lru_p)
    s_lpb = summarize(lpb_t, lpb_p)
    re_lru = _aggregate(lru_re)
    re_lpb = _aggregate(lpb_re)
    co_lru = _aggregate(lru_co)
    co_lpb = _aggregate(lpb_co)

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

    # --- Phase D & E breakdown ---
    def scenario_rows(name: str, sl: dict, sp_: dict) -> None:
        if not sl or not sp_:
            return
        print(f"\n  [{name}]")
        for label, k, fmt in [
            ("  requests",        "n_requests",            "{:.0f}"),
            ("  hit % (TTFT)",    "cache_hit_pct_ttft",    "{:.2f}%"),
            ("  hit % (decode)",  "cache_hit_pct_full",    "{:.2f}%"),
            ("  mean TTFT (ms)",  "mean_ttft_ms",          "{:.1f}"),
            ("  mean TPOT (ms)",  "mean_tpot_ms",          "{:.2f}"),
            ("  throughput tok/s","throughput_tok_per_s",  "{:.1f}"),
            ("  total wall (s)",  "total_wall_s",          "{:.2f}"),
        ]:
            v1 = sl.get(k)
            v2 = sp_.get(k)
            if v1 is None or v2 is None:
                continue
            delta = v2 - v1
            pct = (delta / v1) * 100 if v1 else float("inf")
            print(f"  {label:<28} {fmt.format(v1):>16} {fmt.format(v2):>16} "
                  f"{pct:>+8.1f}%")

    scenario_rows("Phase D: anchor-rehit (LPB BEST)", re_lru, re_lpb)
    scenario_rows("Phase E: no-shared cold (LPB WORST)", co_lru, co_lpb)

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

    # --- Figure 3: per-scenario throughput / TTFT (Phase B/D/E side by side) ---
    if re_lru and co_lru and re_lpb and co_lpb:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        scenarios = ["B: cc burst", "D: anchor re-hit", "E: cold unique"]
        ttfts_lru = [s_lru["mean_ttft_ms"], re_lru["mean_ttft_ms"], co_lru["mean_ttft_ms"]]
        ttfts_lpb = [s_lpb["mean_ttft_ms"], re_lpb["mean_ttft_ms"], co_lpb["mean_ttft_ms"]]
        tpots_lru = [s_lru["mean_tpot_ms"], re_lru["mean_tpot_ms"], co_lru["mean_tpot_ms"]]
        tpots_lpb = [s_lpb["mean_tpot_ms"], re_lpb["mean_tpot_ms"], co_lpb["mean_tpot_ms"]]
        thr_lru = [s_lru["throughput_tok_per_s"], re_lru["throughput_tok_per_s"], co_lru["throughput_tok_per_s"]]
        thr_lpb = [s_lpb["throughput_tok_per_s"], re_lpb["throughput_tok_per_s"], co_lpb["throughput_tok_per_s"]]
        hit_lru = [s_lru["cache_hit_pct_ttft"], re_lru["cache_hit_pct_ttft"], co_lru["cache_hit_pct_ttft"]]
        hit_lpb = [s_lpb["cache_hit_pct_ttft"], re_lpb["cache_hit_pct_ttft"], co_lpb["cache_hit_pct_ttft"]]

        def pair_bars(ax, label, vlru, vlpb, ylabel):
            x = range(len(scenarios))
            w = 0.35
            ax.bar([i - w/2 for i in x], vlru, w, label="LRU", color="#a8b0b8")
            ax.bar([i + w/2 for i in x], vlpb, w, label="LPB", color="#3d8540")
            ax.set_xticks(list(x))
            ax.set_xticklabels(scenarios, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis="y")
            for i, (a, b) in enumerate(zip(vlru, vlpb)):
                ax.annotate(f"{a:.1f}", xy=(i - w/2, a), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8)
                ax.annotate(f"{b:.1f}", xy=(i + w/2, b), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8)
            ax.legend(loc="upper left", fontsize=9)

        pair_bars(axes[0, 0], "TTFT (ms)", ttfts_lru, ttfts_lpb, "ms")
        pair_bars(axes[0, 1], "TPOT (ms/tok)", tpots_lru, tpots_lpb, "ms/tok")
        pair_bars(axes[1, 0], "throughput (out_tok/s)", thr_lru, thr_lpb, "tok/s")
        pair_bars(axes[1, 1], "hit % (TTFT pass)", hit_lru, hit_lpb, "%")
        fig.suptitle("LRU vs LPB across 3 scenarios: best/average/worst", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out3 = FIGDIR / "fig_lru_vs_lpb_scenarios.png"
        fig.savefig(out3, dpi=130)
        plt.close(fig)
        print(f"Wrote {out3}")

    # Dump aggregated stats
    out_json = Path("dev/compare_summary.json")
    payload = {
        "phase_B_cc_burst":   {"lru": s_lru,  "lpb": s_lpb},
        "phase_D_anchor_rehit": {"lru": re_lru, "lpb": re_lpb},
        "phase_E_cold_unique":  {"lru": co_lru, "lpb": co_lpb},
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
