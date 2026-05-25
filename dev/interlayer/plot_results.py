# SPDX-License-Identifier: Apache-2.0
"""Generate figures from dev/e2e_replay.jsonl.

Produces under dev/figures/:
  fig_anchor_survival.png    -- HiMA L1 claim: anchor cache state over the
                                 workload (one probe between each session).
  fig_hit_rate.png           -- per-request cache-hit fraction, chronological.
  fig_cumulative_waste.png   -- partial-block waste accumulating turn-by-turn.
  fig_per_turn_breakdown.png -- per-turn: cached vs new vs partial-waste.
  fig_per_session_waste.png  -- per-session waste % bar chart.
  fig_dashboard.png          -- 2x2 grid summary.

Run after e2e_replay.py completes:
  .venv/bin/python dev/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

JSONL = Path("dev/e2e_replay.jsonl")
FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    turns = df[df.kind == "session_turn"].reset_index(drop=True)
    probes = df[df.kind == "anchor_probe"].reset_index(drop=True)
    return turns, probes


def plot_anchor_survival(probes: pd.DataFrame, anchor_len: int) -> None:
    """L1 claim: how much of the anchor is still cached, between sessions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = probes.after_session_idx
    y_tok = probes.num_cached_tokens
    y_frac = y_tok / anchor_len * 100
    ax.plot(x, y_frac, marker="o", color="C3", lw=2, markersize=6)
    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.6, label="full anchor")
    ax.set_xlabel("after session index")
    ax.set_ylabel(f"anchor cached (%)\nout of {anchor_len} tokens total")
    ax.set_ylim(-2, 105)
    ax.set_title(
        "L1 claim: under LRU, the shared system-prompt anchor is dropped as "
        "session traffic fills the KV cache"
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_anchor_survival.png", dpi=130)
    plt.close(fig)


def plot_hit_rate(turns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    rates = turns.num_cached_tokens / turns.prompt_len * 100
    sc = ax.scatter(
        turns.global_idx, rates, c=turns.session_idx, cmap="viridis",
        s=18, alpha=0.7,
    )
    ax.set_xlabel("global request index (chronological)")
    ax.set_ylabel("cache hit %  =  num_cached_tokens / prompt_len")
    ax.set_title("Per-request prefix-cache hit rate over the cc workload "
                 "(colored by session)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("session index")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_hit_rate.png", dpi=130)
    plt.close(fig)


def plot_cumulative_waste(turns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    cumw = turns.partial_block_waste.cumsum()
    cumc = turns.new_content_tokens.cumsum()
    ax.plot(turns.global_idx, cumw / 1e3, lw=2, color="C3",
            label="cumulative partial-block waste")
    ax.plot(turns.global_idx, cumc / 1e3, lw=2, color="C0", alpha=0.5,
            label="cumulative new content")
    ax.set_xlabel("global request index")
    ax.set_ylabel("tokens  (×1000)")
    ax.set_title("L2 claim: partial-block bubble accumulates linearly "
                 "with workload progress")
    final_pct = (
        100 * cumw.iloc[-1] / cumc.iloc[-1] if cumc.iloc[-1] > 0 else 0
    )
    ax.text(
        0.98, 0.02,
        f"final workload-weighted waste = {final_pct:.1f}%\n"
        f"({int(cumw.iloc[-1]):,} / {int(cumc.iloc[-1]):,} tokens)",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9),
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cumulative_waste.png", dpi=130)
    plt.close(fig)


def plot_per_turn_breakdown(turns: pd.DataFrame) -> None:
    """For each global turn: stacked bar of cached / new_content / wasted.

    Wasted = portion of prev_total that did NOT come back as cached.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    x = turns.global_idx.to_numpy()
    cached = turns.num_cached_tokens.to_numpy()
    new = turns.new_content_tokens.to_numpy()
    waste = turns.partial_block_waste.to_numpy()
    ax.bar(x, cached, width=1.0, color="#3d8540", label="cache hit (free)")
    ax.bar(x, waste, width=1.0, bottom=cached, color="#c74848",
           label="partial-block re-prefill (waste)")
    ax.bar(x, new, width=1.0, bottom=cached + waste, color="#4d80c2",
           label="new content prefill")
    ax.set_xlabel("global request index")
    ax.set_ylabel("tokens in this request's prompt")
    ax.set_title(
        "L2 visualization: every request's prompt decomposes into "
        "(cache hit) + (last-block re-prefill) + (new content)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_per_turn_breakdown.png", dpi=130)
    plt.close(fig)


def plot_per_session_waste(turns: pd.DataFrame) -> None:
    g = turns.groupby("session_idx").agg(
        new=("new_content_tokens", "sum"),
        waste=("partial_block_waste", "sum"),
        turns=("turn_idx", "count"),
    )
    g["waste_pct"] = 100 * g.waste / g.new.replace(0, 1)
    g = g.sort_values("waste_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(g)), g.waste_pct, color="C3",
        tick_label=[f"s{i}\n({t}t)" for i, t in zip(g.index, g.turns)],
    )
    ax.set_ylabel("waste %  =  Σpartial_waste / Σnew_content")
    ax.set_title("Per-session partial-block waste %, sorted ascending "
                 "(label = session_idx, t = #turns)")
    ax.axhline(g.waste_pct.mean(), color="black", ls="--", lw=1, alpha=0.5,
               label=f"mean = {g.waste_pct.mean():.1f}%")
    ax.legend()
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_per_session_waste.png", dpi=130)
    plt.close(fig)


def plot_dashboard(
    turns: pd.DataFrame, probes: pd.DataFrame, anchor_len: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) anchor survival
    ax = axes[0, 0]
    if not probes.empty:
        y_frac = probes.num_cached_tokens / anchor_len * 100
        ax.plot(probes.after_session_idx, y_frac, marker="o", color="C3", lw=2)
        ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-2, 105)
    ax.set_title(f"Anchor cache state (anchor = {anchor_len} tok)")
    ax.set_xlabel("after session idx")
    ax.set_ylabel("anchor cached %")
    ax.grid(True, alpha=0.3)

    # (0,1) hit rate scatter
    ax = axes[0, 1]
    rates = turns.num_cached_tokens / turns.prompt_len * 100
    sc = ax.scatter(turns.global_idx, rates, c=turns.session_idx,
                    cmap="viridis", s=12, alpha=0.7)
    ax.set_title("Per-request hit rate")
    ax.set_xlabel("global idx")
    ax.set_ylabel("hit %")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax).set_label("session idx")

    # (1,0) cumulative waste
    ax = axes[1, 0]
    cumw = turns.partial_block_waste.cumsum()
    cumc = turns.new_content_tokens.cumsum()
    ax.plot(turns.global_idx, cumw / 1e3, lw=2, color="C3",
            label="waste")
    ax.plot(turns.global_idx, cumc / 1e3, lw=2, color="C0", alpha=0.5,
            label="new content")
    final_pct = 100 * cumw.iloc[-1] / cumc.iloc[-1] if cumc.iloc[-1] else 0
    ax.set_title(f"Cumulative bubble (final = {final_pct:.1f}%)")
    ax.set_xlabel("global idx")
    ax.set_ylabel("tokens (×1000)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) per-session waste %
    ax = axes[1, 1]
    g = turns.groupby("session_idx").agg(
        new=("new_content_tokens", "sum"),
        waste=("partial_block_waste", "sum"),
    )
    g["waste_pct"] = 100 * g.waste / g.new.replace(0, 1)
    g = g.sort_values("waste_pct")
    ax.bar(range(len(g)), g.waste_pct, color="C3")
    ax.set_title("Per-session waste % (sorted)")
    ax.set_xlabel("rank (asc)")
    ax.set_ylabel("waste %")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("HiMA L1 + L2 e2e validation on Qwen3.5-35B-A3B + 30 cc sessions",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / "fig_dashboard.png", dpi=130)
    plt.close(fig)


def main() -> None:
    turns, probes = load()
    print(f"Loaded {len(turns)} session_turn rows and "
          f"{len(probes)} anchor_probe rows from {JSONL}.")
    # Anchor length: probes' prompt_len is the anchor itself.
    anchor_len = int(probes.prompt_len.iloc[0]) if not probes.empty else 0

    plot_anchor_survival(probes, anchor_len)
    plot_hit_rate(turns)
    plot_cumulative_waste(turns)
    plot_per_turn_breakdown(turns)
    plot_per_session_waste(turns)
    plot_dashboard(turns, probes, anchor_len)

    print(f"Wrote 6 figures to {FIGDIR}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
