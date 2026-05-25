# SPDX-License-Identifier: Apache-2.0
"""Plot the L1 pressure curve from e2e_l1_pressure_curve.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSONL = Path("dev/e2e_l1_pressure_curve.jsonl")
FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["K"])

    Ks = [r["K"] for r in rows]
    pcts = [r["anchor_pct"] for r in rows]
    cum_new = [r["cum_new_tokens"] for r in rows]
    cum_blocks = [r["cum_new_blocks_lb"] for r in rows]
    anchor_len = rows[0]["anchor_len"] if rows else 4737

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(Ks, pcts, marker="o", color="C3", lw=2.5, markersize=8,
            label="anchor cached % after K cold-burst sessions")

    # Add a baseline line at theoretical max.
    theoretical = 100 * (anchor_len // 1056 * 1056) / anchor_len
    ax.axhline(theoretical, color="gray", ls=":", lw=1, alpha=0.7,
               label=f"theoretical max ({theoretical:.1f}%)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    # Annotate each point with cum_new_blocks.
    for K, pct, b in zip(Ks, pcts, cum_blocks):
        ax.annotate(
            f"{b} blocks",
            xy=(K, pct), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8,
            color="gray", alpha=0.8,
        )

    ax.axhline(50, color="C2", ls="--", lw=0.7, alpha=0.4)
    # KV budget line (1022 blocks → annotate the K where we cross)
    crossover_K = None
    for r in rows:
        if r["cum_new_blocks_lb"] >= 1022:
            crossover_K = r["K"]
            break
    if crossover_K is not None:
        ax.axvline(crossover_K, color="C4", ls="--", lw=1, alpha=0.5)
        ax.text(crossover_K, 70,
                f"  KV budget = 1022 blocks\n  exhausted at K={crossover_K}",
                fontsize=9, color="C4", alpha=0.8)

    ax.set_xlabel("K = cold-burst session count between warm + probe")
    ax.set_ylabel("anchor cached %  (after K silent sessions of cold burst)")
    ax.set_ylim(-3, 100)
    ax.set_title(
        "L1 anchor survival vs cold-burst pressure\n"
        f"(anchor = {anchor_len} tokens warmed 5×, then K disjoint sessions, "
        "then probe; Qwen3.5-35B, util=0.35)"
    )
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "fig_l1_pressure_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    # Print summary
    print("\nL1 pressure curve summary:")
    print(f"  {'K':>3}  {'cum_new':>10}  {'cum_blocks':>10}  {'anchor_pct':>10}")
    for r in rows:
        print(f"  {r['K']:>3}  {r['cum_new_tokens']:>10,}  "
              f"{r['cum_new_blocks_lb']:>10}  {r['anchor_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
