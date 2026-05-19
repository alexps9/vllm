# SPDX-License-Identifier: Apache-2.0
"""Plot L1 anchor-eviction comparison: probe-every-session vs no-probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPLAY_JSONL = Path("dev/e2e_replay.jsonl")
BURST_JSONL = Path("dev/e2e_l1_burst.jsonl")
FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def load_replay_probes() -> list[dict]:
    return [
        json.loads(l) for l in REPLAY_JSONL.read_text().splitlines()
        if l.strip() and json.loads(l).get("kind") == "anchor_probe"
    ]


def load_burst() -> dict:
    rows = [json.loads(l) for l in BURST_JSONL.read_text().splitlines() if l.strip()]
    baseline = next((r for r in rows if r.get("kind") == "probe" and r.get("label") == "baseline"), {})
    final = next((r for r in rows if r.get("kind") == "probe" and r.get("label") == "final"), {})
    return {"baseline": baseline, "final": final}


def main() -> None:
    replay = load_replay_probes()
    burst = load_burst()
    anchor_len = 4737

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # E.1: probe between every session — anchor refreshed each time
    if replay:
        x = [p["after_session_idx"] for p in replay if p.get("after_session_idx", -2) >= 0]
        y = [p["num_cached_tokens"] / anchor_len * 100 for p in replay if p.get("after_session_idx", -2) >= 0]
        ax.plot(x, y, marker="o", color="C0", lw=2, markersize=6,
                label="E.1: probe between every session (anchor refreshed)")

    # E.2: no probes during workload — anchor only measured at endpoints
    if burst.get("baseline") and burst.get("final"):
        x = [0, 29]
        y = [
            burst["baseline"]["cached"] / anchor_len * 100,
            burst["final"]["cached"] / anchor_len * 100,
        ]
        ax.plot(x, y, marker="s", color="C3", lw=2, markersize=10, ls="-",
                label="E.2: NO probes — anchor sits in LRU queue under cold burst")
        # annotate endpoints
        ax.annotate(
            f"BASELINE\n{burst['baseline']['cached']}/{anchor_len}",
            xy=(0, y[0]), xytext=(0.5, y[0] + 3),
            fontsize=9, ha="left",
        )
        ax.annotate(
            f"FULLY EVICTED\n{burst['final']['cached']}/{anchor_len}\n"
            f"({burst['final'].get('cum_new_tokens', 0):,} new tokens of cold burst)",
            xy=(29, y[1]), xytext=(20, 15),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="C3", alpha=0.6),
        )

    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(
        100 * (anchor_len // 1056 * 1056) / anchor_len,
        color="gray", ls=":", lw=0.8, alpha=0.5,
        label=f"theoretical max (floor({anchor_len}/1056)·1056 / {anchor_len} = 89.2%)",
    )
    ax.set_xlabel("after session index")
    ax.set_ylabel("anchor cached %")
    ax.set_ylim(-3, 105)
    ax.set_title(
        f"HiMA L1 e2e validation on Qwen3.5-35B-A3B + real cc traces\n"
        f"(anchor = 4737 tokens, KV budget = 1022 blocks, util=0.35)"
    )
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "fig_l1_anchor_eviction.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
