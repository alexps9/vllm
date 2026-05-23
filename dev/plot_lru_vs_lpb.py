# SPDX-License-Identifier: Apache-2.0
"""Aggregate dev/compare_{lru,lpb}_t*.jsonl across trials and produce
mean ± stddev figures for LRU vs LPB across Phase B / D / E / F.

Loads all trial files for each mode (dev/compare_{mode}_t{N}.jsonl) and
computes per-phase statistics across trials so we can tell stable signal
from per-run noise.

Headline anchor-survival signal uses rehit_turn[j=0].ttft_cached (the
post-burst state, observed before Phase D's first request mutates the
cache). The Phase C final probe is now diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path("dev/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("dev")


# ---------------------------------------------------------------------------
# Per-trial aggregation
# ---------------------------------------------------------------------------


def _aggregate(rows: list[dict]) -> dict:
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


def _aggregate_swarm(row: dict | None, n_tpot_tokens: int = 20) -> dict:
    """Phase G's single batch row → comparable per-phase metric dict.

    TTFT proxy = batch wall (worst-case wait in batch; LRU pays anchor
        prefill, LPB doesn't).
    TPOT proxy = (full_batch_wall - ttft_batch_wall) / N_TPOT_TOKENS
        (post-prefill decode time per token, normalised by output tokens
        per request).
    throughput = total output tokens / full batch wall.
    """
    if not row:
        return {}
    n = row["n_requests"]
    sp = row["total_prompt_tokens"]
    sc_t = row["ttft_batch_cached_total"]
    sc_f = row["full_batch_cached_total"]
    so = row["full_total_output_tokens"]
    ttft_wall = row["ttft_batch_wall_s"]
    full_wall = row["full_batch_wall_s"]
    # Decode-per-token: (full pass wall − TTFT pass wall) / output tokens per req.
    # The full pass also does prefill; the difference between full and TTFT
    # is mostly the additional N_TPOT decode steps.
    decode_wall = max(full_wall - ttft_wall, 1e-9)
    tpot_ms = 1000 * decode_wall / max(n_tpot_tokens, 1)
    return {
        "n_requests": n,
        "total_prompt_tokens": sp,
        "cache_hit_pct_ttft": 100 * sc_t / sp if sp else 0,
        "cache_hit_pct_full": 100 * sc_f / sp if sp else 0,
        "mean_ttft_ms": 1000 * ttft_wall,            # batch wall = worst-case wait
        "mean_full_wall_ms": 1000 * full_wall,
        "mean_tpot_ms": tpot_ms,
        "throughput_tok_per_s": so / full_wall if full_wall else 0,
        "total_wall_s": ttft_wall + full_wall,
        "total_output_tokens": so,
    }


def trial_summary(trial_path: Path) -> dict[str, dict]:
    """Return per-phase aggregated metrics for a single trial."""
    rows = [json.loads(l) for l in trial_path.read_text().splitlines() if l.strip()]
    turns = [r for r in rows if r.get("kind") == "cc_turn"]
    rehits = [r for r in rows if r.get("kind") == "rehit_turn"]
    colds = [r for r in rows if r.get("kind") == "cold_turn"]
    decoys = [r for r in rows if r.get("kind") == "decoy_turn"]
    probes = [r for r in rows if r.get("kind") == "anchor_probe"]
    swarm_batch = next((r for r in rows if r.get("kind") == "swarm_batch"), None)
    swarm_turns = [r for r in rows if r.get("kind") == "swarm_turn"]
    swarm2_batch = next((r for r in rows if r.get("kind") == "swarm2_batch"), None)
    swarm2_turns = [r for r in rows if r.get("kind") == "swarm2_turn"]

    anchor_len = 4737
    base = next((p for p in probes if p.get("label") == "baseline"), None)
    final = next((p for p in probes if p.get("label") == "final"), None)
    if base:
        anchor_len = base["anchor_len"]

    # Post-burst anchor survival from rehit[0] — the very first rehit hasn't
    # yet mutated the cache, so its ttft_cached truthfully reports the state
    # after Phase B's cold burst.
    rehit0 = next((r for r in rehits if r.get("j") == 0), None)
    post_burst_cached = rehit0["ttft_cached"] if rehit0 else None
    # If Phase G exists, it runs *before* Phase D and observes the post-B
    # state directly (all N submitted simultaneously, none has yet mutated
    # cache). The aggregate cache hit % across the batch is a stronger
    # anchor-survival signal than rehit[0] because it averages over N
    # independent observations under identical conditions.
    swarm_anchor_signal = None
    if swarm_batch:
        # Cache hit % of TTFT batch — under LRU with anchor evicted, all N
        # miss (≈0%); under LPB with anchor protected, all N hit (≈89%).
        swarm_anchor_signal = swarm_batch["ttft_batch_cached_total"]
    swarm2_anchor_signal = None
    if swarm2_batch:
        swarm2_anchor_signal = swarm2_batch["ttft_batch_cached_total"]

    return {
        "phase_B": _aggregate(turns),
        "phase_D": _aggregate(rehits),
        "phase_E": _aggregate(colds),
        "phase_F": _aggregate(decoys),
        "phase_G": _aggregate_swarm(swarm_batch),
        "phase_H": _aggregate_swarm(swarm2_batch),  # post-pressure swarm
        "anchor_len": anchor_len,
        "baseline_anchor_cached": base["cached"] if base else None,
        "final_anchor_cached": final["cached"] if final else None,  # diagnostic
        "post_burst_cached": post_burst_cached,  # rehit[0] signal
        "swarm_batch_cached": swarm_anchor_signal,  # G batch aggregate
        "swarm2_batch_cached": swarm2_anchor_signal,  # H batch aggregate
    }


def discover_trials(mode: str, tag: str = "") -> list[Path]:
    """Return all dev/compare_{mode}{tag}_t*.jsonl files, sorted by trial index.

    `tag` selects the sweep (e.g. "" for Path-0, "_pathA" for the util=0.9
    sweep). Files for one sweep don't match the glob of another sweep
    because the chars between `_{mode}_` and `_t*` differ.
    """
    paths = sorted(ROOT.glob(f"compare_{mode}{tag}_t*.jsonl"))
    if paths:
        return paths
    # Fallback: legacy single-trial file (only meaningful with empty tag)
    if not tag:
        legacy = ROOT / f"compare_{mode}.jsonl"
        return [legacy] if legacy.exists() else []
    return []


# ---------------------------------------------------------------------------
# Trial aggregation: mean + stddev across trials
# ---------------------------------------------------------------------------


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def aggregate_trials(trial_summaries: list[dict]) -> dict:
    """Given per-trial summaries, compute mean ± stddev per phase per metric."""
    out: dict = {}
    phase_keys = ["phase_B", "phase_D", "phase_E", "phase_F", "phase_G", "phase_H"]
    metric_keys = [
        "n_requests", "total_prompt_tokens", "cache_hit_pct_ttft",
        "cache_hit_pct_full", "mean_ttft_ms", "mean_full_wall_ms",
        "mean_tpot_ms", "throughput_tok_per_s", "total_wall_s",
        "total_output_tokens",
    ]
    for ph in phase_keys:
        out[ph] = {}
        for k in metric_keys:
            xs = [t[ph].get(k) for t in trial_summaries
                  if t.get(ph) and t[ph].get(k) is not None]
            if xs:
                m, s = _mean_std(xs)
                out[ph][k] = {"mean": m, "std": s, "n": len(xs),
                              "trials": xs}

    # Anchor survival across trials (rehit[0].ttft_cached)
    pburst = [t["post_burst_cached"] for t in trial_summaries
              if t.get("post_burst_cached") is not None]
    if pburst:
        m, s = _mean_std(pburst)
        out["post_burst_cached"] = {"mean": m, "std": s, "n": len(pburst),
                                    "trials": pburst}
    base = [t["baseline_anchor_cached"] for t in trial_summaries
            if t.get("baseline_anchor_cached") is not None]
    if base:
        m, s = _mean_std(base)
        out["baseline_anchor_cached"] = {"mean": m, "std": s, "n": len(base),
                                         "trials": base}
    final = [t["final_anchor_cached"] for t in trial_summaries
             if t.get("final_anchor_cached") is not None]
    if final:
        m, s = _mean_std(final)
        out["final_anchor_cached"] = {"mean": m, "std": s, "n": len(final),
                                      "trials": final}
    swarm = [t["swarm_batch_cached"] for t in trial_summaries
             if t.get("swarm_batch_cached") is not None]
    if swarm:
        m, s = _mean_std(swarm)
        out["swarm_batch_cached"] = {"mean": m, "std": s, "n": len(swarm),
                                     "trials": swarm}
    swarm2 = [t["swarm2_batch_cached"] for t in trial_summaries
              if t.get("swarm2_batch_cached") is not None]
    if swarm2:
        m, s = _mean_std(swarm2)
        out["swarm2_batch_cached"] = {"mean": m, "std": s, "n": len(swarm2),
                                      "trials": swarm2}
    out["anchor_len"] = trial_summaries[0]["anchor_len"] if trial_summaries else 4737
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(val: dict | None, fmt: str = "{:.2f}") -> str:
    if not val:
        return "       —      "
    m, s, n = val["mean"], val["std"], val["n"]
    if n < 2:
        return f"{fmt.format(m):>10}  (n=1)"
    return f"{fmt.format(m):>10} ±{fmt.format(s)}"


def _delta_pct(a: dict | None, b: dict | None) -> str:
    if not a or not b or a["mean"] == 0:
        return "   —    "
    pct = (b["mean"] - a["mean"]) / a["mean"] * 100
    return f"{pct:>+7.1f}%"


def print_phase(name: str, lru: dict, lpb: dict) -> None:
    if not lru or not lpb:
        return
    print(f"\n  [{name}]   (mean ± sample stddev across "
          f"{lru.get('mean_ttft_ms', {}).get('n', '?')} LRU trials, "
          f"{lpb.get('mean_ttft_ms', {}).get('n', '?')} LPB trials)")
    for label, k, fmt in [
        ("requests",        "n_requests",            "{:.0f}"),
        ("hit % (TTFT)",    "cache_hit_pct_ttft",    "{:.2f}"),
        ("hit % (decode)",  "cache_hit_pct_full",    "{:.2f}"),
        ("mean TTFT (ms)",  "mean_ttft_ms",          "{:.2f}"),
        ("mean TPOT (ms)",  "mean_tpot_ms",          "{:.3f}"),
        ("throughput tok/s","throughput_tok_per_s",  "{:.2f}"),
        ("total wall (s)",  "total_wall_s",          "{:.2f}"),
    ]:
        a = lru.get(k); b = lpb.get(k)
        print(f"    {label:<18} LRU: {_fmt(a, fmt)}   "
              f"LPB: {_fmt(b, fmt)}   Δ: {_delta_pct(a, b)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="Sweep tag baked into the filename (e.g. '' for "
                         "Path-0 util=0.35, '_pathA' for util=0.9 sweep, "
                         "'_pathB' for the bigger-model sweep).")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for output figure/summary filenames so a "
                         "Path-A plot doesn't overwrite Path-0's files. "
                         "Defaults to --tag value.")
    args = ap.parse_args()
    tag = args.tag
    suffix = args.out_suffix or tag

    lru_paths = discover_trials("lru", tag)
    lpb_paths = discover_trials("lpb", tag)
    print(f"Sweep tag = '{tag}'. Found {len(lru_paths)} LRU trial files, "
          f"{len(lpb_paths)} LPB trial files.")
    for p in lru_paths + lpb_paths:
        print(f"  {p}")

    lru_summaries = [trial_summary(p) for p in lru_paths]
    lpb_summaries = [trial_summary(p) for p in lpb_paths]

    lru_agg = aggregate_trials(lru_summaries)
    lpb_agg = aggregate_trials(lpb_summaries)

    print("\n" + "=" * 82)
    # Best-effort: read model id from the first trial's meta row
    model_label = "?"
    if lru_paths:
        first = lru_paths[0].read_text().splitlines()
        for line in first:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "meta" and r.get("model"):
                model_label = r["model"]
                break
    print(f"LRU vs LPB on {model_label} — multi-trial aggregate (mean ± stddev)")
    print("=" * 82)

    anchor_len = lru_agg.get("anchor_len", 4737)
    print(f"\n  ANCHOR SURVIVAL (post Phase-B burst, observed via rehit[0].ttft_cached)")
    print(f"    anchor full length: {anchor_len} tokens")
    a = lru_agg.get("post_burst_cached")
    b = lpb_agg.get("post_burst_cached")
    print(f"    LRU rehit[0].ttft_cached: {_fmt(a, '{:.0f}')}")
    print(f"    LPB rehit[0].ttft_cached: {_fmt(b, '{:.0f}')}")
    if a and b:
        print(f"      → LRU anchor survival: {100*a['mean']/anchor_len:.1f}% of full anchor")
        print(f"      → LPB anchor survival: {100*b['mean']/anchor_len:.1f}% of full anchor")

    print_phase("Phase B: cc burst (average)",                   lru_agg["phase_B"], lpb_agg["phase_B"])
    print_phase("Phase G: PRE-pressure concurrent swarm",        lru_agg["phase_G"], lpb_agg["phase_G"])
    print_phase("Phase D: anchor re-hit (serial, post-G)",       lru_agg["phase_D"], lpb_agg["phase_D"])
    print_phase("Phase E: cold-unique random (LPB hot-path)",    lru_agg["phase_E"], lpb_agg["phase_E"])
    print_phase("Phase F: decoy waste (LPB WORST, adversarial)", lru_agg["phase_F"], lpb_agg["phase_F"])
    print_phase("Phase H: POST-pressure swarm (decisive test)",  lru_agg["phase_H"], lpb_agg["phase_H"])

    # Phase G/H headline: anchor cached in the concurrent swarm batch
    sb_l = lru_agg.get("swarm_batch_cached")
    sb_p = lpb_agg.get("swarm_batch_cached")
    s2_l = lru_agg.get("swarm2_batch_cached")
    s2_p = lpb_agg.get("swarm2_batch_cached")
    if sb_l or sb_p:
        print(f"\n  PHASE G — swarm batch cache hit (PRE-pressure)")
        print(f"    LRU sum cached: {_fmt(sb_l, '{:.0f}')}")
        print(f"    LPB sum cached: {_fmt(sb_p, '{:.0f}')}")
    if s2_l or s2_p:
        print(f"\n  PHASE H — swarm batch cache hit (POST-pressure, decisive)")
        print(f"    LRU sum cached: {_fmt(s2_l, '{:.0f}')}")
        print(f"    LPB sum cached: {_fmt(s2_p, '{:.0f}')}")

    # ----- Figure 1: anchor survival (now uses rehit[0]) -----
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = ["LRU\n(vLLM default)", "LPB\n(HiMA L1)"]
    base_a = lru_agg.get("baseline_anchor_cached", {"mean": 0, "std": 0})
    base_b = lpb_agg.get("baseline_anchor_cached", {"mean": 0, "std": 0})
    burst_a = lru_agg.get("post_burst_cached", {"mean": 0, "std": 0})
    burst_b = lpb_agg.get("post_burst_cached", {"mean": 0, "std": 0})
    x = range(len(modes))
    w = 0.35
    ax.bar([i - w/2 for i in x], [base_a["mean"], base_b["mean"]], w,
           yerr=[base_a["std"], base_b["std"]], capsize=4,
           label="BASELINE probe", color="#7ca8c8")
    ax.bar([i + w/2 for i in x], [burst_a["mean"], burst_b["mean"]], w,
           yerr=[burst_a["std"], burst_b["std"]], capsize=4,
           label="POST-BURST (rehit[0])",
           color="#c74848")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel(f"anchor cached tokens (out of {anchor_len})")
    ax.set_title("L1 anchor survival under cold burst — LRU vs LPB "
                 f"(mean ± stddev, n={base_a['n']} trials)")
    ax.axhline(anchor_len, color="gray", ls="--", lw=0.7, alpha=0.5,
               label=f"full anchor ({anchor_len} tokens)")
    for i, (bm, fm) in enumerate(zip([base_a["mean"], base_b["mean"]],
                                     [burst_a["mean"], burst_b["mean"]])):
        ax.annotate(f"{bm:.0f}\n({100*bm/anchor_len:.1f}%)",
                    xy=(i - w/2, bm), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{fm:.0f}\n({100*fm/anchor_len:.1f}%)",
                    xy=(i + w/2, fm), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGDIR / f"fig_lru_vs_lpb_anchor{suffix}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out}")

    # ----- Figure 2: 5-scenario grid (B, G, D, E, F) with error bars -----
    # G inserted between B and D — it's the production-pattern win
    # scenario (concurrent swarm), while D is now the post-G serial
    # follow-up.
    all_scenarios = [
        ("B: cc burst",                      "phase_B"),
        ("G: swarm (pre-pressure)",          "phase_G"),
        ("D: anchor re-hit",                 "phase_D"),
        ("E: cold random",                   "phase_E"),
        ("F: decoy waste (adv)",             "phase_F"),
        ("H: swarm (POST-pressure, decisive)", "phase_H"),
    ]
    # Only include phases that have data for at least one mode
    scenarios = [
        (label, ph) for (label, ph) in all_scenarios
        if lru_agg.get(ph) or lpb_agg.get(ph)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metric_panels = [
        ("TTFT (ms)",            "mean_ttft_ms",           "ms"),
        ("TPOT (ms/tok)",        "mean_tpot_ms",           "ms/tok"),
        ("throughput (tok/s)",   "throughput_tok_per_s",   "tok/s"),
        ("hit % (TTFT pass)",    "cache_hit_pct_ttft",     "%"),
    ]
    for ax, (title, key, unit) in zip(axes.flat, metric_panels):
        # TPOT (decode-per-token) is computed as (full_wall − ttft_wall)/N
        # for serial phases, which is meaningful per-request. For Phase G
        # (batched submission) the same formula compares "extra wall after
        # all TTFTs done" to "20 decode tokens per request", which is not
        # comparable to serial TPOT — drop Phase G from this panel.
        scen_here = (
            [s for s in scenarios if s[1] != "phase_G"]
            if key == "mean_tpot_ms" else scenarios
        )
        labels = [s[0] for s in scen_here]
        vlru_m, vlru_s = [], []
        vlpb_m, vlpb_s = [], []
        for _, ph in scen_here:
            v = lru_agg[ph].get(key)
            vlru_m.append(v["mean"] if v else 0.0)
            vlru_s.append(v["std"] if v else 0.0)
            v = lpb_agg[ph].get(key)
            vlpb_m.append(v["mean"] if v else 0.0)
            vlpb_s.append(v["std"] if v else 0.0)
        x = range(len(scen_here))
        w = 0.35
        ax.bar([i - w/2 for i in x], vlru_m, w, yerr=vlru_s, capsize=4,
               label="LRU", color="#a8b0b8")
        ax.bar([i + w/2 for i in x], vlpb_m, w, yerr=vlpb_s, capsize=4,
               label="LPB", color="#3d8540")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        title_full = title if key != "mean_tpot_ms" else f"{title}   (Phase G excluded: batched-mode formula)"
        ax.set_title(title_full)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=9)
        for i, (a, b) in enumerate(zip(vlru_m, vlpb_m)):
            ax.annotate(f"{a:.1f}", xy=(i - w/2, a), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
            ax.annotate(f"{b:.1f}", xy=(i + w/2, b), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    n_trials_lru = len(lru_paths)
    n_trials_lpb = len(lpb_paths)
    fig.suptitle(
        f"LRU vs LPB across {len(scenarios)} scenarios — mean ± stddev "
        f"(LRU n={n_trials_lru}, LPB n={n_trials_lpb})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = FIGDIR / f"fig_lru_vs_lpb_scenarios{suffix}.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"Wrote {out2}")

    # ----- Per-trial scatter for the noisy scenarios (E and F) -----
    if n_trials_lru >= 2 and n_trials_lpb >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (title, ph) in zip(axes, [("E: cold-unique random", "phase_E"),
                                           ("F: decoy waste (adv)",  "phase_F")]):
            keys = ["mean_ttft_ms", "mean_tpot_ms", "throughput_tok_per_s"]
            xs = range(len(keys))
            for j, k in enumerate(keys):
                lru_trials = lru_agg[ph][k]["trials"] if lru_agg[ph].get(k) else []
                lpb_trials = lpb_agg[ph][k]["trials"] if lpb_agg[ph].get(k) else []
                # Normalize each metric by the LRU mean of that metric so
                # they're plottable on a shared axis.
                lru_m = lru_agg[ph][k]["mean"] if lru_agg[ph].get(k) else 1.0
                lru_m = lru_m if lru_m else 1.0
                jitter_l = [j - 0.15] * len(lru_trials)
                jitter_p = [j + 0.15] * len(lpb_trials)
                ax.scatter(jitter_l, [v / lru_m for v in lru_trials],
                           color="#888", s=60, label="LRU" if j == 0 else None)
                ax.scatter(jitter_p, [v / lru_m for v in lpb_trials],
                           color="#3d8540", s=60, label="LPB" if j == 0 else None)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(["TTFT", "TPOT", "throughput"], fontsize=10)
            ax.set_ylabel("trial value / LRU mean")
            ax.set_title(title)
            ax.axhline(1.0, color="black", ls="--", lw=0.7, alpha=0.5)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(loc="upper right", fontsize=9)
        fig.suptitle("Per-trial dispersion — is the signal stable?", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out3 = FIGDIR / f"fig_lru_vs_lpb_trial_dispersion{suffix}.png"
        fig.savefig(out3, dpi=130)
        plt.close(fig)
        print(f"Wrote {out3}")

    # ----- JSON dump -----
    payload = {
        "n_trials_lru": n_trials_lru,
        "n_trials_lpb": n_trials_lpb,
        "trial_files": {
            "lru": [str(p) for p in lru_paths],
            "lpb": [str(p) for p in lpb_paths],
        },
        "lru": lru_agg,
        "lpb": lpb_agg,
    }
    out_json = Path(f"dev/compare_summary{suffix}.json")
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
