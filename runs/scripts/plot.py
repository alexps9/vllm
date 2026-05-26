#!/usr/bin/env python3
"""Generate plots + summary table from sweep results.

Layout:
  <root>/<mode>/w1/turns_<T>/summary.json
  <root>/<mode>/w2/conc_<N>/result.json

Usage:
  python plot.py <root> <out_dir> [mode1,mode2,...]
  (default modes: baseline,l1_only,l2_only,full — missing modes silently skipped)
"""
from __future__ import annotations
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "baseline": ("-o", "#888"),
    "l1_only":  ("-^", "#d62728"),
    "l2_only":  ("-v", "#2ca02c"),
    "full":     ("-s", "#1f77b4"),
}


def load_w1(root, mode):
    rows = []
    for p in sorted(glob.glob(f"{root}/{mode}/w1/turns_*/summary.json")):
        rows.append(json.load(open(p)))
    return sorted(rows, key=lambda r: r["turns"])


def load_w2(root, mode):
    rows = []
    for p in sorted(glob.glob(f"{root}/{mode}/w2/conc_*/result.json")):
        rows.append(json.load(open(p))["summary"])
    return sorted(rows, key=lambda r: r["concurrency"])


def line_multi(ax, xs, series, title, ylabel, xlabel, logy=False, logx=False):
    for mode, ys in series.items():
        if not any(v is not None for v in ys):
            continue
        marker, color = STYLE.get(mode, ("-x", None))
        ax.plot(xs, ys, marker, label=mode, color=color)
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
    if logy: ax.set_yscale("log")
    if logx: ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3, linestyle="--"); ax.legend()


def plot_w1(root, out, modes):
    data = {m: load_w1(root, m) for m in modes}
    turns = sorted({r["turns"] for rows in data.values() for r in rows})
    if not turns: print("[w1] no data"); return
    by_mode = {m: {r["turns"]: r for r in rows} for m, rows in data.items()}
    col = lambda key: {m: [by_mode[m].get(t, {}).get(key) for t in turns] for m in modes}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    line_multi(axes[0,0], turns, col("server_prefix_cache_hit_rate"),
               "W1: server prefix-cache hit rate", "hits/queries", "turns/conv")
    line_multi(axes[0,1], turns, col("server_prompt_cached_ratio"),
               "W1: cached prompt-token ratio", "fraction", "turns/conv")
    line_multi(axes[0,2], turns, col("server_total_preemptions"),
               "W1: preemptions", "count", "turns/conv")
    line_multi(axes[1,0], turns, col("client_ttft_ms_p50"),
               "W1: p50 TTFT", "ms", "turns/conv", logy=True)
    line_multi(axes[1,1], turns, col("client_ttft_ms_p95"),
               "W1: p95 TTFT", "ms", "turns/conv", logy=True)
    line_multi(axes[1,2], turns, col("client_latency_ms_avg"),
               "W1: avg latency/turn", "ms", "turns/conv")
    fig.tight_layout()
    path = f"{out}/w1.png"; fig.savefig(path, dpi=130); plt.close(fig); print(f"wrote {path}")


def plot_w2(root, out, modes):
    data = {m: load_w2(root, m) for m in modes}
    cs = sorted({r["concurrency"] for rows in data.values() for r in rows})
    if not cs: print("[w2] no data"); return
    by_mode = {m: {r["concurrency"]: r for r in rows} for m, rows in data.items()}
    col = lambda key: {m: [by_mode[m].get(c, {}).get(key) for c in cs] for m in modes}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    line_multi(axes[0,0], cs, col("overall_prefix_cache_hit_rate"),
               "W2: prefix-cache hit rate", "hits/queries", "concurrent agents", logx=True)
    line_multi(axes[0,1], cs, col("overall_prompt_cached_ratio"),
               "W2: cached prompt-token ratio", "fraction", "concurrent agents", logx=True)
    line_multi(axes[0,2], cs, col("overall_preemptions"),
               "W2: preemptions", "count", "concurrent agents", logx=True)
    line_multi(axes[1,0], cs, col("p50_ttft_ms"),
               "W2: p50 TTFT", "ms", "concurrent agents", logy=True, logx=True)
    line_multi(axes[1,1], cs, col("p95_ttft_ms"),
               "W2: p95 TTFT", "ms", "concurrent agents", logy=True, logx=True)
    line_multi(axes[1,2], cs, col("output_throughput_tok_s"),
               "W2: output throughput", "tok/s", "concurrent agents", logx=True)
    fig.tight_layout()
    path = f"{out}/w2.png"; fig.savefig(path, dpi=130); plt.close(fig); print(f"wrote {path}")


def write_table(root, out, modes):
    lines = ["# Results summary", ""]
    w1 = {m: {r["turns"]: r for r in load_w1(root, m)} for m in modes}
    lines += ["## W1 (multi-turn)", "",
              "| turns | mode | hit% | cached% | p50 TTFT | p95 TTFT | preempt | reqs |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    all_t = sorted({t for d in w1.values() for t in d})
    for t in all_t:
        for m in modes:
            r = w1[m].get(t)
            if not r: continue
            lines.append(f"| {t} | {m} | {100*r.get('server_prefix_cache_hit_rate',0):.1f}% |"
                         f" {100*r.get('server_prompt_cached_ratio',0):.1f}% |"
                         f" {r.get('client_ttft_ms_p50','?')} | {r.get('client_ttft_ms_p95','?')} |"
                         f" {r.get('server_total_preemptions','?')} | {r.get('num_requests_completed','?')} |")
    w2 = {m: {r["concurrency"]: r for r in load_w2(root, m)} for m in modes}
    lines += ["", "## W2 (concurrent agents)", "",
              "| conc | mode | hit% | cached% | p50 TTFT | p95 TTFT | tok/s | preempt | reqs |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    all_c = sorted({c for d in w2.values() for c in d})
    for c in all_c:
        for m in modes:
            r = w2[m].get(c)
            if not r: continue
            lines.append(f"| {c} | {m} | {100*r.get('overall_prefix_cache_hit_rate',0):.1f}% |"
                         f" {100*r.get('overall_prompt_cached_ratio',0):.1f}% |"
                         f" {r.get('p50_ttft_ms','?')} | {r.get('p95_ttft_ms','?')} |"
                         f" {r.get('output_throughput_tok_s','?')} | {r.get('overall_preemptions','?')} |"
                         f" {r.get('num_requests','?')} |")
    path = f"{out}/summary.md"
    open(path,"w").write("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_dir")
    ap.add_argument("modes", nargs="?", default="baseline,l1_only,l2_only,full",
                    help="comma-separated mode names (default: baseline,l1_only,l2_only,full)")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    plot_w1(args.root, args.out_dir, modes)
    plot_w2(args.root, args.out_dir, modes)
    write_table(args.root, args.out_dir, modes)


if __name__ == "__main__":
    main()
