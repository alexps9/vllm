#!/usr/bin/env python3
"""Generate plots + summary table from sweep results.

Layout:
  <root>/<mode>/w1/turns_<T>/summary.json
  <root>/<mode>/w2/conc_<N>/result.json
  (mode ∈ {baseline, hima})

Usage:
  python plot.py <root> <out_dir>
"""
from __future__ import annotations
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def line2(ax, xs, yb, yh, title, ylabel, xlabel, logy=False, logx=False):
    if any(v is not None for v in yb): ax.plot(xs, yb, "-o", label="baseline", color="#888")
    if any(v is not None for v in yh): ax.plot(xs, yh, "-s", label="HiMA", color="#1f77b4")
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
    if logy: ax.set_yscale("log")
    if logx: ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3, linestyle="--"); ax.legend()


def plot_w1(root, out):
    b, h = load_w1(root, "baseline"), load_w1(root, "hima")
    turns = sorted({r["turns"] for r in b + h})
    if not turns: print("[w1] no data"); return
    mb, mh = {r["turns"]: r for r in b}, {r["turns"]: r for r in h}
    col = lambda m, k: [m.get(t, {}).get(k) for t in turns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    line2(axes[0,0], turns, col(mb,"server_prefix_cache_hit_rate"), col(mh,"server_prefix_cache_hit_rate"),
          "W1: server prefix-cache hit rate", "hits/queries", "turns/conv")
    line2(axes[0,1], turns, col(mb,"server_prompt_cached_ratio"), col(mh,"server_prompt_cached_ratio"),
          "W1: cached prompt-token ratio", "fraction", "turns/conv")
    line2(axes[0,2], turns, col(mb,"server_total_preemptions"), col(mh,"server_total_preemptions"),
          "W1: preemptions", "count", "turns/conv")
    line2(axes[1,0], turns, col(mb,"client_ttft_ms_p50"), col(mh,"client_ttft_ms_p50"),
          "W1: p50 TTFT", "ms", "turns/conv", logy=True)
    line2(axes[1,1], turns, col(mb,"client_ttft_ms_p95"), col(mh,"client_ttft_ms_p95"),
          "W1: p95 TTFT", "ms", "turns/conv", logy=True)
    line2(axes[1,2], turns, col(mb,"client_latency_ms_avg"), col(mh,"client_latency_ms_avg"),
          "W1: avg latency/turn", "ms", "turns/conv")
    fig.tight_layout()
    path = f"{out}/w1.png"; fig.savefig(path, dpi=130); plt.close(fig); print(f"wrote {path}")


def plot_w2(root, out):
    b, h = load_w2(root, "baseline"), load_w2(root, "hima")
    cs = sorted({r["concurrency"] for r in b + h})
    if not cs: print("[w2] no data"); return
    mb, mh = {r["concurrency"]: r for r in b}, {r["concurrency"]: r for r in h}
    col = lambda m, k: [m.get(c, {}).get(k) for c in cs]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    line2(axes[0,0], cs, col(mb,"overall_prefix_cache_hit_rate"), col(mh,"overall_prefix_cache_hit_rate"),
          "W2: prefix-cache hit rate", "hits/queries", "concurrent agents", logx=True)
    line2(axes[0,1], cs, col(mb,"overall_prompt_cached_ratio"), col(mh,"overall_prompt_cached_ratio"),
          "W2: cached prompt-token ratio", "fraction", "concurrent agents", logx=True)
    line2(axes[0,2], cs, col(mb,"overall_preemptions"), col(mh,"overall_preemptions"),
          "W2: preemptions", "count", "concurrent agents", logx=True)
    line2(axes[1,0], cs, col(mb,"p50_ttft_ms"), col(mh,"p50_ttft_ms"),
          "W2: p50 TTFT", "ms", "concurrent agents", logy=True, logx=True)
    line2(axes[1,1], cs, col(mb,"p95_ttft_ms"), col(mh,"p95_ttft_ms"),
          "W2: p95 TTFT", "ms", "concurrent agents", logy=True, logx=True)
    line2(axes[1,2], cs, col(mb,"output_throughput_tok_s"), col(mh,"output_throughput_tok_s"),
          "W2: output throughput", "tok/s", "concurrent agents", logx=True)
    fig.tight_layout()
    path = f"{out}/w2.png"; fig.savefig(path, dpi=130); plt.close(fig); print(f"wrote {path}")


def write_table(root, out):
    lines = ["# Results summary", ""]
    b1 = {r["turns"]: r for r in load_w1(root,"baseline")}
    h1 = {r["turns"]: r for r in load_w1(root,"hima")}
    lines += ["## W1 (multi-turn)", "",
              "| turns | mode | hit% | cached% | p50 TTFT | p95 TTFT | preempt | reqs |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for t in sorted(set(list(b1)+list(h1))):
        for tag, m in (("baseline", b1.get(t)), ("hima", h1.get(t))):
            if not m: continue
            lines.append(f"| {t} | {tag} | {100*m.get('server_prefix_cache_hit_rate',0):.1f}% |"
                         f" {100*m.get('server_prompt_cached_ratio',0):.1f}% |"
                         f" {m.get('client_ttft_ms_p50','?')} | {m.get('client_ttft_ms_p95','?')} |"
                         f" {m.get('server_total_preemptions','?')} | {m.get('num_requests_completed','?')} |")
    b2 = {r["concurrency"]: r for r in load_w2(root,"baseline")}
    h2 = {r["concurrency"]: r for r in load_w2(root,"hima")}
    lines += ["", "## W2 (concurrent agents)", "",
              "| conc | mode | hit% | cached% | p50 TTFT | p95 TTFT | tok/s | preempt | reqs |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(set(list(b2)+list(h2))):
        for tag, m in (("baseline", b2.get(c)), ("hima", h2.get(c))):
            if not m: continue
            lines.append(f"| {c} | {tag} | {100*m.get('overall_prefix_cache_hit_rate',0):.1f}% |"
                         f" {100*m.get('overall_prompt_cached_ratio',0):.1f}% |"
                         f" {m.get('p50_ttft_ms','?')} | {m.get('p95_ttft_ms','?')} |"
                         f" {m.get('output_throughput_tok_s','?')} | {m.get('overall_preemptions','?')} |"
                         f" {m.get('num_requests','?')} |")
    path = f"{out}/summary.md"
    open(path,"w").write("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_w1(args.root, args.out_dir)
    plot_w2(args.root, args.out_dir)
    write_table(args.root, args.out_dir)


if __name__ == "__main__":
    main()
