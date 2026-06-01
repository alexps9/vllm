# SPDX-License-Identifier: Apache-2.0
"""Apply the partial-block-waste formula to *real* Claude Code traces.

Source data: /data/yuzhou/projects/vllm-songyang/dev/intralayer/cc_long_traces.jsonl
  - 106 multi-turn sessions on hyperswitch (Rust), claude-code-style.
  - Each session: 16-397 user turns (median 46, p75 127).
  - Content is Anthropic-style structured blocks (text / tool_use /
    tool_result), flattened to strings for tokenization.

For each session we:
  1) Flatten messages to strings.
  2) Tokenize using Qwen3.5-35B's tokenizer (matches the model the inflate
     measurement was made on).
  3) Walk message-pair turns (user_i → asst_i counts as one turn).
  4) At each turn boundary, apply the formula
        expected_cache_hit = floor((prev_total_tokens - 1) / 1056) * 1056
     and compute partial_block_waste = prev_total_tokens - expected_cache_hit.

We then report:
  - Per-session: # turns, max context, avg waste/turn, waste / new content
  - Across all 106: distribution of waste %

The formula was empirically validated to be exact (100/100 turns) in
dev/multi_turn_waste.py on the same model. We do not need to re-run vLLM
to apply it to a new token sequence — only to verify the formula itself.

Run via:
  cd /data/yuzhou/projects/vllm-songyang
  .venv/bin/python dev/real_session_waste.py | tee dev/real_session_waste.out
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

DATA = Path("/data/yuzhou/projects/vllm-songyang/dev/intralayer/cc_long_traces.jsonl")
MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 1056  # observed inflated block_size (see dev/inspect_sizes.py)


def flatten_content(content) -> str:
    """Flatten one message's content (str | list[dict]) to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if isinstance(p, dict):
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
                    # Sometimes nested list of text blocks.
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
        else:
            parts.append(str(p))
    return "\n".join(parts)


def session_to_text(messages: list[dict]) -> str:
    """Render a session as a single concatenated string with role markers."""
    chunks: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        chunks.append(f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>")
    return "\n".join(chunks)


def cumulative_token_lengths(
    tokenizer, messages: list[dict]
) -> list[int]:
    """Return cumulative token count *after* each message.

    Tokenizes each message *independently* and sums. This is O(N) instead of
    O(N²) and the tokenization is essentially identical to "tokenize the
    whole concatenated string" because every message starts on a fresh
    <|im_start|>...<|im_end|> boundary that resets BPE state at a byte the
    tokenizer treats as a single token.
    """
    cumlen: list[int] = []
    running = 0
    for m in messages:
        role = m.get("role", "user")
        piece = (
            f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>\n"
        )
        # Approximate per-message tokenization (no special tokens added).
        running += len(tokenizer.encode(piece, add_special_tokens=False))
        cumlen.append(running)
    return cumlen


def main() -> None:
    print(f"Loading tokenizer for {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sessions: list[list[dict]] = []
    with DATA.open() as f:
        for line in f:
            sessions.append(json.loads(line)["messages"])
    print(f"Loaded {len(sessions)} sessions from {DATA.name}.")

    # Aggregate stats across sessions.
    per_session: list[dict] = []
    print("\nTokenizing & applying formula...")

    for s_idx, msgs in enumerate(sessions):
        cumlen = cumulative_token_lengths(tokenizer, msgs)
        # User-turn boundaries: turn i ends after assistant_i (= msg index 2i+1).
        # If msgs alternate user/asst strictly, turn boundaries are odd indices.
        turn_ends: list[int] = []  # cumulative tokens after each turn
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant":
                turn_ends.append(cumlen[i])

        if len(turn_ends) < 2:
            continue  # too short to have a "next-turn" measurement

        # Per-turn waste: at turn N's *next* request issuance,
        # expected cache hit = floor((turn_ends[N-1] - 1) / 1056) * 1056.
        waste_per_turn: list[int] = []
        new_content_per_turn: list[int] = []
        for n in range(1, len(turn_ends)):
            prev = turn_ends[n - 1]
            expected = ((prev - 1) // BLOCK_SIZE) * BLOCK_SIZE
            waste = max(prev - expected, 0)
            waste_per_turn.append(waste)
            new_content_per_turn.append(turn_ends[n] - turn_ends[n - 1])

        total_waste = sum(waste_per_turn)
        total_new = sum(new_content_per_turn)
        waste_pct = (total_waste / total_new * 100) if total_new > 0 else 0
        per_session.append({
            "idx": s_idx,
            "n_turns": len(turn_ends),
            "max_ctx": turn_ends[-1],
            "p50_new_tokens_per_turn": int(statistics.median(new_content_per_turn)),
            "mean_new_per_turn": int(statistics.mean(new_content_per_turn)),
            "total_new_content": total_new,
            "total_waste": total_waste,
            "avg_waste_per_turn": int(statistics.mean(waste_per_turn)),
            "waste_pct": waste_pct,
        })

        if s_idx % 25 == 0:
            print(f"  [{s_idx + 1}/{len(sessions)}] turns={len(turn_ends):>4} "
                  f"max_ctx={turn_ends[-1]:>7} waste_pct={waste_pct:.1f}%")

    # ---- per-session report ----
    print("\n" + "=" * 90)
    print("PER-SESSION (first 20)")
    print("=" * 90)
    print(f"  {'idx':>3} | {'turns':>5} | {'max_ctx':>8} | "
          f"{'p50_new':>7} | {'avg_waste':>9} | {'waste_pct':>10}")
    print("  " + "-" * 88)
    for r in per_session[:20]:
        print(f"  {r['idx']:>3} | {r['n_turns']:>5} | {r['max_ctx']:>8} | "
              f"{r['p50_new_tokens_per_turn']:>7} | {r['avg_waste_per_turn']:>9} | "
              f"{r['waste_pct']:>9.1f}%")

    # ---- aggregate ----
    print("\n" + "=" * 90)
    print("AGGREGATE (across all sessions)")
    print("=" * 90)

    def q(xs, p):
        return statistics.quantiles(xs, n=100)[int(p) - 1]

    keys = [
        ("n_turns", "{:>6}"),
        ("max_ctx", "{:>9}"),
        ("p50_new_tokens_per_turn", "{:>9}"),
        ("avg_waste_per_turn", "{:>9}"),
        ("waste_pct", "{:>8.1f}%"),
    ]
    print(f"  {'metric':<26} {'min':>10} {'p25':>10} {'p50':>10} "
          f"{'p75':>10} {'p95':>10} {'max':>10}")
    print("  " + "-" * 90)
    for k, fmt in keys:
        vs = sorted(r[k] for r in per_session)
        row = [min(vs), q(vs, 25), q(vs, 50), q(vs, 75), q(vs, 95), max(vs)]
        print(f"  {k:<26} " + " ".join(fmt.format(v) for v in row))

    # Workload-weighted waste pct (sum of waste / sum of new content):
    total_w = sum(r["total_waste"] for r in per_session)
    total_n = sum(r["total_new_content"] for r in per_session)
    workload_pct = total_w / total_n * 100 if total_n else 0.0
    print(f"\n  Workload-weighted waste pct (Σwaste / Σnew_content): "
          f"{workload_pct:.2f}%")
    print(f"  Total partial-block waste over all sessions: {total_w:,} tokens")
    print(f"  Total new content tokens over all sessions:  {total_n:,} tokens")


if __name__ == "__main__":
    main()
