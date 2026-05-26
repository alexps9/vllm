"""
W2: concurrent SWE-Bench-Lite agent driver (no real shell execution).

Replays princeton-nlp/SWE-Bench_Lite problem statements through a vLLM
OpenAI endpoint as K-turn agents with mock tool observations.
Snapshots Prometheus /metrics before/after each concurrency level.

Output: <out-dir>/conc_N/result.json per concurrency level.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field

from datasets import load_dataset
from openai import AsyncOpenAI

# --- mini-swe-agent templates (verbatim from swebench.yaml) ---
SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer "
    "shell to solve programming tasks.\n"
)

INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)
"""

# Mock observations of varying length / structure to simulate
# realistic bash output (`ls`, `find`, `cat`, `pytest`, `git diff`,
# `grep`, `python -c`, `pip install`). Each block is ~1.5–3 KB which
# is much closer to real mini-swe-agent telemetry than the previous
# tiny stubs and ensures every turn appends ~500–1000 fresh tokens
# to the agent context.
def _ls_block() -> str:
    rows = []
    names = [
        "src", "tests", "docs", "scripts", "examples", "benchmarks",
        "third_party", "tools", "vendor", "data",
    ]
    for i, n in enumerate(names):
        rows.append(
            f"drwxr-xr-x {i+3:3d} root root {4096+i*8:>6} Jan  {i%28+1:02d} 00:00 {n}"
        )
    files = [
        ("README.md", 8192), ("CHANGELOG.md", 3072), ("LICENSE", 1024),
        ("setup.py", 4096), ("setup.cfg", 1536), ("pyproject.toml", 2048),
        ("MANIFEST.in", 256), ("tox.ini", 768), ("conftest.py", 1024),
        (".pre-commit-config.yaml", 1280), (".gitignore", 512),
        ("Makefile", 1024), ("Dockerfile", 1536),
    ]
    for fn, sz in files:
        rows.append(f"-rw-r--r--  1 root root {sz:>6} Jan  1 00:00 {fn}")
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(rows) + "\n</output>\n"


def _find_block() -> str:
    paths = []
    for mod in ("module", "utils", "cli", "api", "handlers", "state",
                "exceptions", "io", "validators", "serializers", "registry",
                "engine", "parser", "lexer", "transforms", "compiler"):
        paths += [
            f"src/{mod}.py",
            f"src/{mod}_impl.py",
            f"tests/test_{mod}.py",
            f"tests/test_{mod}_edge.py",
        ]
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(paths) + "\n</output>\n"


def _cat_block() -> str:
    body = [
        "from __future__ import annotations",
        "import logging",
        "from dataclasses import dataclass, field",
        "from typing import Any, Iterable, Mapping, Optional, Sequence",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "@dataclass",
        "class Config:",
        "    name: str",
        "    enabled: bool = True",
        "    payload: Mapping[str, Any] = field(default_factory=dict)",
        "",
        "    def merged(self, other: 'Config') -> 'Config':",
        "        if other is None:",
        "            return self",
        "        merged_payload = {**self.payload, **other.payload}",
        "        return Config(self.name, self.enabled and other.enabled, merged_payload)",
        "",
        "def compute_value(x, y, scale: float = 1.0) -> float:",
        "    if x is None or y is None:",
        "        raise ValueError('x and y must not be None')",
        "    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):",
        "        raise TypeError('x and y must be numeric')",
        "    return (x + y) * scale",
        "",
        "def normalize(seq: Sequence[float]) -> list[float]:",
        "    if not seq:",
        "        return []",
        "    total = float(sum(seq))",
        "    if total == 0.0:",
        "        return [0.0] * len(seq)",
        "    return [float(v) / total for v in seq]",
        "",
        "def iter_chunks(data: Iterable[int], size: int) -> Iterable[list[int]]:",
        "    buf: list[int] = []",
        "    for v in data:",
        "        buf.append(v)",
        "        if len(buf) >= size:",
        "            yield buf",
        "            buf = []",
        "    if buf:",
        "        yield buf",
        "",
        "def safe_get(mapping: Mapping[str, Any], key: str, default: Optional[Any] = None) -> Any:",
        "    if not isinstance(mapping, Mapping):",
        "        return default",
        "    value = mapping.get(key, default)",
        "    if value is None:",
        "        return default",
        "    return value",
    ]
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(body) + "\n</output>\n"


def _pytest_block() -> str:
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.4, pytest-8.3.4, pluggy-1.5.0",
        "rootdir: /testbed",
        "configfile: pyproject.toml",
        "plugins: hypothesis-6.110.1, anyio-4.6.0, cov-5.0.0",
        "collected 142 items",
        "",
    ]
    for mod in ("module", "utils", "cli", "api", "handlers", "state",
                "exceptions", "registry"):
        for i in range(8):
            mark = "PASSED" if i < 7 else "FAILED"
            lines.append(
                f"tests/test_{mod}.py::test_{mod}_case_{i:02d} {mark}     [{(i+1)*7:>3}%]"
            )
    lines += [
        "",
        "=================================== FAILURES ===================================",
        "_____________________ test_module_case_07 ____________________",
        "    def test_module_case_07():",
        ">       assert compute_value(None, 3) == 3",
        "E       ValueError: x and y must not be None",
        "tests/test_module.py:42: ValueError",
        "===================== 1 failed, 141 passed in 4.83s ============================",
    ]
    return "<returncode>1</returncode>\n<output>\n" + "\n".join(lines) + "\n</output>\n"


def _git_diff_block() -> str:
    lines = [
        "diff --git a/src/module.py b/src/module.py",
        "index 1234abc..5678def 100644",
        "--- a/src/module.py",
        "+++ b/src/module.py",
        "@@ -38,9 +38,15 @@ def compute_value(x, y, scale=1.0):",
        "-    if x is None or y is None:",
        "+    if x is None and y is None:",
        "+        logger.warning('both inputs None — returning 0.0')",
        "+        return 0.0",
        "+    if x is None or y is None:",
        "         raise ValueError('x and y must not be None')",
        "     return (x + y) * scale",
        "",
        "diff --git a/tests/test_module.py b/tests/test_module.py",
        "--- a/tests/test_module.py",
        "+++ b/tests/test_module.py",
        "@@ -10,6 +10,11 @@ def test_compute_value_basic():",
        "     assert compute_value(1, 2) == 3",
        "+",
        "+def test_compute_value_both_none():",
        "+    # added regression for issue #1234",
        "+    assert compute_value(None, None) == 0.0",
    ]
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(lines) + "\n</output>\n"


def _grep_block() -> str:
    lines = []
    for mod in ("module", "utils", "handlers", "state", "registry"):
        for i in range(10):
            lines.append(
                f"src/{mod}.py:{20 + i*7}:    if value is None or value == '':"
            )
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(lines) + "\n</output>\n"


def _py_inline_block() -> str:
    lines = [
        "Python 3.12.4 (main, Aug  1 2024, 12:34:56) on linux",
        "Type 'help', 'copyright', 'credits' or 'license' for more information.",
        ">>> from src.module import compute_value",
        ">>> compute_value(1, 2)",
        "3",
        ">>> compute_value(None, 5)",
        "Traceback (most recent call last):",
        "  File '<stdin>', line 1, in <module>",
        "  File '/testbed/src/module.py', line 42, in compute_value",
        "    if x is None or y is None: raise ValueError(...)",
        "ValueError: x and y must not be None",
        ">>> compute_value(None, None)",
        "Traceback (most recent call last):",
        "ValueError: x and y must not be None",
    ]
    return "<returncode>1</returncode>\n<output>\n" + "\n".join(lines) + "\n</output>\n"


def _pip_block() -> str:
    pkgs = [
        ("attrs", "24.2.0"), ("certifi", "2024.7.4"), ("charset-normalizer", "3.3.2"),
        ("click", "8.1.7"), ("filelock", "3.16.1"), ("fsspec", "2024.9.0"),
        ("idna", "3.10"), ("iniconfig", "2.0.0"), ("Jinja2", "3.1.4"),
        ("MarkupSafe", "2.1.5"), ("numpy", "1.26.4"), ("packaging", "24.1"),
        ("pluggy", "1.5.0"), ("pyparsing", "3.1.4"), ("pytest", "8.3.4"),
        ("PyYAML", "6.0.2"), ("requests", "2.32.3"), ("setuptools", "75.1.0"),
        ("typing_extensions", "4.12.2"), ("urllib3", "2.2.3"), ("wheel", "0.44.0"),
    ]
    lines = ["Package                  Version", "------------------------ ----------"]
    for n, v in pkgs:
        lines.append(f"{n:<24} {v}")
    return "<returncode>0</returncode>\n<output>\n" + "\n".join(lines) + "\n</output>\n"


MOCK_OBSERVATIONS = [
    _ls_block(), _find_block(), _cat_block(), _pytest_block(),
    _git_diff_block(), _grep_block(), _py_inline_block(), _pip_block(),
]


TRACKED = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def scrape(url: str) -> dict[str, float]:
    out: dict[str, float] = {k: 0.0 for k in TRACKED}
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in TRACKED:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] += value
    return out


def reset_cache(base_url: str) -> None:
    reset = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/reset_prefix_cache"
    try:
        req = urllib.request.Request(reset, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception:
        pass


@dataclass
class TurnRecord:
    agent: int
    instance_id: str
    turn: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0


async def streamed_turn(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[float, float, int, int, str]:
    t0 = time.monotonic()
    ttft = None
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.monotonic() - t0
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
            completion_tokens = chunk.usage.completion_tokens or completion_tokens
    total_ms = (time.monotonic() - t0) * 1000.0
    ttft_ms = (ttft or 0.0) * 1000.0
    return ttft_ms, total_ms, prompt_tokens, completion_tokens, "".join(chunks)


async def run_one_agent(
    *,
    agent_id: int,
    instance: dict,
    client: AsyncOpenAI,
    model: str,
    num_turns: int,
    max_tokens: int,
    max_prompt_tokens: int,
    start_delay_s: float = 0.0,
) -> list[TurnRecord]:
    if start_delay_s > 0:
        await asyncio.sleep(start_delay_s)
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": INSTANCE_TEMPLATE.format(task=instance["problem_statement"]),
        },
    ]
    records: list[TurnRecord] = []
    stopped_early = False
    for turn in range(num_turns):
        try:
            ttft_ms, total_ms, p_tok, c_tok, reply = await streamed_turn(
                client, model, history, max_tokens
            )
        except Exception as exc:
            # Most commonly: context-window exhaustion. Stop this agent.
            print(f"[agent {agent_id:>3} turn {turn:>2}] {type(exc).__name__}: {exc}")
            stopped_early = True
            break
        records.append(
            TurnRecord(
                agent=agent_id,
                instance_id=instance["instance_id"],
                turn=turn,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                ttft_ms=round(ttft_ms, 1),
                total_ms=round(total_ms, 1),
            )
        )
        # If we're close to the model's context window, stop voluntarily so
        # the next streamed_turn doesn't blow up with a 400.
        if p_tok + c_tok >= max_prompt_tokens:
            stopped_early = True
            break
        history.append({"role": "assistant", "content": reply})
        if turn < num_turns - 1:
            obs = MOCK_OBSERVATIONS[turn % len(MOCK_OBSERVATIONS)]
            history.append({"role": "user", "content": obs})
    if stopped_early and records:
        records[-1] = TurnRecord(**{**records[-1].__dict__})
    return records


async def run_concurrency(
    *,
    concurrency: int,
    instances: list[dict],
    base_url: str,
    metrics_url: str,
    model: str,
    num_turns: int,
    max_tokens: int,
    max_prompt_tokens: int,
    ramp_s: float,
    out_path: str,
) -> dict:
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=300.0)
    await asyncio.sleep(0.2)
    snapshot_start = scrape(metrics_url)

    t0 = time.monotonic()
    tasks = [
        run_one_agent(
            agent_id=i,
            instance=instances[i % len(instances)],
            client=client,
            model=model,
            num_turns=num_turns,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
            # ramp first ~min(concurrency, 32) agents over `ramp_s` so the
            # cache for the unique instance prefixes has time to materialise
            # before the high-conc burst hits.
            start_delay_s=(i * ramp_s),
        )
        for i in range(concurrency)
    ]
    all_records_nested = await asyncio.gather(*tasks)
    total_seconds = time.monotonic() - t0

    snapshot_end = scrape(metrics_url)
    flat = [r.__dict__ for sub in all_records_nested for r in sub]

    overall_q = snapshot_end["vllm:prefix_cache_queries_total"] - snapshot_start["vllm:prefix_cache_queries_total"]
    overall_h = snapshot_end["vllm:prefix_cache_hits_total"] - snapshot_start["vllm:prefix_cache_hits_total"]
    overall_pt = snapshot_end["vllm:prompt_tokens_total"] - snapshot_start["vllm:prompt_tokens_total"]
    overall_pc = snapshot_end["vllm:prompt_tokens_cached_total"] - snapshot_start["vllm:prompt_tokens_cached_total"]
    overall_preempt = snapshot_end["vllm:num_preemptions_total"] - snapshot_start["vllm:num_preemptions_total"]
    kv_usage_end = snapshot_end["vllm:kv_cache_usage_perc"]

    by_turn: dict[int, list[dict]] = {}
    for r in flat:
        by_turn.setdefault(r["turn"], []).append(r)

    def pct(xs, p):
        if not xs: return None
        xs = sorted(xs); k = int(p / 100 * (len(xs) - 1)); return xs[k]

    turn_summary = []
    for t in sorted(by_turn):
        recs = by_turn[t]
        ttfts = [r["ttft_ms"] for r in recs]
        prompts = [r["prompt_tokens"] for r in recs]
        completions = [r["completion_tokens"] for r in recs]
        turn_summary.append({
            "turn": t,
            "n": len(recs),
            "avg_prompt_tokens": round(sum(prompts) / len(prompts), 1),
            "avg_completion_tokens": round(sum(completions) / len(completions), 1),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1),
            "p50_ttft_ms": pct(ttfts, 50),
            "p95_ttft_ms": pct(ttfts, 95),
        })

    all_ttft = [r["ttft_ms"] for r in flat]
    total_input = sum(r["prompt_tokens"] for r in flat)
    total_output = sum(r["completion_tokens"] for r in flat)

    result = {
        "concurrency": concurrency,
        "num_turns": num_turns,
        "max_tokens": max_tokens,
        "n_unique_instances": min(concurrency, len(instances)),
        "runtime_sec": round(total_seconds, 2),
        "num_requests": len(flat),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "output_throughput_tok_s": round(total_output / total_seconds, 1),
        "overall_prefix_cache_hit_rate": round(overall_h / overall_q, 4) if overall_q else 0.0,
        "overall_prompt_cached_ratio": round(overall_pc / overall_pt, 4) if overall_pt else 0.0,
        "overall_prompt_tokens": overall_pt,
        "overall_prompt_tokens_cached": overall_pc,
        "overall_preemptions": overall_preempt,
        "final_kv_usage_perc": round(kv_usage_end, 4),
        "avg_ttft_ms": round(sum(all_ttft) / len(all_ttft), 1) if all_ttft else 0.0,
        "p50_ttft_ms": pct(all_ttft, 50),
        "p95_ttft_ms": pct(all_ttft, 95),
        "p99_ttft_ms": pct(all_ttft, 99),
        "by_turn": turn_summary,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": result, "per_request": flat}, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


def load_swe_instances(n_instances: int, seed: int = 42) -> list[dict]:
    ds = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picked = [ds[i] for i in indices[:n_instances]]
    return [
        {
            "instance_id": x["instance_id"],
            "repo": x["repo"],
            "problem_statement": x["problem_statement"],
        }
        for x in picked
    ]


async def main_async(args):
    instances = load_swe_instances(args.n_instances, seed=args.seed)
    print(f"[mini-swe] loaded {len(instances)} SWE-Bench-Lite instances")
    for c in args.concurrencies:
        sub = os.path.join(args.out_dir, f"conc_{c}")
        os.makedirs(sub, exist_ok=True)
        out_path = os.path.join(sub, "result.json")
        # cap ramp at 8s total to keep aggregate runtime sane.
        ramp_per_agent = min(args.ramp_s, 8.0 / max(c, 1))
        print(
            f"\n==== [W2] concurrency={c} turns={args.num_turns} "
            f"max_tokens={args.max_tokens} ramp/agent={ramp_per_agent:.2f}s ===="
        )
        await run_concurrency(
            concurrency=c,
            instances=instances,
            base_url=args.base_url,
            metrics_url=args.metrics_url,
            model=args.model,
            num_turns=args.num_turns,
            max_tokens=args.max_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            ramp_s=ramp_per_agent,
            out_path=out_path,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    ap.add_argument("--num-turns", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-instances", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-prompt-tokens", type=int, default=60000,
        help="Safety stop when an agent's prompt approaches the model "
             "context window. Default = 60000 (server max-model-len=65536).",
    )
    ap.add_argument(
        "--ramp-s", type=float, default=0.3,
        help="Per-agent launch stagger in seconds. Helps the shared system "
             "prefix get cached before the burst.",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
