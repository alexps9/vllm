"""Phase 2 verifier: read smoke_*.out files, grep for HiMA log markers
produced by the EngineCore subprocess, and verify they match each config's
expectations.

Markers searched for (emitted from production code):
  * ``HiMA enabled (page=`` — from vllm/v1/engine/core.py:168 when HiMA on
  * ``HiMA budgeter task started`` — from vllm/v1/core/hima/budgeter_task.py:56
    when L2 enabled and daemon spawned
  * ``HiMA L2 disabled; budgeter daemon not started.`` —
    vllm/v1/core/hima/budgeter_task.py:93 when L1-only

Per config, the expected markers are:

    lru:      no HiMA marker at all
    l1_only:  "HiMA enabled (page=" present
              "HiMA L2 disabled" present
              "HiMA budgeter task started" absent
    l2_only:  "HiMA enabled (page=" present
              "HiMA budgeter task started" present
              "HiMA L2 disabled" absent
    full:     "HiMA enabled (page=" present
              "HiMA budgeter task started" present
              "HiMA L2 disabled" absent

Usage:
    .venv/bin/python smoke_l1l2_verify.py runs/smoke_lru.out runs/smoke_l1_only.out ...
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CONFIG_RE = re.compile(r"^SMOKE_CONFIG_TAG (\w+)$", re.MULTILINE)


@dataclass
class Markers:
    hima_enabled_line: bool   # "HiMA enabled (page=" appears
    budgeter_started: bool    # "HiMA budgeter task started"
    l2_disabled_log: bool     # "HiMA L2 disabled; budgeter daemon not started."
    boot_done: bool
    gen_done: bool


EXPECTED = {
    "lru":     {"hima_enabled_line": False, "budgeter_started": False, "l2_disabled_log": False},
    "l1_only": {"hima_enabled_line": True,  "budgeter_started": False, "l2_disabled_log": True},
    "l2_only": {"hima_enabled_line": True,  "budgeter_started": True,  "l2_disabled_log": False},
    "full":    {"hima_enabled_line": True,  "budgeter_started": True,  "l2_disabled_log": False},
}


def parse(path: Path) -> tuple[str | None, Markers]:
    txt = path.read_text()
    m = CONFIG_RE.search(txt)
    config = m.group(1) if m else None
    markers = Markers(
        hima_enabled_line="HiMA enabled (page=" in txt,
        budgeter_started="HiMA budgeter task started" in txt,
        l2_disabled_log="HiMA L2 disabled; budgeter daemon not started." in txt,
        boot_done="SMOKE_BOOT_DONE" in txt,
        gen_done="SMOKE_GEN_DONE" in txt,
    )
    return config, markers


def verify_one(path: Path) -> dict[str, object]:
    config, markers = parse(path)
    if config is None:
        return {"path": str(path), "verdict": "FAIL", "reason": "no SMOKE_CONFIG_TAG"}
    if config not in EXPECTED:
        return {"path": str(path), "verdict": "FAIL", "reason": f"unknown config {config!r}"}
    exp = EXPECTED[config]
    mismatches: list[str] = []
    for key, want in exp.items():
        got = getattr(markers, key)
        if got != want:
            mismatches.append(f"{key}: expected={want} got={got}")
    if not markers.boot_done:
        mismatches.append("LLM never finished booting (no SMOKE_BOOT_DONE)")
    if not markers.gen_done:
        mismatches.append("LLM never finished generation (no SMOKE_GEN_DONE)")
    return {
        "path": str(path),
        "config": config,
        "markers": markers.__dict__,
        "mismatches": mismatches,
        "verdict": "PASS" if not mismatches else "FAIL",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: smoke_l1l2_verify.py <out_files...>", file=sys.stderr)
        return 2
    all_pass = True
    for arg in sys.argv[1:]:
        result = verify_one(Path(arg))
        print(json.dumps(result))
        if result["verdict"] != "PASS":
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
