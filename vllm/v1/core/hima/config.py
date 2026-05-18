# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA runtime configuration (dependency-free; readable without torch/vllm)."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field


class PoolKind(enum.Enum):
    """Two memory pools: KV-cache (``kv``) and recurrent-state snapshots (``rec``)."""

    KV = "kv"
    REC = "rec"

    def other(self) -> PoolKind:
        return PoolKind.REC if self is PoolKind.KV else PoolKind.KV


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_optional_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


@dataclass(frozen=True)
class HiMAConfig:
    """HiMA knobs; load from ``VLLM_HIMA_*`` env vars via :meth:`from_env`."""

    # ---- master switch & physical-page params -------------------- #
    hima_enabled: bool = False
    hima_page_size_bytes: int | None = None  # None => probe via cuMemGet...

    # ---- L1 (intra-pool) ---------------------------------------- #
    hima_lpb_window_s: float = 60.0  # alias: VLLM_HIMA_HPB_WINDOW_S
    hima_cost_kv_alpha: float | None = None  # legacy single-shape alpha
    hima_cost_rec_alpha: float | None = None

    # ---- L2 (inter-pool) ---------------------------------------- #
    hima_budget_interval_s: float = 30.0  # Budgeter / planner tick period
    hima_queue_penalty: float = 1.0  # Admitter w_q (paper §3.3)

    # ---- Telemetry ---------------------------------------------- #
    ewma_alpha: float = 0.2

    extra: dict[str, float] = field(default_factory=dict)

    # ---- validation --------------------------------------------- #

    def __post_init__(self) -> None:
        if self.hima_page_size_bytes is not None and self.hima_page_size_bytes <= 0:
            raise ValueError(
                f"hima_page_size_bytes must be positive or None, "
                f"got {self.hima_page_size_bytes}"
            )
        if self.hima_budget_interval_s <= 0:
            raise ValueError(
                f"hima_budget_interval_s must be > 0, got {self.hima_budget_interval_s}"
            )
        if self.hima_queue_penalty < 0:
            raise ValueError(
                f"hima_queue_penalty must be >= 0, got {self.hima_queue_penalty}"
            )
        if self.hima_lpb_window_s <= 0:
            raise ValueError(
                f"hima_lpb_window_s must be > 0, got {self.hima_lpb_window_s}"
            )
        for name, value in (
            ("hima_cost_kv_alpha", self.hima_cost_kv_alpha),
            ("hima_cost_rec_alpha", self.hima_cost_rec_alpha),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when set, got {value}")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")

    # ---- env loader --------------------------------------------- #

    @classmethod
    def from_env(cls) -> HiMAConfig:
        """Load from ``VLLM_HIMA_*`` env vars."""

        return cls(
            hima_enabled=_env_bool("VLLM_HIMA_ENABLE", False),
            hima_page_size_bytes=_env_optional_int("VLLM_HIMA_PAGE_SIZE_BYTES"),
            hima_lpb_window_s=_env_float("VLLM_HIMA_HPB_WINDOW_S", 60.0),
            hima_budget_interval_s=_env_float("VLLM_HIMA_BUDGETER_TICK_S", 30.0),
            hima_queue_penalty=_env_float("VLLM_HIMA_QUEUE_PENALTY", 1.0),
            ewma_alpha=_env_float("VLLM_HIMA_EWMA_ALPHA", 0.2),
        )
