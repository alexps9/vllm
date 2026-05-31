# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA runtime configuration (dependency-free; readable without torch/vllm)."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass


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


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class HiMAConfig:
    """HiMA knobs; load from ``VLLM_HIMA_*`` env vars via :meth:`from_env`.

    Scope: **L1 only** (LPB intra-pool eviction — queues + path counter).
    L2 (admitter / budgeter / cross-pool planner) was removed in 2026-05
    pending a from-scratch redesign; see ``dev/archive/L2/``.

    ``hima_enabled`` is a *derived* field (currently == ``hima_l1_enabled``)
    exposed so the engine bootstrap gate in ``vllm/v1/engine/core.py`` can
    read it directly. **Do not set it on the constructor.**
    """

    # ---- enabled flags ------------------------------------------- #
    # hima_enabled is derived in __post_init__; do not set it directly.
    hima_enabled: bool = False
    hima_l1_enabled: bool = False  # LPB queues + path counter

    # ---- L1 (intra-pool) ---------------------------------------- #
    hima_lpb_window_s: float = 60.0  # alias: VLLM_HIMA_HPB_WINDOW_S

    # ---- validation --------------------------------------------- #

    def __post_init__(self) -> None:
        if self.hima_lpb_window_s <= 0:
            raise ValueError(
                f"hima_lpb_window_s must be > 0, got {self.hima_lpb_window_s}"
            )

        # hima_enabled is derived from the L1 sub-flag — overwrite any
        # constructor-supplied value to maintain the invariant.
        if self.hima_enabled != self.hima_l1_enabled:
            object.__setattr__(self, "hima_enabled", self.hima_l1_enabled)

    # ---- env loader --------------------------------------------- #

    @classmethod
    def from_env(cls) -> HiMAConfig:
        """Load from the ``VLLM_HIMA_L1_ENABLE`` env var."""

        return cls(
            hima_l1_enabled=_env_bool("VLLM_HIMA_L1_ENABLE", False),
            hima_lpb_window_s=_env_float("VLLM_HIMA_HPB_WINDOW_S", 60.0),
        )
