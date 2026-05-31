# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA integration façade -- single seam between vLLM hot paths and HiMA L1.

Hot paths call ``get_runtime()`` (returns ``None`` when HiMA is off).
HiMA is disabled by default; the runtime is built once via ``enable_runtime()``.

Scope note: this façade now exposes **L1 only** (LPB intra-pool eviction:
the path-counted hit counter + the LPB free-block queues). L2 (admitter /
budgeter / cross-pool planner / VMM actuator) was removed in 2026-05 — it
was measured neutral (≈ LRU) and is slated for a from-scratch redesign.
Archived design notes: ``dev/archive/L2/``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.intra_pool import (
    CostCurves,
    LPBPriorityQueue,
    PathCountedHitCounter,
    get_cost_curves,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class HiMARuntime:
    """Process-global container for HiMA L1 components.

    Built once at engine start via :func:`enable_runtime`.
    """

    config: HiMAConfig
    cost_curves: CostCurves
    path_counter: PathCountedHitCounter
    intra_queues: dict[PoolKind, LPBPriorityQueue] = field(default_factory=dict)

    # LPBFreeBlockQueue instances registered by BlockPool (one per group).
    _lpb_queues: list[Any] = field(default_factory=list)

    # --------------- intra-pool helpers (used by BlockPool) --------------- #

    def get_or_create_queue(self, pool: PoolKind) -> LPBPriorityQueue:
        q = self.intra_queues.get(pool)
        if q is None:
            q = LPBPriorityQueue()
            self.intra_queues[pool] = q
        return q

    def register_lpb_queue(self, queue: Any) -> None:
        """Register a LPBFreeBlockQueue so record_hit can push depth updates."""
        if queue not in self._lpb_queues:
            self._lpb_queues.append(queue)

    def record_hit(self, path_block_ids: list[int]) -> None:
        """Record a prefix-cache hit; update path-counts and LPB depths."""
        self.path_counter.record_hit(tuple(path_block_ids))
        # Push depth info into every registered LPBFreeBlockQueue. The
        # 'eager' scoring variant additionally re-scores the block now that
        # its hit count just incremented (verify/4).
        for depth, block_id in enumerate(path_block_ids, start=1):
            for q in self._lpb_queues:
                q.set_block_depth(block_id, depth)
                q.maybe_eager_refresh(block_id)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: HiMARuntime | None = None


def is_enabled() -> bool:
    """Returns ``False`` until :func:`enable_runtime` is called."""
    return _RUNTIME is not None


def get_runtime() -> HiMARuntime | None:
    """Return the active runtime, or ``None`` when HiMA is disabled."""
    return _RUNTIME


def enable_runtime(
    config: HiMAConfig | None = None,
) -> HiMARuntime:
    """Build (or return existing) the HiMA L1 runtime."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        cfg = config if config is not None else HiMAConfig.from_env()
        curves = get_cost_curves()
        path_counter = PathCountedHitCounter(window_seconds=cfg.hima_lpb_window_s)

        runtime = HiMARuntime(
            config=cfg,
            cost_curves=curves,
            path_counter=path_counter,
        )
        _RUNTIME = runtime
        return runtime


def disable_runtime() -> None:
    """Tear down the runtime. Test-only / shutdown helper."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None


# ---------------------------------------------------------------------------
# Hot-path helpers (thin & cheap)
# ---------------------------------------------------------------------------


def maybe_record_hit(path_block_ids: list[int]) -> None:
    """Hot-path-safe path-counted hit recording. No-op when disabled."""
    if _RUNTIME is None:
        return
    _RUNTIME.record_hit(path_block_ids)


def maybe_get_free_queue_factory() -> Any | None:
    """Returns the LPBFreeBlockQueue class when HiMA L1 is on, else ``None``."""
    if _RUNTIME is None or not _RUNTIME.config.hima_l1_enabled:
        return None
    from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue  # noqa: PLC0415

    return LPBFreeBlockQueue


__all__ = [
    "HiMARuntime",
    "disable_runtime",
    "enable_runtime",
    "get_runtime",
    "is_enabled",
    "maybe_get_free_queue_factory",
    "maybe_record_hit",
]
