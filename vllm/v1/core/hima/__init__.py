# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA -- Hierarchical Memory Management for hybrid models.

Scope: **L1 only** — LPB intra-pool eviction (LPB free-block queues + the
path-counted hit counter + cost curves). L2 (admitter / budgeter /
cross-pool planner / VMM actuator) was removed in 2026-05 — measured
neutral and slated for a from-scratch redesign. See ``dev/archive/L2/``.
"""

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.coordinator_hima import HiMACoordinator
from vllm.v1.core.hima.integration import (
    HiMARuntime,
    disable_runtime,
    enable_runtime,
    get_runtime,
    is_enabled,
    maybe_get_free_queue_factory,
    maybe_record_hit,
)
from vllm.v1.core.hima.intra_pool import (
    LEGACY_DEFAULT,
    BlockId,
    CostCurve,
    CostCurveCalibrator,
    CostCurveRegistry,
    CostCurves,
    LinearCostCurve,
    LPBPriorityQueue,
    PathCountedHitCounter,
    QuadraticCostCurve,
    get_cost_curves,
    hits_per_byte_score,
    lpb_score,
    reset_cost_curves,
)
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue

__all__ = [
    "LEGACY_DEFAULT",
    "BlockId",
    "CostCurve",
    "CostCurveCalibrator",
    "CostCurveRegistry",
    "CostCurves",
    "HiMACoordinator",
    "HiMAConfig",
    "HiMARuntime",
    "LPBFreeBlockQueue",
    "LPBPriorityQueue",
    "LinearCostCurve",
    "PathCountedHitCounter",
    "PoolKind",
    "QuadraticCostCurve",
    "disable_runtime",
    "enable_runtime",
    "get_cost_curves",
    "get_runtime",
    "hits_per_byte_score",
    "is_enabled",
    "lpb_score",
    "maybe_get_free_queue_factory",
    "maybe_record_hit",
    "reset_cost_curves",
]
