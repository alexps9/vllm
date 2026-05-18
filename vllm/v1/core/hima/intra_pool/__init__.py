# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intra-pool HiMA components."""

from vllm.v1.core.hima.intra_pool.cost_curve import (
    LEGACY_DEFAULT,
    CostCurve,
    CostCurveCalibrator,
    CostCurveRegistry,
    CostCurves,
    LinearCostCurve,
    QuadraticCostCurve,
    RuntimeActuatorCost,
    get_cost_curves,
    get_runtime_actuator_cost,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from vllm.v1.core.hima.intra_pool.lpb_queue import LPBPriorityQueue
from vllm.v1.core.hima.intra_pool.path_count import (
    BlockId,
    PathCountedHitCounter,
)


def lpb_score(
    *,
    hits: int,
    recovery_length: int,
    block_bytes: int,
    cost_curve: CostCurve,
) -> float:
    """Loss-per-byte score ``l(b) = n_b * c_i(s_b) / B_b`` (paper).

    Lower scores are evicted first.
    """

    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be > 0, got {block_bytes}")
    if hits < 0 or recovery_length < 0:
        raise ValueError("hits and recovery_length must be >= 0")
    return hits * cost_curve.cost(recovery_length) / block_bytes


def hits_per_byte_score(*, hits: int, block_bytes: int) -> float:
    """sglang-style HPB: ``hits / bytes``, no cost weighting.

    Equivalent to ``lpb_score`` with a constant ``c(s) = 1``. Provided
    for parity with ``sglang.srt.mem_cache.mamba_radix_cache.TreeNode.
    eviction_priority`` so the two scoring schemes can be A/B'd at
    runtime.
    """

    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be > 0, got {block_bytes}")
    if hits < 0:
        raise ValueError(f"hits must be >= 0, got {hits}")
    return hits / block_bytes


__all__ = [
    "BlockId",
    "CostCurve",
    "CostCurveCalibrator",
    "CostCurveRegistry",
    "CostCurves",
    "LEGACY_DEFAULT",
    "LPBPriorityQueue",
    "LinearCostCurve",
    "PathCountedHitCounter",
    "QuadraticCostCurve",
    "RuntimeActuatorCost",
    "get_cost_curves",
    "get_runtime_actuator_cost",
    "hits_per_byte_score",
    "lpb_score",
    "reset_cost_curves",
    "reset_runtime_actuator_cost",
]
