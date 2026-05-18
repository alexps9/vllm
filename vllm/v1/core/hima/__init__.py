# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA -- Hierarchical Memory Management for hybrid models.

Algorithmic core only; CUDA / scheduler hot-path wiring is intentionally
left as stubs. See ``plan.md`` (roadmap) and ``Summary.md`` (status,
including the sglang prelude alignment).
"""

from vllm.v1.core.hima.actuator import (
    CudaDriver,
    CudaDriverError,
    CudaDriverNotAvailable,
    CuMemVMMPool,
    InMemoryVMMActuator,
    PageHandle,
    VMMActuator,
    get_cuda_driver,
    probe_vmm_environment,
    reset_cuda_driver,
)
from vllm.v1.core.hima.budgeter_task import (
    BudgeterBackgroundTask,
)
from vllm.v1.core.hima.budgeter_task import (
    start_if_enabled as start_budgeter_if_enabled,
)
from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.coordinator_hima import HiMACoordinator
from vllm.v1.core.hima.integration import (
    HiMARuntime,
    disable_runtime,
    enable_runtime,
    get_runtime,
    is_enabled,
    maybe_decide_admission,
    maybe_get_free_queue_factory,
    maybe_record_hit,
)
from vllm.v1.core.hima.inter_pool import (
    AdmissionAction,
    AdmissionDecision,
    Admitter,
    BisectionBudgeter,
    Budgeter,
    CrossPoolPlanner,
    CrossPoolPolicyConfig,
    EnginePressureAdapter,
    MovePlan,
    PlanDecision,
    PressureSignals,
    VLLMPressureAdapter,
    constant_utility,
    get_default_adapter,
    policy_from_env,
    reciprocal_utility,
    reset_default_adapter,
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
    RuntimeActuatorCost,
    get_cost_curves,
    get_runtime_actuator_cost,
    hits_per_byte_score,
    lpb_score,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue
from vllm.v1.core.hima.metrics import HiMAMetricsExporter
from vllm.v1.core.hima.telemetry import HiMATelemetry, PoolTelemetry

__all__ = [
    "AdmissionAction",
    "AdmissionDecision",
    "Admitter",
    "BisectionBudgeter",
    "BlockId",
    "Budgeter",
    "BudgeterBackgroundTask",
    "CostCurve",
    "CostCurveCalibrator",
    "CostCurveRegistry",
    "CostCurves",
    "CrossPoolPlanner",
    "CrossPoolPolicyConfig",
    "CuMemVMMPool",
    "CudaDriver",
    "CudaDriverError",
    "CudaDriverNotAvailable",
    "EnginePressureAdapter",
    "HiMACoordinator",
    "HiMAConfig",
    "HiMAMetricsExporter",
    "HiMARuntime",
    "HiMATelemetry",
    "InMemoryVMMActuator",
    "LEGACY_DEFAULT",
    "LPBFreeBlockQueue",
    "LPBPriorityQueue",
    "LinearCostCurve",
    "MovePlan",
    "PageHandle",
    "PathCountedHitCounter",
    "PlanDecision",
    "PoolKind",
    "PoolTelemetry",
    "PressureSignals",
    "QuadraticCostCurve",
    "RuntimeActuatorCost",
    "VLLMPressureAdapter",
    "VMMActuator",
    "constant_utility",
    "disable_runtime",
    "enable_runtime",
    "get_cost_curves",
    "get_cuda_driver",
    "get_default_adapter",
    "get_runtime",
    "get_runtime_actuator_cost",
    "hits_per_byte_score",
    "is_enabled",
    "lpb_score",
    "maybe_decide_admission",
    "maybe_get_free_queue_factory",
    "maybe_record_hit",
    "policy_from_env",
    "probe_vmm_environment",
    "reciprocal_utility",
    "reset_cost_curves",
    "reset_cuda_driver",
    "reset_default_adapter",
    "reset_runtime_actuator_cost",
    "start_budgeter_if_enabled",
]
