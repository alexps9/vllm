# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inter-pool HiMA: Admitter, Budgeter, CrossPoolPlanner, PressureAdapter.

Two budgeting paths are exposed:

* :class:`BisectionBudgeter` -- paper-pure shadow-price bisection,
  closed-form, ideal for unit testing on synthetic utility curves.
* :class:`CrossPoolPlanner` -- sglang-aligned production gate
  (edge-triggered + NB direction-aware), consumes :class:`PressureSignals`
  and runtime-EWMA actuator cost.

Both produce :class:`MovePlan` so the actuator path is shared.
"""

from vllm.v1.core.hima.inter_pool.admitter import (
    AdmissionAction,
    AdmissionDecision,
    Admitter,
)
from vllm.v1.core.hima.inter_pool.budgeter import (
    BisectionBudgeter,
    Budgeter,
    MovePlan,
    constant_utility,
    reciprocal_utility,
)
from vllm.v1.core.hima.inter_pool.cross_pool_planner import (
    CrossPoolPlanner,
    CrossPoolPolicyConfig,
    PlanDecision,
    policy_from_env,
)
from vllm.v1.core.hima.inter_pool.pressure_adapter import (
    EnginePressureAdapter,
    PressureSignals,
    VLLMPressureAdapter,
    get_default_adapter,
    reset_default_adapter,
)

__all__ = [
    "AdmissionAction",
    "AdmissionDecision",
    "Admitter",
    "BisectionBudgeter",
    "Budgeter",
    "CrossPoolPlanner",
    "CrossPoolPolicyConfig",
    "EnginePressureAdapter",
    "MovePlan",
    "PlanDecision",
    "PressureSignals",
    "VLLMPressureAdapter",
    "constant_utility",
    "get_default_adapter",
    "policy_from_env",
    "reciprocal_utility",
    "reset_default_adapter",
]
