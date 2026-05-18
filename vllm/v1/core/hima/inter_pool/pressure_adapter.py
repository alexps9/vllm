# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-native pressure adapter for the cross-pool gate's net-benefit check.

Aligned with ``sglang.srt.budgeter.pressure_adapter`` (paper §sec:design-l2).
The gate is engine-agnostic ``B (benefit) >= C (cost) * margin``; the
adapter translates each engine's *native* admission-pressure signals
into a uniform "us of GPU time saved" space:

* sglang -- tree-cache eviction is the primary signal (the engine's
  primary pressure-relief mechanism);
* vLLM -- preemption / swap-out is the primary signal; KV usage staying
  high is a secondary "saturation prior".

Concrete adapters subclass :class:`EnginePressureAdapter` and implement
:meth:`signals_from_snapshot`. The :class:`HiMAPlanner` consumes the
returned :class:`PressureSignals` directly.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.hima.intra_pool.cost_curve import CostCurves

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Common signal namedtuple (mirrors sglang.srt.budgeter.pressure_adapter).
# --------------------------------------------------------------------------- #


@dataclass
class PressureSignals:
    """Engine-native admission-pressure signals translated to ``us``.

    Each field is "expected GPU time (us) saved by averting one tick's
    worth of accumulated pressure of that type". The gate sums all
    fields and fires when ``sum >= chunk_cost_us * margin``.
    """

    evict_us: float = 0.0  # tree-cache / prefix eviction backlog
    preempt_us: float = 0.0  # vLLM swap-out / preemption backlog
    retract_us: float = 0.0  # currently retracted reqs
    paused_us: float = 0.0  # admission-paused reqs
    queue_us: float = 0.0  # queued reqs waiting for admission
    persist_us: float = 0.0  # pool above-high dwell (saturation prior)
    edge_us: float = 0.0  # phase-transition signal: |du/dt| > threshold

    @property
    def total_benefit_us(self) -> float:
        return (
            self.evict_us
            + self.preempt_us
            + self.retract_us
            + self.paused_us
            + self.queue_us
            + self.persist_us
            + self.edge_us
        )

    def reason_str(self) -> str:
        parts = []
        for name in (
            "evict",
            "preempt",
            "retract",
            "paused",
            "queue",
            "persist",
            "edge",
        ):
            v = getattr(self, f"{name}_us")
            if v != 0.0:
                parts.append(f"{name}={v:.0f}")
        return " ".join(parts) if parts else "no_signal"


# --------------------------------------------------------------------------- #
# Adapter interface.
# --------------------------------------------------------------------------- #


class EnginePressureAdapter(ABC):
    """Translate engine-native pool-pressure signals to benefit-microseconds.

    Subclasses implement :meth:`signals_from_snapshot`. The planner calls
    this on every gate evaluation; it must be cheap (no I/O, just
    arithmetic on snapshot fields the engine already populates).

    Subclasses may expose ``cost_curves`` so the planner can quote
    ``c_KV(L)`` / ``c_M(L)`` in its decision logs.
    """

    cost_curves: CostCurves | None = None  # populated by concrete subclasses

    @abstractmethod
    def signals_from_snapshot(
        self,
        snapshot: dict,
        kv_consec: int,
        rec_consec: int,
        edge_active: bool = False,
    ) -> PressureSignals: ...


# --------------------------------------------------------------------------- #
# Default adapter for vLLM (preemption-dominated).
# --------------------------------------------------------------------------- #


class VLLMPressureAdapter(EnginePressureAdapter):
    """vLLM's preempt-first scheduler.

    vLLM's primary pressure-relief mechanism is **preemption** (the
    Scheduler returns ``None`` from ``allocate_slots`` and a running
    request is preempted; see ``vllm/v1/core/sched/scheduler.py``). When
    the cross-pool gate fires and frees pages, the deferred re-prefill of
    those preempted requests is the saved cost.

    Coefficients (overridable via ``VLLM_HIMA_*_US`` env):

    * ``prefill_save_us_per_token`` -- per-token GPU cost of prefill on
      target hardware/model; derived from ``CostCurves.c_kv_per_token_us``
      at ``DEFAULT_KV_RECOVER_L`` if curves are calibrated.
    * ``full_prefill_us`` -- "full" re-prefill cost for one preempted
      request, derived from ``c_KV(DEFAULT_PREEMPT_L)``.
    * ``queue_wait_us`` -- penalty per queued req (default 100 us).
    * ``persist_tick_us`` -- per-tick value of sustained pool above-high
      (acts as a "fire even without explicit signal" accumulator).
    """

    DEFAULT_KV_RECOVER_L: float = 2048.0
    DEFAULT_PREEMPT_L: float = 6144.0

    def __init__(
        self,
        prefill_save_us_per_token: float | None = None,
        full_prefill_us: float | None = None,
        queue_wait_us: float | None = None,
        persist_tick_us: float | None = None,
        edge_us: float | None = None,
    ) -> None:
        from vllm.v1.core.hima.intra_pool.cost_curve import get_cost_curves

        self.cost_curves = get_cost_curves()
        kv_default_us_per_tok = self.cost_curves.c_kv_per_token_us(
            self.DEFAULT_KV_RECOVER_L
        )
        preempt_default_us = self.cost_curves.c_kv_us(self.DEFAULT_PREEMPT_L)

        self.prefill_save_us_per_token = (
            prefill_save_us_per_token
            if prefill_save_us_per_token is not None
            else float(
                os.environ.get(
                    "VLLM_HIMA_PREFILL_SAVE_US_PER_TOKEN",
                    str(kv_default_us_per_tok or 12.5),
                )
            )
        )
        self.full_prefill_us = (
            full_prefill_us
            if full_prefill_us is not None
            else float(
                os.environ.get(
                    "VLLM_HIMA_FULL_PREFILL_US",
                    str(preempt_default_us or 75000.0),
                )
            )
        )
        self.queue_wait_us = (
            queue_wait_us
            if queue_wait_us is not None
            else float(os.environ.get("VLLM_HIMA_QUEUE_WAIT_US", "100"))
        )
        self.persist_tick_us = (
            persist_tick_us
            if persist_tick_us is not None
            else float(os.environ.get("VLLM_HIMA_PERSIST_TICK_US", "1000"))
        )
        self.edge_us = (
            edge_us
            if edge_us is not None
            else float(os.environ.get("VLLM_HIMA_EDGE_US", "5000"))
        )

        # Last-call breakdown for diagnostics (kept tiny so it's cheap to copy).
        self.last_breakdown: dict | None = None

    def signals_from_snapshot(
        self,
        snapshot: dict,
        kv_consec: int,
        rec_consec: int,
        edge_active: bool = False,
    ) -> PressureSignals:
        # Preemption: number of newly-preempted requests times their
        # avg input length times prefill cost per token.
        num_preempted = float(snapshot.get("num_preempted_recent", 0) or 0)
        avg_preempt_tokens = float(snapshot.get("avg_preempt_input_tokens", 0) or 0)
        if num_preempted > 0 and avg_preempt_tokens > 0:
            preempt_us = (
                num_preempted * avg_preempt_tokens * self.prefill_save_us_per_token
            )
        elif num_preempted > 0:
            # Fallback: each preempt costs roughly one full re-prefill.
            preempt_us = num_preempted * self.full_prefill_us
        else:
            preempt_us = 0.0

        # vLLM also tracks evict events at the prefix-cache level; mirror
        # the sglang adapter's evict signal when available.
        num_evicted_tokens = float(snapshot.get("num_evicted_tokens_recent", 0) or 0)
        evict_us = num_evicted_tokens * self.prefill_save_us_per_token

        # Queue depth: each queued req contributes a small wait penalty.
        queue_depth = float(snapshot.get("num_queue_reqs", 0) or 0)
        queue_us = queue_depth * self.queue_wait_us

        # Persist: sustained ABOVE_HIGH for either pool accumulates.
        persist_us = (kv_consec + rec_consec) * self.persist_tick_us

        # Edge: one-tick-bounded phase-transition signal.
        edge_us = self.edge_us if edge_active else 0.0

        signals = PressureSignals(
            evict_us=evict_us,
            preempt_us=preempt_us,
            queue_us=queue_us,
            persist_us=persist_us,
            edge_us=edge_us,
        )

        self.last_breakdown = {
            "num_preempted_recent": num_preempted,
            "num_evicted_tokens_recent": num_evicted_tokens,
            "queue_depth": queue_depth,
            "kv_consec": kv_consec,
            "rec_consec": rec_consec,
            "edge_active": edge_active,
            "evict_us": evict_us,
            "preempt_us": preempt_us,
            "queue_us": queue_us,
            "persist_us": persist_us,
            "edge_us": edge_us,
            "total_benefit_us": signals.total_benefit_us,
        }
        return signals


_default_adapter: EnginePressureAdapter | None = None


def get_default_adapter() -> EnginePressureAdapter:
    """Process-wide default adapter (currently :class:`VLLMPressureAdapter`)."""

    global _default_adapter
    if _default_adapter is None:
        _default_adapter = VLLMPressureAdapter()
    return _default_adapter


def reset_default_adapter() -> None:
    """Test hook."""

    global _default_adapter
    _default_adapter = None


__all__ = [
    "EnginePressureAdapter",
    "PressureSignals",
    "VLLMPressureAdapter",
    "get_default_adapter",
    "reset_default_adapter",
]
