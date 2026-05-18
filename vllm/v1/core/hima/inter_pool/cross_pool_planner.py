# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-triggered + NB direction-aware cross-pool transfer planner (production path)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from vllm.v1.core.hima.config import PoolKind
from vllm.v1.core.hima.inter_pool.budgeter import MovePlan
from vllm.v1.core.hima.inter_pool.pressure_adapter import (
    EnginePressureAdapter,
    get_default_adapter,
)
from vllm.v1.core.hima.intra_pool.cost_curve import (
    get_cost_curves,
    get_runtime_actuator_cost,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossPoolPolicyConfig:
    """Knobs for :class:`CrossPoolPlanner`; overridable via ``VLLM_HIMA_XPOOL_*``."""

    kv_high_water: float = 0.85
    kv_low_water: float = 0.50
    rec_high_water: float = 0.80
    rec_low_water: float = 0.40
    cooldown_ticks: int = 16
    dst_chunks_per_action: int = 1
    qdepth_trigger: int = 0
    edge_trigger: bool = True
    net_benefit_enabled: bool = True
    nb_chunk_cost_us: float = 5000.0  # static lower bound (us / chunk)
    nb_margin: float = 1.5
    nb_persist_eval_period: int = 10
    hysteresis_delta: float = 0.05
    both_above_max_threshold: float = 0.95
    both_above_min_gap: float = 0.20
    nb_direction_aware: bool = True


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def policy_from_env() -> CrossPoolPolicyConfig:
    """Read :class:`CrossPoolPolicyConfig` from ``VLLM_HIMA_XPOOL_*`` env."""

    return CrossPoolPolicyConfig(
        kv_high_water=_env_float("VLLM_HIMA_XPOOL_KV_HIGH", 0.85),
        kv_low_water=_env_float("VLLM_HIMA_XPOOL_KV_LOW", 0.50),
        rec_high_water=_env_float("VLLM_HIMA_XPOOL_REC_HIGH", 0.80),
        rec_low_water=_env_float("VLLM_HIMA_XPOOL_REC_LOW", 0.40),
        cooldown_ticks=_env_int("VLLM_HIMA_XPOOL_COOLDOWN", 16),
        dst_chunks_per_action=_env_int("VLLM_HIMA_XPOOL_UNIT", 1),
        qdepth_trigger=_env_int("VLLM_HIMA_XPOOL_QDEPTH_TRIGGER", 0),
        edge_trigger=_env_bool("VLLM_HIMA_XPOOL_EDGE_TRIGGER", True),
        net_benefit_enabled=_env_bool("VLLM_HIMA_XPOOL_NET_BENEFIT", True),
        nb_chunk_cost_us=_env_float("VLLM_HIMA_XPOOL_NB_CHUNK_COST_US", 5000.0),
        nb_margin=_env_float("VLLM_HIMA_XPOOL_NB_MARGIN", 1.5),
        nb_persist_eval_period=_env_int("VLLM_HIMA_XPOOL_NB_PERSIST_EVAL_PERIOD", 10),
        hysteresis_delta=_env_float("VLLM_HIMA_XPOOL_HYSTERESIS_DELTA", 0.05),
        both_above_max_threshold=_env_float("VLLM_HIMA_XPOOL_BOTH_ABOVE_MAX", 0.95),
        both_above_min_gap=_env_float("VLLM_HIMA_XPOOL_BOTH_ABOVE_GAP", 0.20),
        nb_direction_aware=_env_bool("VLLM_HIMA_XPOOL_NB_DIRECTION_AWARE", True),
    )


@dataclass
class PlanDecision:
    """Output of one planner tick; ``move=None`` means no action."""

    move: MovePlan | None
    reason: str
    usage_kv: float
    usage_rec: float
    queue_depth: int = 0

    @property
    def fired(self) -> bool:
        return self.move is not None and self.move.n_pages > 0


class CrossPoolPlanner:
    """Edge-triggered + NB direction-aware cross-pool transfer planner."""

    BELOW_LOW = "below_low"
    IN_BAND = "in_band"
    ABOVE_HIGH = "above_high"

    def __init__(
        self,
        config: CrossPoolPolicyConfig | None = None,
        adapter: EnginePressureAdapter | None = None,
    ) -> None:
        self.config = config if config is not None else policy_from_env()
        self._adapter = adapter if adapter is not None else get_default_adapter()
        self._cost_curves = getattr(self._adapter, "cost_curves", None)
        if self._cost_curves is None:
            self._cost_curves = get_cost_curves()
        self._cooldown_remaining: int = 0
        self._tick_count: int = 0
        self._kv_state: str = self.IN_BAND
        self._rec_state: str = self.IN_BAND
        self._kv_above_consec: int = 0
        self._rec_above_consec: int = 0

    # ------------------------- helpers --------------------------------- #

    def _classify(self, usage: float, low: float, high: float) -> str:
        if usage >= high:
            return self.ABOVE_HIGH
        if usage <= low:
            return self.BELOW_LOW
        return self.IN_BAND

    @staticmethod
    def _p_func(usage: float, low_water: float) -> float:
        """``P_save = P_loss = max(0, (u - u_low) / (1 - u_low))``.

        Hits 0 below low-water, rises smoothly to 1 at full saturation.
        """

        if low_water >= 1.0:
            return 1.0 if usage >= 1.0 else 0.0
        return max(0.0, min(1.0, (usage - low_water) / (1.0 - low_water)))

    def _direction_to_move(self, direction: str, n_pages: int) -> MovePlan:
        if direction == "kv_to_rec":
            return MovePlan(src=PoolKind.KV, dst=PoolKind.REC, n_pages=n_pages)
        if direction == "rec_to_kv":
            return MovePlan(src=PoolKind.REC, dst=PoolKind.KV, n_pages=n_pages)
        raise ValueError(f"unknown direction {direction!r}")

    # ------------------------- NB direction-aware gate ----------------- #

    def _pick_direction_by_nb(
        self,
        usage_kv: float,
        usage_rec: float,
        snapshot: dict | None,
    ) -> tuple[str | None, float, str]:
        """Direction-aware NB gate.

        ``NB(src→dst) = c_dst * P_save(dst) − c_src * P_loss(src)``
        Returns ``(best_dir, nb_us, reason)`` or ``(None, ...)`` if gate not cleared.
        """

        c = self.config
        if self._cost_curves is None:
            return None, 0.0, "nb_direction: no cost curves"
        snap = snapshot or {}
        L_kv = float(snap.get("slow_recovery_len_kv", 0) or 0)
        L_rec = float(snap.get("slow_recovery_len_rec", L_kv) or L_kv)
        if L_kv <= 0 and L_rec <= 0:
            return None, 0.0, "nb_direction: no recovery_len observed"
        c_kv = self._cost_curves.c_kv_us(L_kv if L_kv > 0 else L_rec)
        c_m = self._cost_curves.c_m_us(L_rec if L_rec > 0 else L_kv)

        p_save_kv = self._p_func(usage_kv, c.kv_low_water)
        p_loss_kv = p_save_kv
        p_save_rec = self._p_func(usage_rec, c.rec_low_water)
        p_loss_rec = p_save_rec

        lifetime = max(1, c.cooldown_ticks)
        nb_k2r = lifetime * (c_m * p_save_rec - c_kv * p_loss_kv)
        nb_r2k = lifetime * (c_kv * p_save_kv - c_m * p_loss_rec)

        kv_active = usage_kv
        rec_active = float(snap.get("usage_rec_active", usage_rec) or usage_rec)
        if kv_active >= c.kv_high_water:
            nb_k2r = float("-inf")
        if rec_active >= c.rec_high_water:
            nb_r2k = float("-inf")

        runtime_cost = get_runtime_actuator_cost()
        c_actuator_us = (
            runtime_cost.current_us
            if runtime_cost.is_calibrated
            else max(runtime_cost.current_us, c.nb_chunk_cost_us)
        )
        threshold = c.nb_margin * c.dst_chunks_per_action * c_actuator_us

        if nb_k2r >= nb_r2k and nb_k2r >= threshold:
            best, best_nb = "kv_to_rec", nb_k2r
        elif nb_r2k > nb_k2r and nb_r2k >= threshold:
            best, best_nb = "rec_to_kv", nb_r2k
        else:
            best, best_nb = None, max(nb_k2r, nb_r2k)

        reason = (
            f"NB[k2r]={nb_k2r:.0f}us NB[r2k]={nb_r2k:.0f}us "
            f"thr={threshold:.0f}us "
            f"(c_kv={c_kv:.0f}us@L={L_kv:.0f}, c_m={c_m:.0f}us@L={L_rec:.0f}, "
            f"P: kv={p_save_kv:.2f} rec={p_save_rec:.2f})"
        )
        return best, best_nb, reason

    # ------------------------- net benefit gate ------------------------ #

    def _net_benefit_ok(
        self,
        snapshot: dict | None,
        edge_active: bool,
    ) -> tuple[bool, str]:
        c = self.config
        if not c.net_benefit_enabled:
            return True, "nb=off"
        signals = self._adapter.signals_from_snapshot(
            snapshot or {},
            self._kv_above_consec,
            self._rec_above_consec,
            edge_active=edge_active,
        )
        benefit_us = signals.total_benefit_us
        cost_us = c.dst_chunks_per_action * c.nb_chunk_cost_us
        if benefit_us <= 0:
            return False, f"nb: no pressure ({signals.reason_str()})"
        if benefit_us < cost_us * c.nb_margin:
            return False, (
                f"nb: B={benefit_us:.0f}us < C={cost_us:.0f}us x "
                f"margin={c.nb_margin} ({signals.reason_str()})"
            )
        return True, (
            f"nb: B={benefit_us:.0f}us >= C={cost_us:.0f}us x "
            f"margin={c.nb_margin} ({signals.reason_str()})"
        )

    # ------------------------- public entry ---------------------------- #

    def decide(
        self,
        usage_kv: float,
        usage_rec: float,
        queue_depth: int = 0,
        snapshot: dict | None = None,
        edge_active: bool = False,
    ) -> PlanDecision:
        decision = self._decide_inner(
            usage_kv, usage_rec, queue_depth, snapshot, edge_active
        )
        if decision.fired:
            logger.info(
                "[hima-xpool] FIRE tick=%d move=%s usage_kv=%.2f "
                "usage_rec=%.2f reason=%s",
                self._tick_count,
                decision.move,
                decision.usage_kv,
                decision.usage_rec,
                decision.reason,
            )
        return decision

    def _decide_inner(
        self,
        usage_kv: float,
        usage_rec: float,
        queue_depth: int,
        snapshot: dict | None,
        edge_active: bool,
    ) -> PlanDecision:
        self._tick_count += 1
        c = self.config

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return PlanDecision(
                move=None,
                reason=f"cooldown ({self._cooldown_remaining} left)",
                usage_kv=usage_kv,
                usage_rec=usage_rec,
                queue_depth=queue_depth,
            )

        new_kv = self._classify(usage_kv, c.kv_low_water, c.kv_high_water)
        new_rec = self._classify(usage_rec, c.rec_low_water, c.rec_high_water)
        self._kv_above_consec = (
            self._kv_above_consec + 1 if new_kv == self.ABOVE_HIGH else 0
        )
        self._rec_above_consec = (
            self._rec_above_consec + 1 if new_rec == self.ABOVE_HIGH else 0
        )
        old_kv, old_rec = self._kv_state, self._rec_state
        self._kv_state, self._rec_state = new_kv, new_rec

        if c.nb_direction_aware:
            best_dir, best_nb, why = self._pick_direction_by_nb(
                usage_kv, usage_rec, snapshot
            )
            if best_dir is not None:
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    move=self._direction_to_move(best_dir, c.dst_chunks_per_action),
                    reason=f"nb_direction: best={best_dir} NB={best_nb:.0f}us [{why}]",
                    usage_kv=usage_kv,
                    usage_rec=usage_rec,
                    queue_depth=queue_depth,
                )
            return PlanDecision(
                move=None,
                reason=f"nb_direction: no candidate cleared gate [{why}]",
                usage_kv=usage_kv,
                usage_rec=usage_rec,
                queue_depth=queue_depth,
            )

        if c.edge_trigger:
            kv_changed = new_kv != old_kv
            rec_changed = new_rec != old_rec
            if not (kv_changed or rec_changed):
                return PlanDecision(
                    move=None,
                    reason=f"edge: stable kv={new_kv} rec={new_rec}",
                    usage_kv=usage_kv,
                    usage_rec=usage_rec,
                    queue_depth=queue_depth,
                )

            def _try_fire(direction: str, edge_reason: str) -> PlanDecision:
                ok, why = self._net_benefit_ok(snapshot, edge_active)
                if not ok:
                    return PlanDecision(
                        move=None,
                        reason=(
                            f"edge: would fire {direction} ({edge_reason}) but {why}"
                        ),
                        usage_kv=usage_kv,
                        usage_rec=usage_rec,
                        queue_depth=queue_depth,
                    )
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    move=self._direction_to_move(direction, c.dst_chunks_per_action),
                    reason=f"edge: {edge_reason} [{why}]",
                    usage_kv=usage_kv,
                    usage_rec=usage_rec,
                    queue_depth=queue_depth,
                )

            if rec_changed and new_rec == self.ABOVE_HIGH and new_kv != self.ABOVE_HIGH:
                return _try_fire(
                    "kv_to_rec",
                    f"rec {old_rec}->ABOVE_HIGH ({usage_rec:.2f}); kv={new_kv}",
                )
            if kv_changed and new_kv == self.ABOVE_HIGH and new_rec != self.ABOVE_HIGH:
                return _try_fire(
                    "rec_to_kv",
                    f"kv {old_kv}->ABOVE_HIGH ({usage_kv:.2f}); rec={new_rec}",
                )
            return PlanDecision(
                move=None,
                reason=f"edge: transition {old_kv}->{new_kv} "
                f"rec {old_rec}->{new_rec} (no actionable pattern)",
                usage_kv=usage_kv,
                usage_rec=usage_rec,
                queue_depth=queue_depth,
            )

        if usage_kv >= c.kv_high_water and usage_rec <= c.rec_low_water:
            self._cooldown_remaining = c.cooldown_ticks
            return PlanDecision(
                move=self._direction_to_move("rec_to_kv", c.dst_chunks_per_action),
                reason=f"level: kv={usage_kv:.2f}>={c.kv_high_water:.2f} & "
                f"rec={usage_rec:.2f}<={c.rec_low_water:.2f}",
                usage_kv=usage_kv,
                usage_rec=usage_rec,
                queue_depth=queue_depth,
            )
        if usage_rec >= c.rec_high_water and usage_kv <= c.kv_low_water:
            self._cooldown_remaining = c.cooldown_ticks
            return PlanDecision(
                move=self._direction_to_move("kv_to_rec", c.dst_chunks_per_action),
                reason=f"level: rec={usage_rec:.2f}>={c.rec_high_water:.2f} & "
                f"kv={usage_kv:.2f}<={c.kv_low_water:.2f}",
                usage_kv=usage_kv,
                usage_rec=usage_rec,
                queue_depth=queue_depth,
            )
        return PlanDecision(
            move=None,
            reason=f"both within band: kv={usage_kv:.2f} rec={usage_rec:.2f}",
            usage_kv=usage_kv,
            usage_rec=usage_rec,
            queue_depth=queue_depth,
        )


__all__ = [
    "CrossPoolPlanner",
    "CrossPoolPolicyConfig",
    "PlanDecision",
    "policy_from_env",
]
