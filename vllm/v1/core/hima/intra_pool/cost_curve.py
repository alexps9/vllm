# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery cost curves c_KV(L) = αL²+βL+γ and c_M(L) = αL+β (ms).

Coefficient sources (in priority order):
1. ``VLLM_HIMA_CSIGMA_*`` env vars
2. ``VLLM_HIMA_CSIGMA_JSON`` calibration file
3. ``LEGACY_DEFAULT`` (H200 BF16 reference; warns on Blackwell)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.v1.core.hima.config import PoolKind

logger = logging.getLogger(__name__)


class CostCurve(ABC):
    """Abstract recovery-cost curve ``c(s)``."""

    @abstractmethod
    def cost(self, s: int) -> float: ...


@dataclass(frozen=True)
class QuadraticCostCurve(CostCurve):
    """``c(s) = alpha * s^2`` -- KV simplification (no launch overhead)."""

    alpha: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")

    def cost(self, s: int) -> float:
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")
        return self.alpha * s * s


@dataclass(frozen=True)
class LinearCostCurve(CostCurve):
    """``c(s) = alpha * s`` -- recurrent simplification."""

    alpha: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")

    def cost(self, s: int) -> float:
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")
        return self.alpha * s


@dataclass
class CostCurveRegistry:
    """Maps a :class:`PoolKind` to its :class:`CostCurve` (unit-test helper)."""

    kv_curve: CostCurve
    rec_curve: CostCurve

    def for_pool(self, pool: PoolKind) -> CostCurve:
        return self.kv_curve if pool is PoolKind.KV else self.rec_curve


class CostCurveCalibrator:
    """Least-squares fit for single-shape curves (test helper)."""

    @staticmethod
    def fit_quadratic(samples: list[tuple[int, float]]) -> QuadraticCostCurve:
        if not samples:
            raise ValueError("Need at least one sample to fit a curve")
        num = sum(s * s * t for s, t in samples)
        den = sum(s**4 for s, _ in samples)
        if den == 0:
            raise ValueError("All samples have s=0; cannot fit")
        alpha = num / den
        if alpha <= 0:
            raise ValueError(f"Fitted alpha must be > 0, got {alpha}")
        return QuadraticCostCurve(alpha=alpha)

    @staticmethod
    def fit_linear(samples: list[tuple[int, float]]) -> LinearCostCurve:
        if not samples:
            raise ValueError("Need at least one sample to fit a curve")
        num = sum(s * t for s, t in samples)
        den = sum(s * s for s, _ in samples)
        if den == 0:
            raise ValueError("All samples have s=0; cannot fit")
        alpha = num / den
        if alpha <= 0:
            raise ValueError(f"Fitted alpha must be > 0, got {alpha}")
        return LinearCostCurve(alpha=alpha)


@dataclass(frozen=True)
class CostCurves:
    """Full quadratic KV + linear recurrent cost curves (ms).

    ``c_KV(L) = kv_alpha * L^2 + kv_beta * L + kv_gamma``
    ``c_M(L)  = m_alpha  * L   + m_beta``
    ``L_star``: crossover point where ``c_KV = c_M``.
    """

    kv_alpha: float  # ms / token^2
    kv_beta: float  # ms / token
    kv_gamma: float  # ms (kernel launch / fixed)
    m_alpha: float  # ms / token
    m_beta: float  # ms (chunk setup overhead)
    L_star: float = 0.0  # tokens; 0 if no real crossover
    source: str = "unspecified"

    def c_kv_ms(self, L: float) -> float:
        if L <= 0:
            return self.kv_gamma
        return max(0.0, self.kv_alpha * L * L + self.kv_beta * L + self.kv_gamma)

    def c_m_ms(self, L: float) -> float:
        if L <= 0:
            return self.m_beta
        return max(0.0, self.m_alpha * L + self.m_beta)

    def c_kv_per_token_us(self, L: float) -> float:
        if L <= 0:
            return 0.0
        return self.c_kv_ms(L) * 1000.0 / L

    def c_m_per_token_us(self, L: float) -> float:
        if L <= 0:
            return 0.0
        return self.c_m_ms(L) * 1000.0 / L

    def c_kv_us(self, L: float) -> float:
        return self.c_kv_ms(L) * 1000.0

    def c_m_us(self, L: float) -> float:
        return self.c_m_ms(L) * 1000.0


LEGACY_DEFAULT = CostCurves(
    kv_alpha=1.19e-7,
    kv_beta=0.0,
    kv_gamma=0.44,
    m_alpha=2.17e-3,
    m_beta=6.99,
    L_star=21780.0,
    source="legacy_default_h200",
)


def _try_load_env() -> CostCurves | None:
    """Load CostCurves from ``VLLM_HIMA_CSIGMA_*`` (sglang-compatible names)."""

    if os.environ.get("VLLM_HIMA_CSIGMA_KV_ALPHA") is None:
        return None
    try:
        curves = CostCurves(
            kv_alpha=float(os.environ["VLLM_HIMA_CSIGMA_KV_ALPHA"]),
            kv_beta=float(os.environ.get("VLLM_HIMA_CSIGMA_KV_BETA", "0")),
            kv_gamma=float(os.environ["VLLM_HIMA_CSIGMA_KV_GAMMA"]),
            m_alpha=float(os.environ["VLLM_HIMA_CSIGMA_M_ALPHA"]),
            m_beta=float(os.environ["VLLM_HIMA_CSIGMA_M_BETA"]),
            L_star=float(os.environ.get("VLLM_HIMA_CSIGMA_LSTAR", "0")),
            source="env",
        )
        logger.info(
            "[hima] CostCurves[env]: c_KV=%.3eL^2%+.3eL%+.3e ms, "
            "c_M=%.3eL%+.3e ms, L*=%.0f tok",
            curves.kv_alpha,
            curves.kv_beta,
            curves.kv_gamma,
            curves.m_alpha,
            curves.m_beta,
            curves.L_star,
        )
        return curves
    except (ValueError, KeyError) as e:
        logger.warning("[hima] failed to parse VLLM_HIMA_CSIGMA_* env: %s", e)
        return None


def _try_load_json() -> CostCurves | None:
    """Load CostCurves from ``VLLM_HIMA_CSIGMA_JSON`` calibration file."""

    path = os.environ.get("VLLM_HIMA_CSIGMA_JSON")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
        fit = rec["fit"]
        curves = CostCurves(
            kv_alpha=float(fit["c_kv"]["alpha_ms_per_tok2"]),
            kv_beta=float(fit["c_kv"].get("beta_ms_per_tok", 0.0)),
            kv_gamma=float(fit["c_kv"]["gamma_ms"]),
            m_alpha=float(fit["c_m"]["alpha_ms_per_tok"]),
            m_beta=float(fit["c_m"]["beta_ms"]),
            L_star=float(fit.get("crossover_L_star", 0.0)),
            source=f"json:{path}",
        )
        logger.info("[hima] CostCurves[%s] loaded", curves.source)
        return curves
    except Exception as e:
        logger.warning("[hima] failed to load VLLM_HIMA_CSIGMA_JSON=%s: %s", path, e)
        return None


_singleton: CostCurves | None = None
_warned_legacy: bool = False


def get_cost_curves() -> CostCurves:
    """Process-wide singleton: env → JSON file → legacy default (warns)."""

    global _singleton, _warned_legacy
    if _singleton is not None:
        return _singleton
    curves = _try_load_env() or _try_load_json()
    if curves is None:
        curves = LEGACY_DEFAULT
        if not _warned_legacy:
            logger.warning(
                "[hima] No VLLM_HIMA_CSIGMA_* calibration present. Using "
                "legacy default (Qwen3.5-35B-A3B / H200 BF16 reference). "
                "On RTX PRO 6000 Blackwell this is materially wrong; "
                "calibrate via benchmarks/hima_cost_curve.py."
            )
            _warned_legacy = True
    _singleton = curves
    return _singleton


def reset_cost_curves() -> None:
    """Test hook: clear the singleton so the next get reloads from env."""

    global _singleton, _warned_legacy
    _singleton = None
    _warned_legacy = False


__all__ = [
    "CostCurve",
    "CostCurveCalibrator",
    "CostCurveRegistry",
    "CostCurves",
    "LEGACY_DEFAULT",
    "LinearCostCurve",
    "QuadraticCostCurve",
    "get_cost_curves",
    "reset_cost_curves",
]
