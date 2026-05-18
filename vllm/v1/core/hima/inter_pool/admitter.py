# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request 5-candidate admission controller (µs-fast, no memory movement)."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass

from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.telemetry import HiMATelemetry


class AdmissionAction(enum.Enum):
    OWN_FREE = "own_free"
    OWN_EVICT = "own_evict"
    CROSS_FREE = "cross_free"
    CROSS_EVICT = "cross_evict"
    DEFER = "defer"


@dataclass(frozen=True)
class AdmissionDecision:
    action: AdmissionAction
    target_pool: PoolKind
    source_pool: PoolKind | None
    estimated_cost: float

    @property
    def needs_remap(self) -> bool:
        return self.action in (
            AdmissionAction.CROSS_FREE,
            AdmissionAction.CROSS_EVICT,
        )


EvictPeek = Callable[[PoolKind, int], float]
RemapPeek = Callable[[int], float]


@dataclass
class Admitter:
    """5-candidate admission controller.

    Candidates: own-free / own-evict / cross-free / cross-evict / defer.
    """

    cfg: HiMAConfig
    telemetry: HiMATelemetry
    peek_evict_cost: EvictPeek
    peek_remap_cost: RemapPeek

    def decide(self, target_pool: PoolKind, n_pages: int) -> AdmissionDecision:
        if n_pages <= 0:
            raise ValueError(f"n_pages must be > 0, got {n_pages}")

        other = target_pool.other()
        own_free = self.telemetry.free_pages(target_pool)
        cross_free = self.telemetry.free_pages(other)

        candidates: list[tuple[AdmissionAction, PoolKind | None, float]] = []
        if own_free >= n_pages:
            candidates.append((AdmissionAction.OWN_FREE, None, 0.0))
        candidates.append(
            (
                AdmissionAction.OWN_EVICT,
                None,
                self.peek_evict_cost(target_pool, n_pages),
            )
        )
        if cross_free >= n_pages:
            candidates.append(
                (
                    AdmissionAction.CROSS_FREE,
                    other,
                    self.peek_remap_cost(n_pages),
                )
            )
        candidates.append(
            (
                AdmissionAction.CROSS_EVICT,
                other,
                self.peek_evict_cost(other, n_pages) + self.peek_remap_cost(n_pages),
            )
        )

        action, src, cost = min(candidates, key=lambda x: x[2])

        defer_threshold = self._defer_threshold(target_pool, n_pages)
        if cost > defer_threshold:
            return AdmissionDecision(
                action=AdmissionAction.DEFER,
                target_pool=target_pool,
                source_pool=None,
                estimated_cost=cost,
            )
        return AdmissionDecision(
            action=action,
            target_pool=target_pool,
            source_pool=src,
            estimated_cost=cost,
        )

    # -------------------------- internals ------------------------------- #

    def _defer_threshold(self, pool: PoolKind, n_pages: int) -> float:
        # Q * w_q * n / X; infinite when queue is empty (never defer).
        Q = self.telemetry.queue_depth(pool)
        if Q <= 0:
            return float("inf")
        X = self.telemetry.throughput_ewma()
        wq = self.cfg.hima_queue_penalty
        if X <= 0:
            return float("inf")
        return Q * wq * n_pages / X


__all__ = ["Admitter", "AdmissionAction", "AdmissionDecision"]
