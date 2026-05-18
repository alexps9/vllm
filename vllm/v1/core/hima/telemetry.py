# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EWMA-smoothed engine telemetry for HiMA Budgeter / Admitter."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.core.hima.config import PoolKind


def _ewma(prev: float | None, new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


@dataclass
class PoolTelemetry:
    """Per-pool EWMA signals."""

    hit_rate: float | None = None
    queue_depth: float | None = None
    free_pages: int = 0
    slow_recovery_len: float | None = None


@dataclass
class HiMATelemetry:
    """EWMA-smoothed engine signals consumed by the Cross-Pool Planner."""

    alpha: float = 0.2
    throughput: float | None = None
    pools: dict[PoolKind, PoolTelemetry] = field(default_factory=dict)

    # Engine-level signals (set via observe()).
    num_queue_reqs: int = 0
    num_preempted_recent: int = 0
    usage_kv: float = 0.0
    usage_rec: float = 0.0
    usage_rec_active: float = 0.0
    avg_preempt_input_tokens: float | None = None
    edge_active: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        for kind in PoolKind:
            self.pools.setdefault(kind, PoolTelemetry())

    # -------------------------- public api ------------------------------ #

    def observe_pool(
        self,
        pool: PoolKind,
        *,
        hit_rate: float | None = None,
        queue_depth: float | None = None,
        free_pages: int | None = None,
        slow_recovery_len: float | None = None,
    ) -> None:
        slot = self.pools[pool]
        if hit_rate is not None:
            slot.hit_rate = _ewma(slot.hit_rate, hit_rate, self.alpha)
        if queue_depth is not None:
            slot.queue_depth = _ewma(slot.queue_depth, queue_depth, self.alpha)
        if free_pages is not None:
            slot.free_pages = int(free_pages)
        if slow_recovery_len is not None:
            slot.slow_recovery_len = _ewma(
                slot.slow_recovery_len, slow_recovery_len, self.alpha
            )

    def observe_throughput(self, bytes_per_sec: float) -> None:
        self.throughput = _ewma(self.throughput, bytes_per_sec, self.alpha)

    def observe(self, signals: dict) -> None:
        """Update known engine-level signals; unknown keys silently ignored."""
        if "num_queue_reqs" in signals:
            self.num_queue_reqs = int(signals["num_queue_reqs"])
        if "num_preempted_recent" in signals:
            self.num_preempted_recent = int(signals["num_preempted_recent"])
        for key in ("usage_kv", "usage_rec", "usage_rec_active"):
            if key in signals:
                cur = getattr(self, key)
                new = float(signals[key])
                setattr(self, key, _ewma(cur or None, new, self.alpha))
        if "avg_preempt_input_tokens" in signals:
            self.avg_preempt_input_tokens = _ewma(
                self.avg_preempt_input_tokens,
                float(signals["avg_preempt_input_tokens"]),
                self.alpha,
            )
        if "edge_active" in signals:
            self.edge_active = bool(signals["edge_active"])
        if "throughput" in signals:
            self.observe_throughput(float(signals["throughput"]))

    # -------------------------- accessors ------------------------------- #

    def hit_rate(self, pool: PoolKind) -> float | None:
        return self.pools[pool].hit_rate

    def queue_depth(self, pool: PoolKind) -> float:
        return self.pools[pool].queue_depth or 0.0

    def free_pages(self, pool: PoolKind) -> int:
        return self.pools[pool].free_pages

    def throughput_ewma(self) -> float:
        return self.throughput or 1.0

    def snapshot(self) -> dict:
        """Return a dict of current signals for the Cross-Pool Planner."""

        kv = self.pools[PoolKind.KV]
        rec = self.pools[PoolKind.REC]
        return {
            "slow_recovery_len_kv": kv.slow_recovery_len or 0.0,
            "slow_recovery_len_rec": rec.slow_recovery_len or 0.0,
            "kv_free_pages": kv.free_pages,
            "rec_free_pages": rec.free_pages,
            "usage_kv": float(self.usage_kv),
            "usage_rec": float(self.usage_rec),
            "usage_rec_active": float(self.usage_rec_active),
            "num_queue_reqs": int(self.num_queue_reqs),
            "num_preempted_recent": int(self.num_preempted_recent),
            "avg_preempt_input_tokens": (
                float(self.avg_preempt_input_tokens)
                if self.avg_preempt_input_tokens is not None
                else 0.0
            ),
            "edge_active": bool(self.edge_active),
        }


__all__ = ["HiMATelemetry", "PoolTelemetry"]
