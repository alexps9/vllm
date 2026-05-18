# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus-compatible metrics exporter for HiMA (no prometheus_client dep)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.hima.config import PoolKind

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime


@dataclass(frozen=True)
class _Gauge:
    name: str
    help: str
    value: float


@dataclass(frozen=True)
class _Counter:
    name: str
    help: str
    value: float


class HiMAMetricsExporter:
    """Pulls live state from :class:`HiMARuntime` and emits ``hima_*`` metrics."""

    def __init__(self, runtime: HiMARuntime) -> None:
        self._runtime = runtime

    # ------------------------------ JSON --------------------------------- #

    def snapshot(self) -> dict[str, float | int | str]:
        rt = self._runtime
        actuator = rt.actuator
        out: dict[str, float | int | str] = {
            "hima_enabled": 1.0,
            "hima_page_size_bytes": int(
                getattr(
                    actuator, "chunk_size_bytes", rt.config.hima_page_size_bytes or 0
                )
            ),
            "hima_actuator_total_pages": float(actuator.total_pages()),
            "hima_actuator_remap_cost_us": float(actuator.remap_cost(1)),
            "hima_runtime_actuator_cost_us": float(rt.runtime_cost.current_us),
        }
        for pool in PoolKind:
            out[f"hima_free_pages_{pool.value}"] = float(actuator.free_pages(pool))
            q = rt.intra_queues.get(pool)
            out[f"hima_lpb_queue_len_{pool.value}"] = float(len(q) if q else 0)
        for k, v in rt.telemetry.snapshot().items():
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                out[f"hima_telemetry_{k}"] = float(v)
        for action, count in rt.decisions.items():
            out[f"hima_decisions_{action}_total"] = float(count)
        return out

    # ---------------------- Prometheus exposition ------------------------ #

    def prom_text(self) -> str:
        """Render the snapshot as a Prometheus exposition string."""
        lines: list[str] = []
        snap = self.snapshot()
        gauges: list[_Gauge] = []
        decisions: list[_Counter] = []
        for k, v in snap.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("hima_decisions_") and k.endswith("_total"):
                action = k[len("hima_decisions_") : -len("_total")]
                decisions.append(
                    _Counter(
                        name="hima_decisions_total",
                        help=f"HiMA admitter/planner action counter (action={action})",
                        value=float(v),
                    )
                )
            else:
                gauges.append(_Gauge(name=k, help=f"HiMA gauge {k}", value=float(v)))

        for g in gauges:
            lines.append(f"# HELP {g.name} {g.help}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name} {g.value}")

        if decisions:
            lines.append("# HELP hima_decisions_total HiMA decision counter")
            lines.append("# TYPE hima_decisions_total counter")
            for k, v in snap.items():
                if k.startswith("hima_decisions_") and k.endswith("_total"):
                    action = k[len("hima_decisions_") : -len("_total")]
                    lines.append(
                        f'hima_decisions_total{{action="{action}"}} {float(v)}'
                    )

        return "\n".join(lines) + "\n"


__all__ = ["HiMAMetricsExporter"]
