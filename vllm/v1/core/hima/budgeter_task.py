# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background daemon thread driving the HiMA Cross-Pool Planner slow loop (τ≈30 s)."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime

logger = logging.getLogger(__name__)


class BudgeterBackgroundTask:
    """Calls ``runtime.planner_tick()`` every ``interval_s`` seconds (daemon thread)."""

    def __init__(
        self,
        runtime: HiMARuntime,
        interval_s: float | None = None,
        name: str = "hima-budgeter",
    ) -> None:
        self._runtime = runtime
        self._interval_s = (
            interval_s
            if interval_s is not None
            else runtime.config.hima_budget_interval_s
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._name = name
        self._tick_count = 0
        self._error_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("HiMA budgeter task started: interval=%.1fs", self._interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        if not self.is_running():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("HiMA budgeter task stopped (ticks=%d)", self._tick_count)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._runtime.planner_tick()
                self._tick_count += 1
            except Exception as exc:  # noqa: BLE001
                self._error_count += 1
                logger.warning("HiMA budgeter tick failed: %s", exc)
            elapsed = time.monotonic() - t0
            remaining = max(0.0, self._interval_s - elapsed)
            if self._stop.wait(timeout=remaining):
                break


def start_if_enabled(runtime: HiMARuntime | None) -> BudgeterBackgroundTask | None:
    """Start the background task if HiMA L2 is active; returns ``None`` otherwise.

    The budgeter daemon drives the cross-pool planner — an L2 component.
    When L2 is disabled (``runtime.config.hima_l2_enabled`` is False),
    spawning the daemon would just call ``planner_tick`` on a None
    planner; skip it entirely.
    """
    if runtime is None:
        return None
    if not runtime.config.hima_l2_enabled:
        logger.info("HiMA L2 disabled; budgeter daemon not started.")
        return None
    task = BudgeterBackgroundTask(runtime)
    task.start()
    return task


__all__ = ["BudgeterBackgroundTask", "start_if_enabled"]
