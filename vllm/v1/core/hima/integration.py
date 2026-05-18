# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA integration façade -- single seam between vLLM hot paths and the HiMA core.

Hot paths call ``get_runtime()`` (returns ``None`` when HiMA is off).
HiMA is disabled by default; the runtime is built once via ``enable_runtime()``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.hima.actuator import (
    CudaDriverNotAvailable,
    CuMemVMMPool,
    InMemoryVMMActuator,
    VMMActuator,
)
from vllm.v1.core.hima.config import HiMAConfig, PoolKind
from vllm.v1.core.hima.inter_pool import (
    AdmissionAction,
    AdmissionDecision,
    Admitter,
    BisectionBudgeter,
    CrossPoolPlanner,
    CrossPoolPolicyConfig,
    MovePlan,
    PlanDecision,
    VLLMPressureAdapter,
    constant_utility,
    policy_from_env,
)
from vllm.v1.core.hima.intra_pool import (
    CostCurves,
    LPBPriorityQueue,
    PathCountedHitCounter,
    RuntimeActuatorCost,
    get_cost_curves,
    get_runtime_actuator_cost,
)
from vllm.v1.core.hima.telemetry import HiMATelemetry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class HiMARuntime:
    """Process-global container for all HiMA components.

    Built once at engine start via :func:`enable_runtime`.
    ``lock`` guards actuator + intra_queues mutations across admitter/budgeter threads.
    """

    config: HiMAConfig
    actuator: VMMActuator
    cost_curves: CostCurves
    runtime_cost: RuntimeActuatorCost
    path_counter: PathCountedHitCounter
    intra_queues: dict[PoolKind, LPBPriorityQueue] = field(default_factory=dict)
    telemetry: HiMATelemetry = field(default_factory=HiMATelemetry)
    pressure_adapter: VLLMPressureAdapter = field(default_factory=VLLMPressureAdapter)
    admitter: Admitter | None = None
    planner: CrossPoolPlanner | None = None
    budgeter: BisectionBudgeter | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # --- decision counters (for metrics) ------------------------------ #
    decisions: dict[str, int] = field(
        default_factory=lambda: {
            "own_free": 0,
            "own_evict": 0,
            "cross_free": 0,
            "cross_evict": 0,
            "defer": 0,
            "planner_fired": 0,
            "planner_skipped": 0,
        }
    )

    # LPBFreeBlockQueue instances registered by BlockPool (one per group).
    _lpb_queues: list[Any] = field(default_factory=list)

    # --------------- intra-pool helpers (used by BlockPool) --------------- #

    def get_or_create_queue(self, pool: PoolKind) -> LPBPriorityQueue:
        q = self.intra_queues.get(pool)
        if q is None:
            q = LPBPriorityQueue()
            self.intra_queues[pool] = q
        return q

    def register_lpb_queue(self, queue: Any) -> None:
        """Register a LPBFreeBlockQueue so record_hit can push depth updates."""
        if queue not in self._lpb_queues:
            self._lpb_queues.append(queue)

    def record_hit(self, path_block_ids: list[int]) -> None:
        """Record a prefix-cache hit; update path-counts and LPB depths."""
        self.path_counter.record_hit(tuple(path_block_ids))
        # Push depth info into every registered LPBFreeBlockQueue.
        for depth, block_id in enumerate(path_block_ids, start=1):
            for q in self._lpb_queues:
                q.set_block_depth(block_id, depth)

    # --------------- admission (used by Scheduler) ------------------------ #

    def decide_admission(
        self,
        target_pool: PoolKind,
        n_pages: int,
    ) -> AdmissionDecision:
        """Run 5-candidate admission logic; returns action + estimated cost (µs)."""
        assert self.admitter is not None  # built in __post_init__
        with self.lock:
            decision = self.admitter.decide(
                target_pool=target_pool,
                n_pages=n_pages,
            )
        self.decisions[decision.action.value] += 1
        return decision

    # --------------- planner tick (used by background task) -------------- #

    def planner_tick(self, snapshot: dict[str, float] | None = None) -> PlanDecision:
        """Run one Cross-Pool Planner step (thread-safe via lock)."""
        assert self.planner is not None
        snap = snapshot if snapshot is not None else self.telemetry.snapshot()
        with self.lock:
            decision = self.planner.decide(
                usage_kv=snap.get("usage_kv", 0.0),
                usage_rec=snap.get("usage_rec", 0.0),
                queue_depth=int(snap.get("num_queue_reqs", 0)),
                snapshot=snap,
                edge_active=bool(snap.get("edge_active", False)),
            )
            if decision.fired and decision.move is not None:
                moved = self.actuator.remap(
                    n_pages=decision.move.n_pages,
                    src=decision.move.src,
                    dst=decision.move.dst,
                )
                self.decisions["planner_fired"] += 1
                decision = PlanDecision(
                    move=MovePlan(
                        src=decision.move.src,
                        dst=decision.move.dst,
                        n_pages=moved,
                    ),
                    reason=decision.reason,
                    usage_kv=decision.usage_kv,
                    usage_rec=decision.usage_rec,
                    queue_depth=decision.queue_depth,
                )
            else:
                self.decisions["planner_skipped"] += 1
        return decision

    # ------------------------ shutdown ---------------------------------- #

    def shutdown(self) -> None:
        """Release VMM resources held by the actuator. Safe to call multiple times."""
        actuator = self.actuator
        close = getattr(actuator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.warning("HiMARuntime.shutdown: actuator.close raised; ignoring")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: HiMARuntime | None = None


def is_enabled() -> bool:
    """Returns ``False`` until :func:`enable_runtime` is called."""
    return _RUNTIME is not None


def get_runtime() -> HiMARuntime | None:
    """Return the active runtime, or ``None`` when HiMA is disabled."""
    return _RUNTIME


def enable_runtime(
    config: HiMAConfig | None = None,
    actuator: VMMActuator | None = None,
    page_size_bytes_override: int | None = None,
    planner_policy: CrossPoolPolicyConfig | None = None,
    n_pages: int | None = None,
    kv_slots: int | None = None,
    rec_slots: int | None = None,
    device_id: int = 0,
) -> HiMARuntime:
    """Build (or return existing) the HiMA runtime.

    Actuator priority: caller-supplied → CuMemVMMPool → InMemoryVMMActuator (fallback).
    """
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and actuator is None:
            return _RUNTIME

        cfg = config if config is not None else HiMAConfig.from_env()
        chunk_size = page_size_bytes_override or cfg.hima_page_size_bytes

        # 1) Actuator
        if actuator is None:
            if n_pages is not None and kv_slots is not None and rec_slots is not None:
                try:
                    actuator = CuMemVMMPool(
                        n_handles=n_pages,
                        kv_slots=kv_slots,
                        rec_slots=rec_slots,
                        chunk_size_bytes=chunk_size,
                        device_id=device_id,
                    )
                    logger.info(
                        "HiMA actuator: CuMemVMMPool live (device=%d, "
                        "n_pages=%d, chunk=%d KiB)",
                        device_id,
                        n_pages,
                        actuator.chunk_size_bytes // 1024,
                    )
                except (CudaDriverNotAvailable, Exception) as exc:  # noqa: BLE001
                    logger.warning(
                        "HiMA: CuMemVMMPool unavailable (%s); falling back to "
                        "InMemoryVMMActuator. Cross-pool remaps are best-effort.",
                        exc,
                    )
                    actuator = InMemoryVMMActuator(
                        kv_pages=kv_slots,
                        rec_pages=rec_slots,
                        page_size_bytes=chunk_size or 2 * 1024 * 1024,
                    )
            else:
                actuator = InMemoryVMMActuator(
                    kv_pages=0,
                    rec_pages=0,
                    page_size_bytes=chunk_size or 2 * 1024 * 1024,
                )

        curves = get_cost_curves()
        runtime_cost = get_runtime_actuator_cost()
        path_counter = PathCountedHitCounter(window_seconds=cfg.hima_lpb_window_s)

        telemetry = HiMATelemetry(alpha=cfg.ewma_alpha)
        adapter = VLLMPressureAdapter()
        policy = planner_policy if planner_policy is not None else policy_from_env()
        planner = CrossPoolPlanner(config=policy, adapter=adapter)

        for pool in PoolKind:
            telemetry.observe_pool(pool, free_pages=actuator.free_pages(pool))

        def _evict_cost(pool: PoolKind, n: int) -> float:
            """Estimated µs cost of evicting the n cheapest LPB blocks in pool."""
            q = _RUNTIME.intra_queues.get(pool) if _RUNTIME is not None else None
            if q is None or q.is_empty():
                return 0.0
            cheapest_scores = q.peek_n_scores(min(n, len(q)))
            chunk = getattr(actuator, "chunk_size_bytes", 2 * 1024 * 1024)
            return float(sum(s * chunk for s in cheapest_scores))

        admitter = Admitter(
            cfg=cfg,
            telemetry=telemetry,
            peek_evict_cost=_evict_cost,
            peek_remap_cost=lambda n: actuator.remap_cost(n),
        )

        budgeter = BisectionBudgeter(
            cfg=cfg, marginal_utility=constant_utility(0.0), max_move_per_cycle=1024
        )

        runtime = HiMARuntime(
            config=cfg,
            actuator=actuator,
            cost_curves=curves,
            runtime_cost=runtime_cost,
            path_counter=path_counter,
            telemetry=telemetry,
            pressure_adapter=adapter,
            admitter=admitter,
            planner=planner,
            budgeter=budgeter,
        )
        _RUNTIME = runtime
        return runtime


def disable_runtime() -> None:
    """Tear down the runtime. Test-only / shutdown helper."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            _RUNTIME.shutdown()
            _RUNTIME = None


# ---------------------------------------------------------------------------
# Hot-path helpers (thin & cheap)
# ---------------------------------------------------------------------------


def maybe_decide_admission(
    target_pool: PoolKind,
    n_pages: int,
) -> AdmissionDecision | None:
    """Returns ``None`` when HiMA is disabled (legacy path preserved)."""
    if _RUNTIME is None:
        return None
    return _RUNTIME.decide_admission(target_pool, n_pages)


def maybe_record_hit(path_block_ids: list[int]) -> None:
    """Hot-path-safe path-counted hit recording. No-op when disabled."""
    if _RUNTIME is None:
        return
    _RUNTIME.record_hit(path_block_ids)


def maybe_get_free_queue_factory() -> Any | None:
    """Returns the LPBFreeBlockQueue class when HiMA is on, else ``None``."""
    if _RUNTIME is None:
        return None
    from vllm.v1.core.hima.lpb_free_queue import LPBFreeBlockQueue  # noqa: PLC0415

    return LPBFreeBlockQueue


__all__ = [
    "AdmissionAction",
    "AdmissionDecision",
    "HiMARuntime",
    "MovePlan",
    "PlanDecision",
    "disable_runtime",
    "enable_runtime",
    "get_runtime",
    "is_enabled",
    "maybe_decide_admission",
    "maybe_get_free_queue_factory",
    "maybe_record_hit",
]
