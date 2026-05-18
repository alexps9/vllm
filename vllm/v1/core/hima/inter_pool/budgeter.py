# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper-pure bisection budgeter.

``plan.md`` §3.4 + paper §sec:design-l2. The two-pool case is
one-dimensional, so instead of pulling in ``cvxpy`` we just **bisect**
for the marginal-utility-equalising boundary.

This is the *closed-form* shadow-price equaliser; it does not understand
cooldown / hysteresis / runtime EWMA actuator cost. For the production
deployment path (sglang-aligned, edge-triggered + NB direction-aware
gate) see :class:`vllm.v1.core.hima.inter_pool.CrossPoolPlanner`.
Both produce :class:`MovePlan` so the actuator side is shared.

The class name is :class:`BisectionBudgeter`; ``Budgeter`` remains as
a compatibility alias so existing callers keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vllm.v1.core.hima.config import HiMAConfig, PoolKind

# Marginal utility of giving one more *page* to ``pool``. Telemetry-derived;
# injected as a callable to keep this module independent of HiMATelemetry.
MarginalUtility = Callable[[PoolKind, int], float]


@dataclass(frozen=True)
class MovePlan:
    """A batch cross-pool transfer plan.

    Args:
        src: Pool to take pages from.
        dst: Pool to give pages to.
        n_pages: Number of pages to migrate. ``0`` means "no-op".
    """

    src: PoolKind
    dst: PoolKind
    n_pages: int

    def __post_init__(self) -> None:
        if self.n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {self.n_pages}")
        if self.n_pages > 0 and self.src is self.dst:
            raise ValueError("src and dst must differ for a non-empty plan")

    @property
    def is_noop(self) -> bool:
        return self.n_pages == 0


@dataclass
class BisectionBudgeter:
    """Periodic budget planner (paper-pure bisection).

    Args:
        cfg: HiMA knob bundle (uses ``hima_budget_interval_s``).
        marginal_utility: ``f(pool, allocation_pages)`` returning the
            estimated benefit (e.g. saved us of recompute) of having
            ``allocation_pages`` pages in ``pool``. Must be **monotonically
            non-increasing** in ``allocation_pages``: more memory yields
            diminishing returns. Tests can pass closed-form functions; the
            real engine derives this from telemetry (LPB scores at the
            margin in each pool).
        max_move_per_cycle: Hard cap on pages per Budgeter tick to bound
            actuator pressure between ticks.
    """

    cfg: HiMAConfig
    marginal_utility: MarginalUtility
    max_move_per_cycle: int = 1024

    def plan(
        self,
        kv_pages: int,
        rec_pages: int,
        *,
        page_step: int = 1,
    ) -> MovePlan:
        """Compute a non-negative migration plan given the current allocation.

        Args:
            kv_pages: Current physical pages mapped into the KV pool.
            rec_pages: Current physical pages mapped into the rec pool.
            page_step: Granularity of the search; defaults to 1 page.
        """

        if kv_pages < 0 or rec_pages < 0 or page_step <= 0:
            raise ValueError("page counts must be >= 0 and page_step > 0")

        total = kv_pages + rec_pages
        if total == 0:
            return MovePlan(src=PoolKind.KV, dst=PoolKind.REC, n_pages=0)

        kv_star = self._optimal_kv_split(total, page_step)
        delta = kv_star - kv_pages
        n = abs(delta)
        if n == 0:
            return MovePlan(src=PoolKind.KV, dst=PoolKind.REC, n_pages=0)

        n = min(n, self.max_move_per_cycle)
        # Round down to whole pages.
        n = (n // page_step) * page_step
        if n == 0:
            return MovePlan(src=PoolKind.KV, dst=PoolKind.REC, n_pages=0)

        if delta > 0:
            return MovePlan(src=PoolKind.REC, dst=PoolKind.KV, n_pages=n)
        return MovePlan(src=PoolKind.KV, dst=PoolKind.REC, n_pages=n)

    # -------------------------- internals ------------------------------- #

    def _optimal_kv_split(self, total: int, page_step: int) -> int:
        """Find ``kv*`` equalising marginal utility, by bisection.

        Marginal utility is monotonically non-increasing per the contract
        in the docstring, so ``g(kv) = u_kv(kv) - u_rec(total - kv)`` is
        also non-increasing in ``kv``. We bisect for the largest ``kv``
        with ``g(kv) >= 0``.
        """

        lo, hi = 0, total
        u = self.marginal_utility

        def g(kv: int) -> float:
            return u(PoolKind.KV, max(kv, page_step)) - u(
                PoolKind.REC, max(total - kv, page_step)
            )

        if g(lo) <= 0:
            return 0
        if g(hi) >= 0:
            return total

        while lo + page_step < hi:
            mid = ((lo + hi) // (2 * page_step)) * page_step
            if mid <= lo:
                mid = lo + page_step
            if mid >= hi:
                mid = hi - page_step
            if g(mid) >= 0:
                lo = mid
            else:
                hi = mid
        return lo if abs(g(lo)) <= abs(g(hi)) else hi


def constant_utility(c: float) -> MarginalUtility:
    """Trivial helper for tests: a flat utility function."""

    def _f(_: PoolKind, __: int) -> float:
        return c

    return _f


def reciprocal_utility(scale: float) -> MarginalUtility:
    """``u(pool, k) = scale / k`` -- monotonically decreasing, well-behaved."""

    def _f(_: PoolKind, k: int) -> float:
        return scale / max(k, 1)

    return _f


# Backwards-compatible alias.
Budgeter = BisectionBudgeter


__all__ = [
    "BisectionBudgeter",
    "Budgeter",  # alias
    "MarginalUtility",
    "MovePlan",
    "constant_utility",
    "reciprocal_utility",
]
