# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-memory fake actuator for unit tests.

Mirrors :class:`VMMActuator` so that the rest of HiMA (Admitter, Budgeter,
coordinator wiring) can be exercised without CUDA. Production
``cuMemUnmap`` / ``cuMemMap`` actions live in :mod:`vllm.v1.core.hima.
actuator.vmm_pool` (P1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from vllm.v1.core.hima.actuator.vmm_pool import PageHandle, VMMActuator
from vllm.v1.core.hima.config import PoolKind


@dataclass
class InMemoryVMMActuator(VMMActuator):
    """Fake VMM actuator with deterministic, microsecond-style cost model.

    Args:
        kv_pages: Initial pages mapped to the KV pool.
        rec_pages: Initial pages mapped to the recurrent pool.
        page_size_bytes: Used for cost accounting; the fake never touches
            real device memory.
        per_page_remap_us: Synthetic per-page ``cuMemUnmap+cuMemMap`` cost,
            roughly matching plan.md's < 100 us / 2 MiB target on Blackwell.
    """

    kv_pages: int
    rec_pages: int
    page_size_bytes: int = 2 * 1024 * 1024
    per_page_remap_us: float = 5.0

    _free: dict = field(default_factory=dict, init=False, repr=False)
    _remap_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kv_pages < 0 or self.rec_pages < 0:
            raise ValueError("page counts must be >= 0")
        if self.per_page_remap_us < 0:
            raise ValueError("per_page_remap_us must be >= 0")
        self._free = {
            PoolKind.KV: deque(
                PageHandle(handle_id=i, size_bytes=self.page_size_bytes)
                for i in range(self.kv_pages)
            ),
            PoolKind.REC: deque(
                PageHandle(
                    handle_id=self.kv_pages + i,
                    size_bytes=self.page_size_bytes,
                )
                for i in range(self.rec_pages)
            ),
        }

    # ------------------------- VMMActuator API -------------------------- #

    def total_pages(self) -> int:
        return sum(len(q) for q in self._free.values())

    def free_pages(self, pool: PoolKind) -> int:
        return len(self._free[pool])

    def map(self, handle: PageHandle, pool: PoolKind, va_offset: int) -> None:
        # The fake doesn't model individual VA offsets; we just append.
        del va_offset
        self._free[pool].append(handle)

    def unmap(self, pool: PoolKind, va_offset: int) -> PageHandle:
        del va_offset
        if not self._free[pool]:
            raise RuntimeError(f"pool {pool} has no free pages to unmap")
        return self._free[pool].popleft()

    def remap(self, n_pages: int, src: PoolKind, dst: PoolKind) -> int:
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        if src is dst:
            raise ValueError("src and dst must differ")
        moved = 0
        for _ in range(n_pages):
            if not self._free[src]:
                break
            handle = self._free[src].popleft()
            self._free[dst].append(handle)
            moved += 1
        self._remap_count += moved
        return moved

    def remap_cost(self, n_pages: int) -> float:
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        return n_pages * self.per_page_remap_us

    # -------------------------- introspection --------------------------- #

    @property
    def remap_count(self) -> int:
        """Total pages remapped so far -- handy for tests / metrics."""
        return self._remap_count


__all__ = ["InMemoryVMMActuator"]
