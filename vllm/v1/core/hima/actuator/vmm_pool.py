# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM actuator: physical handle pool + two VA windows + atomic remap.

Cross-pool transfer is ``cuMemUnmap`` + ``cuMemMap`` -- zero data movement,
µs latency, does not invalidate CUDA Graphs.
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.v1.core.hima.actuator.cuda_driver import (
    CudaDriver,
    CudaDriverError,
    CudaDriverNotAvailable,
    get_cuda_driver,
)
from vllm.v1.core.hima.config import PoolKind


@dataclass(frozen=True)
class PageHandle:
    """Index into the actuator's handle table + chunk size (JSON-serialisable)."""

    handle_id: int
    size_bytes: int


class VMMActuator(ABC):
    """Abstract page-granularity VMM actuator (CuMemVMMPool or InMemoryVMMActuator)."""

    @abstractmethod
    def total_pages(self) -> int:
        """Total physical pages owned by the actuator."""

    @abstractmethod
    def free_pages(self, pool: PoolKind) -> int:
        """Pages currently mapped to ``pool`` but not yet allocated to a block."""

    @abstractmethod
    def map(self, handle: PageHandle, pool: PoolKind, va_offset: int) -> None:
        """Map ``handle`` into ``pool``'s VA at ``va_offset``."""

    @abstractmethod
    def unmap(self, pool: PoolKind, va_offset: int) -> PageHandle:
        """Unmap the page at ``va_offset`` of ``pool``; return its handle."""

    @abstractmethod
    def remap(
        self,
        n_pages: int,
        src: PoolKind,
        dst: PoolKind,
    ) -> int:
        """Atomically migrate ``n_pages`` from ``src`` to ``dst``.

        Returns the number of pages actually migrated (may be < ``n_pages``
        if ``src`` ran out of free pages mid-batch).
        """

    @abstractmethod
    def remap_cost(self, n_pages: int) -> float:
        """Estimated cost of a ``remap(n_pages, ...)`` call (us)."""


# ---------------------------------------------------------------------------
# Per-pool VA window bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _PoolWindow:
    """One pool's VA window with O(1) free/mapped slot stacks."""

    name: PoolKind
    va_base: int
    n_slots: int
    mapped: list[int | None]
    free_slot_stack: list[int]
    mapped_slot_stack: list[int]

    @classmethod
    def fresh(cls, name: PoolKind, va_base: int, n_slots: int) -> _PoolWindow:
        return cls(
            name=name,
            va_base=va_base,
            n_slots=n_slots,
            mapped=[None] * n_slots,
            free_slot_stack=list(range(n_slots - 1, -1, -1)),
            mapped_slot_stack=[],
        )

    def first_free_slot(self) -> int | None:
        return self.free_slot_stack[-1] if self.free_slot_stack else None

    def last_mapped_slot(self) -> int | None:
        return self.mapped_slot_stack[-1] if self.mapped_slot_stack else None

    def mark_mapped(self, slot: int, handle_idx: int) -> None:
        self.mapped[slot] = handle_idx
        self.mapped_slot_stack.append(slot)
        if self.free_slot_stack and self.free_slot_stack[-1] == slot:
            self.free_slot_stack.pop()
        else:
            with contextlib.suppress(ValueError):
                self.free_slot_stack.remove(slot)

    def mark_unmapped(self, slot: int) -> int:
        handle_idx = self.mapped[slot]
        if handle_idx is None:
            raise RuntimeError(f"pool={self.name} slot={slot} not mapped")
        self.mapped[slot] = None
        if self.mapped_slot_stack and self.mapped_slot_stack[-1] == slot:
            self.mapped_slot_stack.pop()
        else:
            with contextlib.suppress(ValueError):
                self.mapped_slot_stack.remove(slot)
        self.free_slot_stack.append(slot)
        return handle_idx

    def mapped_count(self) -> int:
        return len(self.mapped_slot_stack)


# ---------------------------------------------------------------------------
# CuMemVMMPool
# ---------------------------------------------------------------------------


class CuMemVMMPool(VMMActuator):
    """CUDA VMM handle pool + two VA windows + atomic remap.

    Cross-pool transfer uses ``cuMemUnmap``/``cuMemMap`` -- zero data copy.
    """

    def __init__(
        self,
        n_handles: int,
        kv_slots: int,
        rec_slots: int,
        chunk_size_bytes: int | None = None,
        device_id: int = 0,
        initial_distribution: tuple[int, int] | None = None,
        driver: CudaDriver | None = None,
        ewma_alpha: float = 0.3,
        initial_per_page_cost_us: float = 50.0,
    ) -> None:
        if n_handles <= 0:
            raise ValueError(f"n_handles must be > 0, got {n_handles}")
        if kv_slots <= 0 or rec_slots <= 0:
            raise ValueError(
                f"slot counts must be > 0, got kv={kv_slots}, rec={rec_slots}"
            )
        if n_handles > kv_slots + rec_slots:
            raise ValueError(
                f"n_handles ({n_handles}) > kv_slots ({kv_slots}) + "
                f"rec_slots ({rec_slots}); reduce handle count or grow windows"
            )
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1], got {ewma_alpha}")

        self._driver: CudaDriver = driver if driver is not None else get_cuda_driver()
        self._device_id = device_id

        if not self._driver.vmm_supported(device_id):
            raise CudaDriverNotAvailable(
                f"CUDA VMM not supported on device {device_id}; cannot "
                "instantiate CuMemVMMPool. HiMA requires VMM."
            )

        self._prop = self._driver.make_alloc_prop(device_id)
        probed = self._driver.granularity(self._prop)
        if chunk_size_bytes is None:
            chunk_size_bytes = probed
        elif chunk_size_bytes % probed != 0:
            raise ValueError(
                f"chunk_size_bytes={chunk_size_bytes} is not a multiple of "
                f"the recommended VMM granularity ({probed}). Hard-coding "
                "page sizes is unsafe across architectures -- prefer None."
            )

        self._chunk_size = chunk_size_bytes
        self._kv_slots = kv_slots
        self._rec_slots = rec_slots
        self._n_handles = n_handles
        self._total_va_size = chunk_size_bytes * (kv_slots + rec_slots)

        self._va_base = self._driver.reserve(self._total_va_size, alignment=0)
        kv_base = self._va_base
        rec_base = self._va_base + chunk_size_bytes * kv_slots
        self._windows: dict[PoolKind, _PoolWindow] = {
            PoolKind.KV: _PoolWindow.fresh(PoolKind.KV, kv_base, kv_slots),
            PoolKind.REC: _PoolWindow.fresh(PoolKind.REC, rec_base, rec_slots),
        }

        self._handles: list[int] = []
        try:
            for _ in range(n_handles):
                h = self._driver.mem_create(chunk_size_bytes, self._prop)
                self._handles.append(h)
        except CudaDriverError:
            for h in self._handles:
                with contextlib.suppress(CudaDriverError):
                    self._driver.mem_release(h)
            self._handles.clear()
            with contextlib.suppress(CudaDriverError):
                self._driver.address_free(self._va_base, self._total_va_size)
            raise

        self._free_handles: list[int] = list(range(n_handles))
        self._allocated_pages: dict[PoolKind, int] = {
            PoolKind.KV: 0,
            PoolKind.REC: 0,
        }

        if initial_distribution is None:
            initial_distribution = (min(kv_slots, n_handles), 0)
        n_kv, n_rec = initial_distribution
        if n_kv < 0 or n_rec < 0:
            raise ValueError(
                f"initial_distribution must be non-negative, got ({n_kv}, {n_rec})"
            )
        if n_kv + n_rec > n_handles:
            raise ValueError(
                f"initial_distribution sums to {n_kv + n_rec} but only "
                f"{n_handles} handles exist"
            )
        if n_kv > kv_slots:
            raise ValueError(f"initial KV pages {n_kv} > kv_slots {kv_slots}")
        if n_rec > rec_slots:
            raise ValueError(f"initial REC pages {n_rec} > rec_slots {rec_slots}")

        try:
            self._grow(PoolKind.KV, n_kv)
            self._grow(PoolKind.REC, n_rec)
        except CudaDriverError:
            self.close()
            raise

        if initial_per_page_cost_us <= 0:
            raise ValueError(
                f"initial_per_page_cost_us must be > 0, got {initial_per_page_cost_us}"
            )
        self._ewma_alpha = ewma_alpha
        self._per_page_cost_us = float(initial_per_page_cost_us)
        self._remap_count = 0
        self._closed = False

    # -- VA / shape introspection ------------------------------------ #

    @property
    def chunk_size_bytes(self) -> int:
        return self._chunk_size

    @property
    def device_id(self) -> int:
        return self._device_id

    def va_base(self, pool: PoolKind) -> int:
        """Start VA of ``pool``'s window. Stable for the actuator lifetime."""
        return self._windows[pool].va_base

    def slot_count(self, pool: PoolKind) -> int:
        return self._windows[pool].n_slots

    def mapped_pages(self, pool: PoolKind) -> int:
        return self._windows[pool].mapped_count()

    def allocated_pages(self, pool: PoolKind) -> int:
        return self._allocated_pages[pool]

    # -- VMMActuator API --------------------------------------------- #

    def total_pages(self) -> int:
        return self._n_handles

    def free_pages(self, pool: PoolKind) -> int:
        return self.mapped_pages(pool) - self._allocated_pages[pool]

    def map(self, handle: PageHandle, pool: PoolKind, va_offset: int) -> None:
        """Map ``handle`` to ``va_offset`` inside ``pool``'s VA window."""
        if not 0 <= handle.handle_id < self._n_handles:
            raise ValueError(
                f"handle_id={handle.handle_id} out of range [0, {self._n_handles})"
            )
        if va_offset % self._chunk_size != 0:
            raise ValueError(
                f"va_offset={va_offset} is not a multiple of "
                f"chunk_size={self._chunk_size}"
            )
        win = self._windows[pool]
        slot = va_offset // self._chunk_size
        if not 0 <= slot < win.n_slots:
            raise ValueError(
                f"va_offset corresponds to slot {slot}, outside window "
                f"[0, {win.n_slots})"
            )
        if win.mapped[slot] is not None:
            raise RuntimeError(
                f"pool={pool} slot={slot} already mapped (handle {win.mapped[slot]})"
            )
        try:
            self._free_handles.remove(handle.handle_id)
        except ValueError as exc:
            raise RuntimeError(
                f"handle {handle.handle_id} is not free; unmap from its "
                "current pool first"
            ) from exc
        try:
            self._map_handle_to_slot(win, slot, handle.handle_id)
        except CudaDriverError:
            self._free_handles.append(handle.handle_id)
            raise

    def unmap(self, pool: PoolKind, va_offset: int) -> PageHandle:
        """Unmap the page at ``va_offset`` and return it to the free-handle pool."""
        if va_offset % self._chunk_size != 0:
            raise ValueError(
                f"va_offset={va_offset} is not a multiple of "
                f"chunk_size={self._chunk_size}"
            )
        win = self._windows[pool]
        slot = va_offset // self._chunk_size
        if not 0 <= slot < win.n_slots:
            raise ValueError(
                f"va_offset corresponds to slot {slot}, outside window "
                f"[0, {win.n_slots})"
            )
        handle_idx = self._unmap_slot(win, slot)
        self._free_handles.append(handle_idx)
        return PageHandle(handle_id=handle_idx, size_bytes=self._chunk_size)

    def remap(self, n_pages: int, src: PoolKind, dst: PoolKind) -> int:
        """Move up to ``n_pages`` free pages from ``src`` to ``dst``.

        Returns the count actually moved.
        """
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        if src is dst:
            raise ValueError("src and dst must differ")
        if n_pages == 0:
            return 0

        free_in_src = self.free_pages(src)
        budget = min(n_pages, free_in_src)
        dst_room = self._windows[dst].n_slots - self._windows[dst].mapped_count()
        budget = min(budget, dst_room)
        if budget == 0:
            return 0

        src_win = self._windows[src]
        dst_win = self._windows[dst]

        t0 = time.perf_counter()
        moved = 0
        try:
            for _ in range(budget):
                src_slot = src_win.last_mapped_slot()
                if src_slot is None:
                    break
                dst_slot = dst_win.first_free_slot()
                if dst_slot is None:
                    break
                handle_idx = self._unmap_slot(src_win, src_slot)
                self._map_handle_to_slot(dst_win, dst_slot, handle_idx)
                moved += 1
        finally:
            t1 = time.perf_counter()
            if moved > 0:
                wall_us = (t1 - t0) * 1_000_000.0
                per_page = wall_us / moved
                a = self._ewma_alpha
                self._per_page_cost_us = (
                    a * per_page + (1.0 - a) * self._per_page_cost_us
                )
                self._remap_count += moved

        return moved

    def remap_cost(self, n_pages: int) -> float:
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        return n_pages * self._per_page_cost_us

    # -- allocation accounting --------------------------------------- #

    def allocate(self, pool: PoolKind, n_pages: int = 1) -> int:
        """Mark ``n_pages`` mapped pages as in-use; returns count actually reserved."""
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        granted = min(n_pages, self.free_pages(pool))
        self._allocated_pages[pool] += granted
        return granted

    def release(self, pool: PoolKind, n_pages: int = 1) -> None:
        """Return ``n_pages`` back to the free pool (pages remain mapped)."""
        if n_pages < 0:
            raise ValueError(f"n_pages must be >= 0, got {n_pages}")
        if n_pages > self._allocated_pages[pool]:
            raise RuntimeError(
                f"release({n_pages}, {pool}) exceeds allocated "
                f"{self._allocated_pages[pool]}"
            )
        self._allocated_pages[pool] -= n_pages

    # -- diagnostics ------------------------------------------------- #

    @property
    def remap_count(self) -> int:
        return self._remap_count

    @property
    def per_page_cost_us(self) -> float:
        """Current EWMA estimate of per-page remap latency."""
        return self._per_page_cost_us

    # -- lifetime ---------------------------------------------------- #

    def close(self) -> None:
        """Unmap all pages, release handles, free VA range. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # pragma: no cover - all branches below are best-effort cleanup paths
        for pool, win in self._windows.items():
            for slot, handle_idx in enumerate(win.mapped):
                if handle_idx is None:
                    continue
                va = win.va_base + slot * self._chunk_size
                with contextlib.suppress(CudaDriverError):
                    self._driver.mem_unmap(va, self._chunk_size)
            win.mapped = [None] * win.n_slots
            win.free_slot_stack = list(range(win.n_slots - 1, -1, -1))
            win.mapped_slot_stack = []
        for h in self._handles:
            with contextlib.suppress(CudaDriverError):
                self._driver.mem_release(h)
        self._handles.clear()
        with contextlib.suppress(CudaDriverError):
            self._driver.address_free(self._va_base, self._total_va_size)

    def __del__(self) -> None:  # pragma: no cover - GC timing
        with contextlib.suppress(Exception):
            self.close()

    # -- internal helpers -------------------------------------------- #

    def _map_handle_to_slot(
        self,
        win: _PoolWindow,
        slot: int,
        handle_idx: int,
    ) -> None:
        va = win.va_base + slot * self._chunk_size
        self._driver.mem_map(va, self._chunk_size, self._handles[handle_idx])
        self._driver.mem_set_access(va, self._chunk_size, self._device_id)
        win.mark_mapped(slot, handle_idx)

    def _unmap_slot(self, win: _PoolWindow, slot: int) -> int:
        va = win.va_base + slot * self._chunk_size
        handle_idx = win.mark_unmapped(slot)
        try:
            self._driver.mem_unmap(va, self._chunk_size)
        except CudaDriverError:
            win.mark_mapped(slot, handle_idx)  # roll back bookkeeping
            raise
        return handle_idx

    def _grow(self, pool: PoolKind, n: int) -> int:
        """Map ``n`` free handles into ``pool``'s first free slots."""
        if n <= 0:
            return 0
        win = self._windows[pool]
        mapped = 0
        for _ in range(n):
            if not self._free_handles:
                break
            slot = win.first_free_slot()
            if slot is None:
                break
            handle_idx = self._free_handles.pop()
            self._map_handle_to_slot(win, slot, handle_idx)
            mapped += 1
        return mapped


__all__ = ["CuMemVMMPool", "PageHandle", "VMMActuator"]
