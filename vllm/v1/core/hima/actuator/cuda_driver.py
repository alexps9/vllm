# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy ctypes bindings for ``libcuda.so`` VMM APIs (cuMemCreate/Map/Unmap/etc.).

Driver loading is deferred to the first ``get_cuda_driver()`` call so this
module is safe to import on CPU-only hosts.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading

# ---------------------------------------------------------------------------
# CUDA constants (cuda.h)
# ---------------------------------------------------------------------------

CU_SUCCESS = 0

# cuMemCreate properties
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

# cuMemGetAllocationGranularity flags
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

# cuDeviceGetAttribute selectors we care about
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102


# ---------------------------------------------------------------------------
# Struct definitions (CUDA driver ABI)
# ---------------------------------------------------------------------------


class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int),
    ]


# CUmemGenericAllocationHandle and CUdeviceptr are both 64-bit unsigned.
CU_HANDLE = ctypes.c_ulonglong
CU_DPTR = ctypes.c_ulonglong


class CudaDriverNotAvailable(RuntimeError):
    """Raised when ``libcuda.so`` cannot be loaded or the device lacks VMM support."""


class CudaDriverError(RuntimeError):
    """A CUDA driver call returned a non-zero status."""

    def __init__(self, rc: int, what: str, message: str = "") -> None:
        self.rc = rc
        self.what = what
        suffix = f" -- {message}" if message else ""
        super().__init__(f"{what} failed: rc={rc}{suffix}")


# ---------------------------------------------------------------------------
# Driver singleton (lazy)
# ---------------------------------------------------------------------------


class CudaDriver:
    """Typed ctypes wrapper for the libcuda.so entry points HiMA needs."""

    def __init__(self, library_name: str | None = None) -> None:
        if library_name is not None:
            try:
                self.cuda = ctypes.CDLL(library_name)
            except OSError as exc:
                raise CudaDriverNotAvailable(
                    f"failed to load CUDA driver from {library_name!r}: {exc}"
                ) from exc
        else:
            try:
                self.cuda = ctypes.CDLL("libcuda.so")
            except OSError:
                try:
                    self.cuda = ctypes.CDLL("libcuda.so.1")
                except OSError as exc:  # pragma: no cover - host-specific
                    raise CudaDriverNotAvailable(
                        "libcuda.so not found; install NVIDIA driver >= R570 "
                        "for Blackwell VMM support"
                    ) from exc

        self._setup_argtypes()
        self._check(self.cuda.cuInit(0), "cuInit")

    # -- ctypes argtype wiring ---------------------------------------- #

    def _setup_argtypes(self) -> None:
        c = self.cuda
        c.cuInit.argtypes = [ctypes.c_uint]
        c.cuInit.restype = ctypes.c_int

        c.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        c.cuDeviceGet.restype = ctypes.c_int

        c.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        c.cuDeviceGetCount.restype = ctypes.c_int

        c.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        c.cuDeviceGetAttribute.restype = ctypes.c_int

        c.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        c.cuMemGetAllocationGranularity.restype = ctypes.c_int

        c.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(CU_DPTR),
            ctypes.c_size_t,
            ctypes.c_size_t,
            CU_DPTR,
            ctypes.c_ulonglong,
        ]
        c.cuMemAddressReserve.restype = ctypes.c_int

        c.cuMemAddressFree.argtypes = [CU_DPTR, ctypes.c_size_t]
        c.cuMemAddressFree.restype = ctypes.c_int

        c.cuMemCreate.argtypes = [
            ctypes.POINTER(CU_HANDLE),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        c.cuMemCreate.restype = ctypes.c_int

        c.cuMemRelease.argtypes = [CU_HANDLE]
        c.cuMemRelease.restype = ctypes.c_int

        c.cuMemMap.argtypes = [
            CU_DPTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            CU_HANDLE,
            ctypes.c_ulonglong,
        ]
        c.cuMemMap.restype = ctypes.c_int

        c.cuMemUnmap.argtypes = [CU_DPTR, ctypes.c_size_t]
        c.cuMemUnmap.restype = ctypes.c_int

        c.cuMemSetAccess.argtypes = [
            CU_DPTR,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        c.cuMemSetAccess.restype = ctypes.c_int

        c.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        c.cuGetErrorString.restype = ctypes.c_int

        c.cuCtxSynchronize.argtypes = []
        c.cuCtxSynchronize.restype = ctypes.c_int

    # -- error helper ------------------------------------------------- #

    def _check(self, rc: int, what: str) -> None:
        if rc == CU_SUCCESS:
            return
        msg = ctypes.c_char_p()
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            self.cuda.cuGetErrorString(rc, ctypes.byref(msg))
        decoded = msg.value.decode() if msg.value else ""
        raise CudaDriverError(rc, what, decoded)

    # -- high-level helpers ------------------------------------------ #

    def device_count(self) -> int:
        n = ctypes.c_int()
        self._check(
            self.cuda.cuDeviceGetCount(ctypes.byref(n)),
            "cuDeviceGetCount",
        )
        return n.value

    def device_get(self, ordinal: int) -> int:
        dev = ctypes.c_int()
        self._check(
            self.cuda.cuDeviceGet(ctypes.byref(dev), ordinal),
            f"cuDeviceGet({ordinal})",
        )
        return dev.value

    def vmm_supported(self, device_id: int) -> bool:
        attr = ctypes.c_int()
        self._check(
            self.cuda.cuDeviceGetAttribute(
                ctypes.byref(attr),
                CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device_id,
            ),
            "cuDeviceGetAttribute(VMM_SUPPORTED)",
        )
        return bool(attr.value)

    def make_alloc_prop(self, device_id: int) -> CUmemAllocationProp:
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        return prop

    def granularity(
        self,
        prop: CUmemAllocationProp,
        flag: int = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    ) -> int:
        g = ctypes.c_size_t()
        self._check(
            self.cuda.cuMemGetAllocationGranularity(
                ctypes.byref(g), ctypes.byref(prop), flag
            ),
            "cuMemGetAllocationGranularity",
        )
        return g.value

    def reserve(self, size_bytes: int, alignment: int = 0) -> int:
        """Reserve a contiguous virtual address range; returns base ptr."""
        ptr = CU_DPTR(0)
        self._check(
            self.cuda.cuMemAddressReserve(
                ctypes.byref(ptr),
                size_bytes,
                alignment,
                CU_DPTR(0),
                0,
            ),
            f"cuMemAddressReserve({size_bytes})",
        )
        return ptr.value

    def address_free(self, ptr: int, size_bytes: int) -> None:
        self._check(
            self.cuda.cuMemAddressFree(ptr, size_bytes),
            "cuMemAddressFree",
        )

    def mem_create(self, size_bytes: int, prop: CUmemAllocationProp) -> int:
        h = CU_HANDLE(0)
        self._check(
            self.cuda.cuMemCreate(ctypes.byref(h), size_bytes, ctypes.byref(prop), 0),
            "cuMemCreate",
        )
        return h.value

    def mem_release(self, handle: int) -> None:
        self._check(self.cuda.cuMemRelease(handle), "cuMemRelease")

    def mem_map(self, va: int, size_bytes: int, handle: int) -> None:
        self._check(
            self.cuda.cuMemMap(va, size_bytes, 0, handle, 0),
            f"cuMemMap(va=0x{va:x}, size={size_bytes})",
        )

    def mem_unmap(self, va: int, size_bytes: int) -> None:
        self._check(
            self.cuda.cuMemUnmap(va, size_bytes),
            f"cuMemUnmap(va=0x{va:x}, size={size_bytes})",
        )

    def mem_set_access(
        self,
        va: int,
        size_bytes: int,
        device_id: int,
        flags: int = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    ) -> None:
        desc = CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_id
        desc.flags = flags
        self._check(
            self.cuda.cuMemSetAccess(va, size_bytes, ctypes.byref(desc), 1),
            "cuMemSetAccess",
        )

    def ctx_synchronize(self) -> None:
        self._check(self.cuda.cuCtxSynchronize(), "cuCtxSynchronize")


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------


_DRIVER_LOCK = threading.Lock()
_DRIVER: CudaDriver | None = None


def get_cuda_driver(library_name: str | None = None) -> CudaDriver:
    """Return the process-wide :class:`CudaDriver` singleton.

    Loads ``libcuda.so`` on the first call; subsequent calls are O(1).
    """
    global _DRIVER
    if _DRIVER is None or library_name is not None:
        with _DRIVER_LOCK:
            if _DRIVER is None or library_name is not None:
                _DRIVER = CudaDriver(library_name=library_name)
    return _DRIVER


def reset_cuda_driver() -> None:
    """Drop the singleton -- intended for test cleanup."""
    global _DRIVER
    with _DRIVER_LOCK:
        _DRIVER = None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def probe_vmm_environment(device_id: int = 0) -> dict[str, object]:
    """Snapshot local VMM environment for diagnostics; never raises."""
    info: dict[str, object] = {
        "available": False,
        "driver_version": "n/a",
        "library": os.environ.get("VLLM_HIMA_CUDA_LIB", "libcuda.so"),
    }
    try:
        driver = get_cuda_driver(library_name=info["library"])  # type: ignore[arg-type]
    except CudaDriverNotAvailable as exc:
        info["error"] = str(exc)
        return info
    except CudaDriverError as exc:  # pragma: no cover - host-specific
        info["error"] = str(exc)
        return info

    info["available"] = True
    try:
        info["device_count"] = driver.device_count()
        dev = driver.device_get(device_id)
        info["device_id"] = dev
        info["vmm_supported"] = driver.vmm_supported(dev)
        prop = driver.make_alloc_prop(dev)
        info["granularity_recommended"] = driver.granularity(prop)
        info["granularity_minimum"] = driver.granularity(
            prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    except CudaDriverError as exc:  # pragma: no cover - host-specific
        info["error"] = str(exc)
    return info


__all__ = [
    "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
    "CU_DPTR",
    "CU_HANDLE",
    "CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
    "CU_MEM_ALLOCATION_TYPE_PINNED",
    "CU_MEM_ALLOC_GRANULARITY_MINIMUM",
    "CU_MEM_ALLOC_GRANULARITY_RECOMMENDED",
    "CU_MEM_LOCATION_TYPE_DEVICE",
    "CU_SUCCESS",
    "CUmemAccessDesc",
    "CUmemAllocationProp",
    "CUmemLocation",
    "CudaDriver",
    "CudaDriverError",
    "CudaDriverNotAvailable",
    "get_cuda_driver",
    "probe_vmm_environment",
    "reset_cuda_driver",
]
