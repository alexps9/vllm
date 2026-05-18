# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM actuator layer for HiMA.

* :class:`VMMActuator` -- the abstract contract.
* :class:`CuMemVMMPool` -- CUDA-backed real implementation (P1).
* :class:`InMemoryVMMActuator` -- test fake mirroring the contract.
* :class:`CudaDriver` -- lazy ``ctypes`` wrapper over ``libcuda.so``.
"""

from vllm.v1.core.hima.actuator.cuda_driver import (
    CudaDriver,
    CudaDriverError,
    CudaDriverNotAvailable,
    get_cuda_driver,
    probe_vmm_environment,
    reset_cuda_driver,
)
from vllm.v1.core.hima.actuator.remap import InMemoryVMMActuator
from vllm.v1.core.hima.actuator.vmm_pool import (
    CuMemVMMPool,
    PageHandle,
    VMMActuator,
)

__all__ = [
    "CuMemVMMPool",
    "CudaDriver",
    "CudaDriverError",
    "CudaDriverNotAvailable",
    "InMemoryVMMActuator",
    "PageHandle",
    "VMMActuator",
    "get_cuda_driver",
    "probe_vmm_environment",
    "reset_cuda_driver",
]
