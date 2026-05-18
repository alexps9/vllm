# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA-aware KV cache coordinator (opt-in, hybrid models only).

Subclasses ``HybridKVCacheCoordinator`` to add path-counted hit recording
and LPB eviction. Falls back to the legacy coordinator when HiMA is off.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.hima.integration import HiMARuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _hybrid_base():
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    return HybridKVCacheCoordinator


class HiMACoordinator:
    """Deferred subclass of HybridKVCacheCoordinator; use :meth:`build` factory."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        base = _hybrid_base()
        dynamic_cls = type(
            "HiMACoordinatorImpl",
            (base,),
            {
                "__init__": _hima_init,
                "find_longest_cache_hit": _hima_find_longest_cache_hit,
                "cache_blocks": _hima_cache_blocks,
            },
        )
        instance: Any = object.__new__(dynamic_cls)
        return instance

    @classmethod
    def build(
        cls,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: Any | None = None,
        runtime: HiMARuntime | None = None,
    ) -> Any:
        """Preferred factory; mirrors the signature of ``get_kv_cache_coordinator``."""
        return cls(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            runtime=runtime,
        )


# ---------------------------------------------------------------------------
# Mixin methods (bound onto HiMACoordinatorImpl)
# ---------------------------------------------------------------------------


def _hima_init(self: Any, runtime: HiMARuntime | None = None, **kwargs: Any) -> None:
    base = _hybrid_base()
    base.__init__(self, **kwargs)
    self._hima_runtime = runtime
    if runtime is not None:
        logger.info(
            "HiMACoordinator: attached to runtime (cost_curves L*=%d)",
            int(runtime.cost_curves.L_star),
        )


def _hima_find_longest_cache_hit(self: Any, request: Any) -> tuple:  # type: ignore[no-untyped-def]
    """Delegate to parent and forward the full hit chain to the path counter."""
    base = _hybrid_base()
    hits, num_tokens = base.find_longest_cache_hit(self, request)
    runtime = self._hima_runtime
    if runtime is not None and hits:
        block_ids: list[int] = []
        for group_hits in hits:
            for blk in group_hits:
                if not getattr(blk, "is_null", False):
                    block_ids.append(blk.block_id)
        if block_ids:
            runtime.record_hit(block_ids)
    return hits, num_tokens


def _hima_cache_blocks(self: Any, request: Any, num_computed_tokens: int) -> None:
    """Pass-through; hook reserved for future LPB metadata sync."""
    base = _hybrid_base()
    base.cache_blocks(self, request, num_computed_tokens)


__all__ = ["HiMACoordinator"]
