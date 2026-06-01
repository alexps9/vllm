# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""interlayer: two-level (sub-block) KV allocator for hybrid models.

Eliminates the page-size bubble — attention packs ``kernel_block_size`` (e.g.
32-token) sub-blocks into mamba-sized physical pages (e.g. 1056) instead of
allocating whole pages it fills only fractionally. Mamba keeps whole pages.

See ``dev/interlayer/`` for the design + feasibility gate, and
``dev/interlayer/1_allocator/PLAN.md`` for the implementation plan. This package
is inert until wired into the attention allocation path (Stage 2, flag-gated).
"""

from vllm.v1.core.interlayer.sub_block_pool import IndexedHeap, SubBlockPool

__all__ = ["SubBlockPool", "IndexedHeap"]
