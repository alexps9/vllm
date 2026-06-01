# SPDX-License-Identifier: Apache-2.0
"""Replicate ``_align_hybrid_block_size`` arithmetic offline.

Loads vllm_config the way the engine does, then evaluates each input
that feeds the inflate formula. No GPU, no model load.

Run via: .venv/bin/python dev/trace_inflate.py
"""

from __future__ import annotations

import math

import torch
from vllm.engine.arg_utils import EngineArgs

MODEL = "Qwen/Qwen3.5-35B-A3B"


def main() -> None:
    print(f"Resolving config for {MODEL} (TP=2)...")
    args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_model_len=16384,
        trust_remote_code=True,
    )
    vllm_config = args.create_engine_config()

    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    # Reproduce ``_align_hybrid_block_size`` math:
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    print("\n--- attn_page_size_1_token ---")
    spec1 = FullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )
    attn_page_size_1_token = spec1.page_size_bytes
    print(f"  num_kv_heads (per-worker) = {spec1.num_kv_heads}")
    print(f"  head_size                  = {spec1.head_size}")
    print(f"  head_size_v                = {spec1.head_size_v}")
    print(f"  dtype                      = {spec1.dtype}")
    print(f"  -> attn_page_size_1_token  = {attn_page_size_1_token} B")

    print("\n--- mamba_page_size ---")
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture, model_config=model_config,
    )
    shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    print(f"  shapes = {shapes}")
    print(f"  dtypes = {dtypes}")
    spec2 = MambaSpec(shapes=shapes, dtypes=dtypes, block_size=-1)
    mamba_page_size = spec2.page_size_bytes
    print(f"  -> mamba_page_size         = {mamba_page_size} B "
          f"= {mamba_page_size / 1024:.2f} KiB")

    print("\n--- kernel_block_alignment_size ---")
    # Reproduce the lookup:
    from vllm.attention.backends.registry import backend_name_to_enum  # noqa
    from vllm.attention.selector import get_attn_backend
    from vllm.attention.layer import Attention  # noqa: F401  (init side effects)

    try:
        backend_cls = get_attn_backend(
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
            kv_cache_dtype=cache_config.cache_dtype,
            block_size=cache_config.block_size,
            use_mla=model_config.use_mla,
        )
        sup = list(backend_cls.get_supported_kernel_block_sizes())
        print(f"  backend = {backend_cls.get_name()}")
        print(f"  supported_kernel_block_sizes = {sup}")
        # mimic the min() inside _align_hybrid_block_size
        from vllm.attention.backends.utils import MultipleOf  # noqa
        from vllm.attention.backends.abstract import AttentionBackend  # noqa
        flat = []
        for s in sup:
            base = getattr(s, "base", None)
            flat.append(base if base is not None else s)
        backend_min = min(flat)
        kernel_block_alignment_size = max(backend_min, cache_config.block_size)
        print(f"  cache_config.block_size = {cache_config.block_size}")
        print(f"  backend_min = {backend_min}")
        print(f"  -> kernel_block_alignment_size = {kernel_block_alignment_size}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (backend lookup failed: {exc})")
        kernel_block_alignment_size = 16

    print("\n--- inflate formula ---")
    attn_block_size = kernel_block_alignment_size * math.ceil(
        mamba_page_size / (kernel_block_alignment_size * attn_page_size_1_token)
    )
    print(f"  attn_block_size = {kernel_block_alignment_size} × ceil("
          f"{mamba_page_size} / ({kernel_block_alignment_size} × "
          f"{attn_page_size_1_token}))")
    print(f"                 = {kernel_block_alignment_size} × ceil("
          f"{mamba_page_size / (kernel_block_alignment_size * attn_page_size_1_token):.4f})")
    print(f"                 = {attn_block_size}")

    attn_page_size = attn_block_size * attn_page_size_1_token
    padding_pct = 100.0 * (attn_page_size - mamba_page_size) / mamba_page_size
    print(f"  attn_page_size = {attn_block_size} × {attn_page_size_1_token} "
          f"= {attn_page_size} B")
    print(f"  padding pct    = ({attn_page_size} - {mamba_page_size}) "
          f"/ {mamba_page_size} = {padding_pct:.2f}%")


if __name__ == "__main__":
    main()
