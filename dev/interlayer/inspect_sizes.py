# SPDX-License-Identifier: Apache-2.0
"""Inspect actual KV cache sizing for Qwen3-Next-80B-A3B-Instruct.

Computes:
  * attn_page_size_1_token (bytes per token, full attention)
  * mamba_page_size (bytes, natural — one Gated DeltaNet state snapshot)
  * the inflated block_size that vLLM would force
  * the resulting attn_page_size, mamba_page_size_padded, padding pct
  * lcm_block_size (the prefix-cache hit granularity)

Uses the *exact* arithmetic from vllm.platforms.interface.check_and_update_config
and the gated_delta_net_state_shape function in vllm.model_executor.layers.mamba.mamba_utils.

Run via: .venv/bin/python dev/inspect_sizes.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

# Local model snapshots (already downloaded). Use the first snapshot dir found.
def _first_snapshot(model_dir: str) -> str | None:
    p = Path(model_dir) / "snapshots"
    if not p.is_dir():
        return None
    children = [c for c in p.iterdir() if c.is_dir()]
    return str(children[0]) if children else None


MODELS: dict[str, str | None] = {
    "qwen3-next-80b": _first_snapshot(
        "/scratch/yuzhou/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-Next-80B-A3B-Instruct"
    ),
    "qwen3.5-35b": _first_snapshot(
        "/scratch/yuzhou/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-35B-A3B"
    ),
    "qwen3.5-35b-base": _first_snapshot(
        "/scratch/yuzhou/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-35B-A3B-Base"
    ),
}


def dtype_bytes(dt: torch.dtype) -> int:
    return torch.tensor([], dtype=dt).element_size()


def gated_delta_net_state_shape(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    tp: int = 1,
    num_spec: int = 0,
):
    """Verbatim from vllm.model_executor.layers.mamba.mamba_utils.gated_delta_net_state_shape."""
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    conv_state_shape = (conv_dim // tp, conv_kernel_size - 1 + num_spec)
    temporal_state_shape = (num_v_heads // tp, head_v_dim, head_k_dim)
    return conv_state_shape, temporal_state_shape


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def inspect_one(model_name: str, model_path: str) -> None:
    banner(f"MODEL: {model_name}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Qwen3.5 nests its language params under text_config (it's a VL model).
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    layer_types = text_cfg.layer_types
    n_full_attn_layers = sum(1 for lt in layer_types if lt == "full_attention")
    n_linear_layers = sum(1 for lt in layer_types if lt == "linear_attention")
    print(f"Path: {model_path}")
    print(f"  architectures           = {cfg.architectures}")
    print(f"  num_hidden_layers       = {text_cfg.num_hidden_layers}")
    print(f"  n_full_attn_layers      = {n_full_attn_layers}")
    print(f"  n_linear_layers         = {n_linear_layers}")

    # ----------------------------- attention ----------------------------- #
    banner("Attention (full attention layers)")

    num_kv_heads = text_cfg.num_key_value_heads  # 2 (GQA)
    head_size = text_cfg.head_dim                # 256
    head_size_v = head_size                 # MHA-symmetric for Qwen3-Next
    dtype = torch.bfloat16                  # config.torch_dtype
    bytes_per_token = num_kv_heads * (head_size + head_size_v) * dtype_bytes(dtype)
    # cross-check with FullAttentionSpec.real_page_size_bytes
    # = block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
    print(f"  num_kv_heads            = {num_kv_heads}")
    print(f"  head_size               = {head_size}  (head_size_v = {head_size_v})")
    print(f"  dtype                   = {dtype}  ({dtype_bytes(dtype)} B/elem)")
    print(f"  attn_page_size_1_token  = {bytes_per_token} B "
          f"= {bytes_per_token / 1024:.2f} KiB")
    # one token of KV across all attention layers = bytes_per_token * n_full_attn_layers
    full_attn_per_token = bytes_per_token * n_full_attn_layers
    print(f"  per-token KV (all attn layers) "
          f"= {full_attn_per_token} B = {full_attn_per_token / 1024:.2f} KiB")

    # ------------------------------- mamba ------------------------------- #
    banner("Linear attention (Gated DeltaNet) state")

    conv_shape, temporal_shape = gated_delta_net_state_shape(
        num_k_heads=text_cfg.linear_num_key_heads,
        num_v_heads=text_cfg.linear_num_value_heads,
        head_k_dim=text_cfg.linear_key_head_dim,
        head_v_dim=text_cfg.linear_value_head_dim,
        conv_kernel_size=text_cfg.linear_conv_kernel_dim,
        tp=1,
        num_spec=0,
    )
    print(f"  conv_state_shape        = {conv_shape}   (elems = {prod(conv_shape)})")
    print(f"  temporal_state_shape    = {temporal_shape} (elems = {prod(temporal_shape)})")

    # IMPORTANT: conv_state is in model dtype (bf16), but the SSM temporal
    # state is in fp32 by default — for numerical stability of the recurrent
    # accumulation. See vllm.model_executor.layers.mamba.mamba_utils
    # _mamba_state_dtype + gated_delta_net_state_dtype.
    conv_dtype = torch.bfloat16
    temporal_dtype = torch.float32
    conv_bytes = prod(conv_shape) * dtype_bytes(conv_dtype)
    temporal_bytes = prod(temporal_shape) * dtype_bytes(temporal_dtype)
    mamba_page_size_natural = conv_bytes + temporal_bytes
    print(f"  conv bytes  (dtype={conv_dtype}) = "
          f"{conv_bytes} = {conv_bytes / 1024:.2f} KiB")
    print(f"  temporal bytes (dtype={temporal_dtype}) = "
          f"{temporal_bytes} = {temporal_bytes / 1024:.2f} KiB")
    print(f"  mamba_page_size (1 lyr) = {mamba_page_size_natural} B "
          f"= {mamba_page_size_natural / 1024:.2f} KiB")

    total_mamba_per_req = mamba_page_size_natural * n_linear_layers
    print(f"  total mamba state/req   = {total_mamba_per_req} B "
          f"= {total_mamba_per_req / 1024 / 1024:.2f} MiB")

    # --------------------------- inflate logic --------------------------- #
    banner("vLLM hybrid inflate logic (platforms/interface.py:613-674)")
    # Note: vLLM treats EACH layer's KV cache as ONE entry in the BlockPool.
    # So the "per-block bytes" budget is *per layer*, not stack-aggregated.
    # The match is: attn_page_size (1 layer) >= mamba_page_size (1 layer).

    # Use a conservative kernel_block_alignment_size; most attention backends use 16.
    kernel_block_alignment_size = 16
    user_block_size = 16  # vLLM default
    mamba_page = mamba_page_size_natural

    # Step 1: inflate attention block_size so attn_page_size >= mamba_page
    attn_tokens_per_mamba_state = math.ceil(mamba_page / bytes_per_token)
    attn_block_size_inflated = kernel_block_alignment_size * math.ceil(
        mamba_page / (kernel_block_alignment_size * bytes_per_token)
    )
    print(f"  kernel_block_alignment  = {kernel_block_alignment_size}")
    print(f"  user-requested block_sz = {user_block_size} tokens")
    print(f"  natural attn_tokens/mamba_state = {attn_tokens_per_mamba_state}")
    print(f"  inflated attn_block_sz  = {attn_block_size_inflated} tokens "
          f"(×{attn_block_size_inflated/user_block_size:.1f} over default)")

    final_block_size = max(user_block_size, attn_block_size_inflated)
    attn_page_size = final_block_size * bytes_per_token
    print(f"  final block_size        = {final_block_size} tokens")
    print(f"  attn_page_size (1 lyr)  = {attn_page_size} B "
          f"= {attn_page_size / 1024:.2f} KiB")

    # Step 2: pad mamba page up to attn_page_size
    mamba_page_size_padded = attn_page_size
    padding_bytes = mamba_page_size_padded - mamba_page
    padding_pct = 100.0 * padding_bytes / mamba_page_size_padded
    print(f"  mamba_page_size_padded  = {mamba_page_size_padded} B "
          f"= {mamba_page_size_padded / 1024:.2f} KiB")
    print(f"  mamba padding waste     = {padding_bytes} B "
          f"({padding_pct:.1f}% of block)")

    # ----------------------- prefix cache granularity ----------------------- #
    banner("Prefix cache granularity")
    lcm_block_size = math.lcm(final_block_size, final_block_size)  # both groups equal
    print(f"  group_block_sizes       = [{final_block_size}, {final_block_size}]  "
          f"(attn, mamba — equal after inflate)")
    print(f"  lcm_block_size          = {lcm_block_size}  "
          f"(== minimum non-zero hit length)")
    print(f"  hash_block_size default = gcd = {final_block_size}")

    # -------------------- hit-rate prediction table -------------------- #
    banner("Predicted prefix-cache hit length vs shared-prefix length")
    print(f"  (formula: hit = floor(L / {lcm_block_size}) × {lcm_block_size})\n")
    print("  shared_prefix_tokens | hit_tokens | wasted_tokens | wasted_pct")
    print("  -------------------- | ---------- | ------------- | ----------")
    for L in [100, 200, 500, 1000, 1500, 2000, 3000, 5000, 8000, 10000, 16000]:
        hit = (L // lcm_block_size) * lcm_block_size
        wasted = L - hit
        pct = 100.0 * wasted / L if L > 0 else 0.0
        print(f"  {L:>20} | {hit:>10} | {wasted:>13} | {pct:>7.1f}%")


def main() -> None:
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for name in selected:
        path = MODELS.get(name)
        if path is None:
            print(f"[skip] {name}: not found on disk")
            continue
        inspect_one(name, path)


if __name__ == "__main__":
    main()
