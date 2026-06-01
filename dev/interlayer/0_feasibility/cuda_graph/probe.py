# SPDX-License-Identifier: Apache-2.0
"""cuda_graph feasibility probe — is a SCATTERED sub-block block-table safe
under CUDA-graph capture/replay?

The two-level allocator makes an attention sequence's sub-blocks non-contiguous
(arbitrary physical ids), unlike the existing virtual-block-splitting which
fans a manager block out to a *contiguous* run `N*ratio+[0..ratio)`. The worry:
a captured CUDA graph might bake in a contiguity assumption and fault / give
wrong results when the block-table holds scattered ids.

Code read says no (block_table.py:140-145: the block-table is a *persistent
input* tensor with the same address across replays; the slot kernel reads it
data-driven, no `N*ratio` assumption). This probe CONFIRMS it on the real
flash-attn kernel that virtual_split established runs at kernel_block_size=32.

Method (one sequence, prefill, causal):
  * Fill paged K/V cache so that SEVERAL disjoint sets of physical blocks each
    hold the SAME logical KV for the sequence (contiguous set + scattered sets).
  * Eager reference: flash_attn_varlen_func with the contiguous block-table.
  * Capture a CUDA graph wrapping the kernel, reading a PERSISTENT block-table
    tensor; then REPLAY while overwriting that tensor with each scattering.
  * PASS iff: no replay fault; captured ONCE (no recapture); every scattered
    replay's output == the contiguous reference (same logical KV → same math).
  * CONTROL: one replay with the block-table pointing at DIFFERENT KV must
    NOT match the reference — proving the graph re-reads the live table (else
    "match" would be trivially true and the test would be meaningless).
"""

from __future__ import annotations

import json
import sys

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def run_mode(mode: str, S_q: int, S_kv: int) -> bool:
    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    H_Q, H_KV, D = 8, 2, 128       # GQA: 8 query heads, 2 kv heads, head_dim 128
    BS = 32                        # kernel_block_size (the ksize=32 from virtual_split)
    S = S_kv                       # cached sequence length (tokens)
    nblk = (S + BS - 1) // BS      # logical blocks for the seq
    NB = 512                       # physical blocks in the paged cache (room to scatter)
    scale = D ** -0.5

    # paged KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.zeros(NB, BS, H_KV, D, device=dev, dtype=dtype)
    v_cache = torch.zeros(NB, BS, H_KV, D, device=dev, dtype=dtype)

    # the sequence's logical KV (what every block-set must hold), padded to nblk*BS
    padded = nblk * BS
    k_logical = torch.randn(padded, H_KV, D, device=dev, dtype=dtype)
    v_logical = torch.randn(padded, H_KV, D, device=dev, dtype=dtype)
    q = torch.randn(S_q, H_Q, D, device=dev, dtype=dtype)

    rng = torch.Generator(device="cpu").manual_seed(1)

    def fill(block_ids):
        """Write the logical KV into the given physical blocks."""
        for i, b in enumerate(block_ids):
            k_cache[b] = k_logical[i * BS:(i + 1) * BS]
            v_cache[b] = v_logical[i * BS:(i + 1) * BS]

    # block-id sets, all holding identical logical KV:
    contig = list(range(nblk))                              # [0..6]
    scatters = [contig]
    used = set(contig)
    for _ in range(5):                                      # 5 disjoint scatterings
        ids = []
        while len(ids) < nblk:
            c = int(torch.randint(0, NB, (1,), generator=rng).item())
            if c not in used:
                used.add(c); ids.append(c)
        scatters.append(ids)
    for s in scatters:
        fill(s)

    # a DIFFERENT-content set for the control (different KV)
    ctrl_ids = []
    while len(ctrl_ids) < nblk:
        c = int(torch.randint(0, NB, (1,), generator=rng).item())
        if c not in used:
            used.add(c); ctrl_ids.append(c)
    for i, b in enumerate(ctrl_ids):
        k_cache[b] = torch.randn(BS, H_KV, D, device=dev, dtype=dtype)
        v_cache[b] = torch.randn(BS, H_KV, D, device=dev, dtype=dtype)

    # persistent tensors for graph capture (fixed addresses; values mutated)
    cu_q = torch.tensor([0, S_q], device=dev, dtype=torch.int32)
    seqused_k = torch.tensor([S], device=dev, dtype=torch.int32)
    block_table = torch.zeros(1, nblk, device=dev, dtype=torch.int32)
    out = torch.empty(S_q, H_Q, D, device=dev, dtype=dtype)

    def run(bt_row):
        block_table[0].copy_(torch.tensor(bt_row, device=dev, dtype=torch.int32))
        flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=S_q,
            seqused_k=seqused_k, max_seqlen_k=S,
            softmax_scale=scale, causal=True, block_table=block_table,
        )

    # ---- eager reference (contiguous) ----
    run(contig)
    torch.cuda.synchronize()
    ref = out.clone()

    # eager scattered sanity (no graph): must equal ref
    run(scatters[1]); torch.cuda.synchronize()
    eager_scatter_max = (out - ref).abs().max().item()

    # ---- CUDA graph capture (ONCE) ----
    # warmup in a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            flash_attn_varlen_func(
                q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q,
                max_seqlen_q=S_q, seqused_k=seqused_k, max_seqlen_k=S,
                softmax_scale=scale, causal=True, block_table=block_table,
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    block_table[0].copy_(torch.tensor(contig, device=dev, dtype=torch.int32))
    with torch.cuda.graph(g):
        # captured op reads the PERSISTENT block_table + KV cache, writes `out`
        flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q,
            max_seqlen_q=S_q, seqused_k=seqused_k, max_seqlen_k=S,
            softmax_scale=scale, causal=True, block_table=block_table,
        )

    # ---- replay with each scattering (mutate block_table values, NO recapture) ----
    results = []
    faulted = False
    try:
        for idx, bt in enumerate(scatters):
            block_table[0].copy_(torch.tensor(bt, device=dev, dtype=torch.int32))
            g.replay()
            torch.cuda.synchronize()
            diff = (out - ref).abs().max().item()
            results.append({"replay": idx, "kind": "contig" if idx == 0 else "scatter",
                            "max_abs_diff_vs_ref": round(diff, 6),
                            "match": diff < 5e-3})
        # CONTROL: point at DIFFERENT KV; output MUST differ (proves live read)
        block_table[0].copy_(torch.tensor(ctrl_ids, device=dev, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        ctrl_diff = (out - ref).abs().max().item()
    except Exception as e:           # noqa: BLE001
        faulted = True
        print(json.dumps({"FAULT": repr(e)}))
        sys.exit(1)

    all_match = all(r["match"] for r in results)
    control_differs = ctrl_diff > 5e-2
    for r in results:
        print(json.dumps({"mode": mode, **r}))
    print(json.dumps({
        "mode": mode, "S_q": S_q, "S_kv": S_kv, "ksize": BS,
        "eager_scatter_max_diff": round(eager_scatter_max, 6),
        "control_diff_vs_ref": round(ctrl_diff, 4),
        "control_differs(expected True)": control_differs,
        "replay_fault": faulted,
        "all_scatter_replays_match_ref": all_match,
        "n_replays": len(results) + 1,
        "captured_once": True,
    }))
    return (not faulted) and all_match and control_differs


def main() -> None:
    # prefill (S_q=S_kv, causal) and decode (1 query token over a cached seq)
    modes = [("prefill", 200, 200), ("decode", 1, 200)]
    verdicts = {}
    for mode, sq, sk in modes:
        print(f"=== cuda_graph probe: {mode} (S_q={sq}, S_kv={sk}, ksize=32) ===")
        verdicts[mode] = run_mode(mode, sq, sk)
        print()
    ok = all(verdicts.values())
    print("VERDICT:", "PASS" if ok else "FAIL", json.dumps(verdicts),
          "- scattered sub-block block-table replays correctly under a captured"
          " graph (no fault, no recapture, == contiguous ref); the graph"
          " re-reads the live block-table (control differs)." if ok else "")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
