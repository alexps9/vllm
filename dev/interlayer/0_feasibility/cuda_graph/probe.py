# SPDX-License-Identifier: Apache-2.0
"""cuda_graph feasibility probe — is a SCATTERED sub-block block-table safe
under CUDA-graph capture/replay?

The two-level allocator makes an attention sequence's sub-blocks non-contiguous
(arbitrary physical ids), unlike the existing virtual-block-splitting which
fans a manager block out to a *contiguous* run `N*ratio+[0..ratio)`. The worry:
a captured CUDA graph might bake in a contiguity assumption and fault / give
wrong results when the block-table holds scattered ids — which would force
eager mode, a design-level showstopper.

Code read says no (block_table.py:140-145: the block-table is a *persistent
input* tensor with the same address across replays; the slot kernel reads it
data-driven, no `N*ratio` assumption). This probe CONFIRMS it on the real
flash-attn kernel — using the **production CUDA-graph config** that the vLLM
backend passes (flash_attn.py:796-818): FA version from `get_flash_attn_version`,
`num_splits = flash_attn_max_num_splits_for_cuda_graph` (=32), and the FA3
`scheduler_metadata` AOT schedule. (v1 of this probe omitted those; an audit
(a49bf77f) flagged it and verified the conclusion is unchanged with them — now
folded in so the probe exercises the real path itself.)

Method, per case (a batch of sequences; prefill and/or decode):
  * Fill the paged K/V cache so SEVERAL disjoint physical block-sets each hold
    the SAME logical KV for every sequence (1 contiguous + R scatterings).
  * Eager reference = flash_attn_varlen_func with the contiguous block-table.
  * Capture a CUDA graph ONCE wrapping the kernel, reading a PERSISTENT
    block-table tensor; REPLAY while overwriting it with each scattering.
  * PASS iff: no replay fault; captured exactly once (counter); every scattered
    replay == the contiguous reference (same logical KV → same math).
  * CONTROL: one replay with one row pointing at DIFFERENT KV must NOT match —
    proving the graph re-reads the live table (else "match" is trivially true).

Cases: prefill (1×200), decode (1×1 over 200), mixed batch (prefill+2 decode,
per-row scatter), long-context decode (1 over 4096 → split-KV genuinely
partitions the sequence across many scattered blocks).
"""

from __future__ import annotations

import json
import sys

import torch

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
    get_scheduler_metadata,
)

DEV = "cuda"
DT = torch.bfloat16
H_Q, H_KV, D = 8, 2, 128
BS = 32                         # kernel_block_size (the ksize=32 from virtual_split)
NB = 2048                       # physical blocks in the paged cache (room to scatter)
NUM_SPLITS = 32                 # flash_attn_max_num_splits_for_cuda_graph (prod default)
SCALE = D ** -0.5


def run_case(name: str, seqs: list[tuple[int, int]], n_scatter: int = 5) -> bool:
    """seqs = list of (q_len, kv_len). Returns verdict bool."""
    torch.manual_seed(0)
    g_rng = torch.Generator(device="cpu").manual_seed(1)
    fa_version = get_flash_attn_version()
    n = len(seqs)
    q_lens = [s[0] for s in seqs]
    kv_lens = [s[1] for s in seqs]
    nblks = [(kv + BS - 1) // BS for kv in kv_lens]
    max_blocks = max(nblks)

    k_cache = torch.zeros(NB, BS, H_KV, D, device=DEV, dtype=DT)
    v_cache = torch.zeros(NB, BS, H_KV, D, device=DEV, dtype=DT)

    # per-seq logical KV (every block-set for that seq holds this)
    k_log = [torch.randn(nb * BS, H_KV, D, device=DEV, dtype=DT) for nb in nblks]
    v_log = [torch.randn(nb * BS, H_KV, D, device=DEV, dtype=DT) for nb in nblks]
    q = torch.randn(sum(q_lens), H_Q, D, device=DEV, dtype=DT)

    used: set[int] = set()

    def fresh_blocks(cnt: int) -> list[int]:
        out = []
        while len(out) < cnt:
            c = int(torch.randint(0, NB, (1,), generator=g_rng).item())
            if c not in used:
                used.add(c); out.append(c)
        return out

    def fill(seq_i: int, ids: list[int]) -> None:
        for j, b in enumerate(ids):
            k_cache[b] = k_log[seq_i][j * BS:(j + 1) * BS]
            v_cache[b] = v_log[seq_i][j * BS:(j + 1) * BS]

    # variant 0 = contiguous-ish (first fresh run), variants 1..R = scattered.
    # each (seq, variant) gets a DISJOINT physical block-set with identical KV.
    variants = []
    for v in range(1 + n_scatter):
        rows = []
        for i in range(n):
            ids = (list(range(i * max_blocks, i * max_blocks + nblks[i]))
                   if v == 0 else fresh_blocks(nblks[i]))
            if v == 0:
                used.update(ids)
            fill(i, ids)
            rows.append(ids)
        variants.append(rows)

    # control: replace seq 0's row with blocks holding DIFFERENT content
    ctrl_ids = fresh_blocks(nblks[0])
    for j, b in enumerate(ctrl_ids):
        k_cache[b] = torch.randn(BS, H_KV, D, device=DEV, dtype=DT)
        v_cache[b] = torch.randn(BS, H_KV, D, device=DEV, dtype=DT)

    # persistent tensors (fixed addresses; values mutated between replays)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
                        device=DEV, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, device=DEV, dtype=torch.int32)
    block_table = torch.zeros(n, max_blocks, device=DEV, dtype=torch.int32)
    out = torch.empty(sum(q_lens), H_Q, D, device=DEV, dtype=DT)
    max_q, max_k = max(q_lens), max(kv_lens)

    # FA3 AOT schedule (prod passes this under CUDA graphs) — computed from
    # seqlens/heads, NOT from block ids (so a scatter never invalidates it).
    sched = get_scheduler_metadata(
        batch_size=n, max_seqlen_q=max_q, max_seqlen_k=max_k,
        num_heads_q=H_Q, num_heads_kv=H_KV, headdim=D,
        cache_seqlens=seqused_k, qkv_dtype=DT, cu_seqlens_q=cu_q,
        page_size=BS, causal=True, num_splits=NUM_SPLITS,
    ) if fa_version == 3 else None

    def set_bt(rows):
        for i, ids in enumerate(rows):
            block_table[i, :len(ids)].copy_(
                torch.tensor(ids, device=DEV, dtype=torch.int32))

    def call():
        flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=max_q,
            seqused_k=seqused_k, max_seqlen_k=max_k,
            softmax_scale=SCALE, causal=True, block_table=block_table,
            scheduler_metadata=sched, num_splits=NUM_SPLITS,
            fa_version=fa_version,
        )

    set_bt(variants[0]); call(); torch.cuda.synchronize()
    ref = out.clone()

    # warmup in a side stream (required before capture)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(st)

    n_captures = 0
    g = torch.cuda.CUDAGraph()
    set_bt(variants[0])
    with torch.cuda.graph(g):
        n_captures += 1
        call()

    results, faulted = [], False
    try:
        for idx, rows in enumerate(variants):
            set_bt(rows); g.replay(); torch.cuda.synchronize()
            diff = (out - ref).abs().max().item()
            results.append({"variant": idx, "kind": "contig" if idx == 0 else "scatter",
                            "max_abs_diff_vs_ref": round(diff, 6), "match": diff < 5e-3})
        ctrl_rows = [list(ctrl_ids)] + variants[0][1:]   # seq0 -> different KV
        set_bt(ctrl_rows); g.replay(); torch.cuda.synchronize()
        ctrl_diff = (out - ref).abs().max().item()
    except Exception as e:                               # noqa: BLE001
        print(json.dumps({"case": name, "FAULT": repr(e)})); return False

    all_match = all(r["match"] for r in results)
    control_differs = ctrl_diff > 5e-2
    for r in results:
        print(json.dumps({"case": name, **r}))
    print(json.dumps({
        "case": name, "seqs": seqs, "fa_version": fa_version, "num_splits": NUM_SPLITS,
        "ksize": BS, "control_diff_vs_ref": round(ctrl_diff, 4),
        "control_differs(expect True)": control_differs,
        "replay_fault": faulted, "all_scatter_match_ref": all_match,
        "n_captures(expect 1)": n_captures, "n_replays": len(results) + 1,
    }))
    return (not faulted) and all_match and control_differs and n_captures == 1


def main() -> None:
    cases = [
        ("prefill", [(200, 200)]),
        ("decode", [(1, 200)]),
        ("batch_mixed_prefill+2decode", [(200, 200), (1, 96), (1, 160)]),
        ("long_decode_splitKV", [(1, 4096)]),       # 128 blocks → split-KV partitions
    ]
    verdicts = {}
    for name, seqs in cases:
        print(f"=== cuda_graph probe: {name} {seqs} (FA{get_flash_attn_version()}, "
              f"num_splits={NUM_SPLITS}, ksize={BS}) ===")
        verdicts[name] = run_case(name, seqs)
        print()
    ok = all(verdicts.values())
    print("VERDICT:", "PASS" if ok else "FAIL", json.dumps(verdicts))
    print("- scattered/per-row sub-block block-tables replay correctly under a"
          " captured graph (prod FA3 + num_splits=32 + scheduler_metadata): 0"
          " faults, captured once, == contiguous ref; control differs (live read)."
          if ok else "- see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
