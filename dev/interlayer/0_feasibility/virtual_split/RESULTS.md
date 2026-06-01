# virtual_split — RESULTS

> ℹ️ **interlayer CLOSED — not pursued.** This check passed (feasibility was never the blocker); the effort was dropped for *value* reasons. See [`../../POSTMORTEM.md`](../../POSTMORTEM.md).

**Verdict: the kernel runs correctly at sub-page granularity with no kernel
change (premise holds). It is NOT bit-identical across block sizes — and that
bar was misconceived: block size inherently changes floating-point reduction
order. The fix will be numerically-equivalent-but-not-bit-identical vs the
1056 baseline, the same class of variation stock vLLM already has across
`block_size`.**

## What was established

1. **kernel_block_size = 32, splitting active** (live, Qwen3.5-35B-A3B align):
   `select_common_block_size(1056, [FlashAttn]) = 32`. The kernel runs at 32,
   allocation at 1056 → the bubble is allocator-only. For this fp32-SSM hybrid
   flash-attn only supports `[16,32,64]` (NaN branch), and 1056's only viable
   factor is 32 — the kernel **cannot** even use the 1056 page. (`probe.py` →
   `runs/probe_result.json`.)

2. **Granularity invariance (the real test): kernel computes valid attention
   at both 16 and 32, numerically equivalent.** Forced `kernel_block_size` to
   16 vs 32 (both legal factors of 1056), greedy, identical prompts
   (`probe_invariance.py` → `runs/inv_compare.json`):

   | prompt | tokens identical (16 vs 32) | first-tok logprob \|Δ\| | first divergence |
   |---|---|---|---|
   | 0 | ✅ | 4.3e-4 | — |
   | 1 | ✅ | 1.7e-3 | — |
   | 2 | ❌ | 3.0e-3 | token 58 / 128 |
   | 3 | ✅ | 9.1e-4 | — |

   Logprob deltas are ~1e-3 (fp-rounding level); prompt 2's divergence is at
   token 58 after 57 identical tokens — the signature of tiny fp differences
   compounding through greedy argmax until one near-tied step flips. **Not a
   correctness bug** (a wrong-KV read would give garbage/early divergence, not
   "57 identical then one flip" with ~1e-3 deltas).

## Corrected pass criterion

The original bar — "outputs bit-identical (atol=rtol=0)" — is **unachievable
by construction**: different block sizes accumulate attention in different
orders, so logits differ at the fp level regardless of correctness. Stock
vLLM already changes numerics when `block_size` changes; it does not promise
bit-identical output across block_size configs.

The correct, met criterion: **the kernel computes valid attention at the
target sub-page granularity (16/32) with no kernel change, and output is
numerically equivalent (logit Δ ~1e-3)** — i.e. the same class of variation
vLLM already exhibits across `block_size`.

## Design implication (acceptance decision)

The bubble fix changes attention's allocation granularity, so model output
will be **numerically equivalent but not bit-identical** to the 1056 baseline
(tiny fp differences; occasional token flips on long greedy generations).
This is acceptable **iff** we accept block-size-level numerical variation —
which vLLM already has. This is a stated, accepted property of the fix, not a
defect, but it is a product decision worth recording.

## Scope (unchanged)

Still tests *granularity*-invariance (contiguous fan-out at 16 vs 32). The
*arbitrary-scatter / multi-tenant-page* placement remains scoped to
`sub_block_allocator/` (phase 2) + end-to-end correctness in `the_win/`.
Non-eager (CUDA-graph) correctness is `cuda_graph/` (phase 6).
