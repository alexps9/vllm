# dev/interlayer (vLLM)

vLLM-side **interlayer** work: the cross-pool / page-size **bubble** in
hybrid models (attention KV + mamba recurrent state). Counterpart to the
sglang repo's `dev/interlayer/`, but vLLM's bubble is a *page-size*
(internal-fragmentation) bubble, not sglang's fixed-split bubble — see
[`design.md`](design.md).

Layout mirrors sglang's (top-level `design.md` + numbered phase subdirs):

| phase | what |
|---|---|
| [`design.md`](design.md) | problem statement + why vLLM ≠ sglang + solution direction |
| [`0_page_bubble/`](0_page_bubble/) | **prove the bubble exists** — block_size inflation (→1056) + realistic-trace waste (42.6% on 106 CC sessions). README + scripts + RESULTS. |

History: phase 0's measurement was first done in commit `438ad0397`
("empirical study of vLLM hybrid-model BlockPool inflate"), then deleted
with the pcache tree (`7d974f6f6`) when pcache — the *wrong* fix — was
removed. The bubble proof is sound and is restored here to motivate the
*right* fix.
