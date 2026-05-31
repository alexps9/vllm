# verify/5 — path-counter window sensitivity: RESULTS

**Verdict: done (5/5 cells).** Songyang's "hit-window decay" hypothesis is
directionally right but quantitatively small — the anchor-survival cliff
moves only ~K=20→25 across windows 60s→600s. Recommendation: pin
`VLLM_HIMA_HPB_WINDOW_S=3600` (used by all PathA runs) so the anchor's hits
don't expire mid-run. Full sweep + reasoning in [`README.md`](README.md).
