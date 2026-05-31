# verify/8 — L1 intact after L2 removal (2026-05-31)

Smoke after deleting all L2 code: l1_only on the PathA pipeline
(Qwen3.5-35B-A3B, util=0.9, TP=2, phase-f-scale=10).

PhaseH hit% = **88.87**, anchor cached = **126720** — **bit-identical** to
every prior l1_only run (verify/1, verify/4, verify/7). The L1 win is
preserved exactly; removing L2 broke nothing.
