# sub_block_allocator — RESULTS

**PASS (memory-safety).** A prototype two-level allocator (physical page ↔ K
sub-blocks; per-sub-block ownership; mamba↔attention↔free page-flip; packing
bias) survives **3 seeds × 1e6 randomized ops + a genuine adversarial
max-scatter phase with ZERO invariant violations.** `fuzz_allocator.py` →
`runs/fuzz.out`.

## Invariants checked (every op O(1) + full sweep periodically)

- no sub-block aliased by two live owners; alloc never returns an occupied slot
- no use-after-free / wrong-owner free (attention slots and mamba pages)
- page mode ⇔ occupancy (FREE empty; ATTN 1..K held; MAMBA whole-page, no attn)
- a page is in `free_pages` **iff** fully empty (the mamba-usable flip)
- conservation: handle count = occupied sub-blocks; FREE+ATTN+MAMBA = n_pages;
  free/mamba index sizes match page modes

Result across all seeds: **`invariant_violations: 0`** (violations raise, so a
clean run is a proof for that trace). The random phase exercised heavy
saturation — ~246k mamba-starve and ~344k attention-OOM events — and every
one stayed memory-safe.

## Adversarial max-scatter (the genuine test)

Fresh allocator; fill every page with K size-1 reqs, then free all-but-one per
page → **every page has exactly 1/K slot used, none fully free**:

| check | result | meaning |
|---|---|---|
| `adv_starve_when_full` | True | mamba can't get a page when pool full |
| `adv_starve_max_scatter` | **True** | 1/K used everywhere → mamba **starves** despite (K−1)·n_pages free sub-blocks |
| `adv_mamba_recovered_after_freeing_one_page` | True | free one page's last slot → it flips FREE → mamba gets it |

Invariants held through all of it.

## What this does and does NOT establish

- ✅ The two-level allocator **structure is memory-safe** and the page-flip
  works — the data-structure foundation of the fix is sound.
- ⚠️ It **reproduces the P1 risk**: genuine max-scatter starves mamba even
  with abundant free sub-block space. This is **expected** — phase 2 owns
  *safety*, not *policy*. Whether the packing bias + cost-model reclaim keep
  this starvation bounded under realistic (not pathological) load is exactly
  `cost_reclaim/` (phase 3, the make-or-break). The prototype here is the
  thing phase 3 will drive.

Note: this is a standalone prototype, not the vLLM `BlockPool` integration
(that's implementation, post-gate). It proves the design's allocator *can* be
safe; the real integration must preserve these invariants under vLLM's
append-only block-id constraint.
