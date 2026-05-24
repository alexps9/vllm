# Finding M.11 — nsys profiling of the M.10 residual regression

After M.10 fixed the in-place mutation bug, the apply-mode partial
cache still showed +13.8% wall regression on cc workload (16.47s vs
14.47s baseline). M.11 uses nsys to localize where the 2-second
regression actually goes.

## What nsys captured

Two profiling runs of `09_cc_workload_compare.py`:
- `baseline.nsys-rep` (env unset)
- `partial.nsys-rep` (env set)
- `baseline_cpu.nsys-rep` / `partial_cpu.nsys-rep` (added CPU+python
  sampling — but no samples captured, likely needs root for
  `perf_event_paranoid`)

Trace settings: `--trace=cuda,nvtx --trace-fork-before-exec=true`
(critical — without fork-trace, vLLM v1's EngineCore subprocess
isn't captured).

## Headline finding

**The regression is NOT in GPU kernel time.** GPU kernel time is
*lower* in partial mode (-6.8%). The 2-second wall regression is
entirely in **CPU and GPU-idle time between kernels**.

```
                   baseline     partial      delta
wall (issue API)   14.47 s      16.47 s      +2.00 s  (+13.8%)
GPU kernel time     6.22 s       5.79 s      -0.42 s  (-6.8%)
CPU + idle (gap)    8.25 s      10.68 s      +2.43 s  (+29.5%)
```

The GPU is mostly idle in both runs:
- baseline: 81.83% idle (18.17% busy)
- partial:  80.81% idle (19.19% busy)

The workload is **CPU-bound on the engine subprocess**, and partial-
cache application makes that CPU-bound portion ~30% slower.

## Per-kernel breakdown reveals shape shift

The dominant decode-time kernel is `FlashAttnFwdSm90` at grid 132x1x1:

| variant | calls | avg per-call | total |
|---|---|---|---|
| baseline 132x1x1 | 5127 | 207 µs | 1062 ms |
| partial  132x1x1 | 3276 | **313 µs (+51%)** | 1026 ms |
| baseline 132x1x1 (small) | 1224 | 88 µs | 108 ms |
| partial  132x1x1 (small) | 1728 | 77 µs | 133 ms |

The same grid shape runs ~51% LONGER per call in partial mode. But
fewer total calls means total time is similar. So per-call shape
parameters (q_seqlen, kv_seqlen, etc.) must be different even though
grid is the same — likely longer kv_seqlen per query because cached-
hit context resides in non-contiguous block_table entries that
hurt memory locality during attention's KV read.

## Per-launch wall amortization

```
                   kernels    wall      wall/launch
baseline           120210     14.47 s   120 µs
partial             95640     16.47 s   172 µs  (+52 µs per launch)
```

Per-kernel wall time increased by 52µs. Across the partial run's
95640 launches that's ~5s of extra wall — more than the 2s
regression. The math doesn't perfectly close because partial has
24570 fewer launches AND each one is slower. The net is the
observed +2s.

## What's actually slower on the CPU?

CPU sampling didn't capture (permission issue), so we can only
inspect CUDA API call totals (which represent CPU thread time spent
in the CUDA driver):

```
                                  baseline    partial     delta
cudaEventSynchronize  total       10031 ms    8503 ms    -1528 ms
cudaMemcpyAsync       total        4928 ms    3256 ms    -1672 ms
cudaLaunchKernel      total         697 ms     628 ms      -69 ms
cuLaunchKernelEx      total         233 ms     189 ms      -44 ms
```

Every CUDA API total is LOWER in partial mode (because fewer kernels
launched). So the regression isn't in CUDA API CPU time either.

The +2.43s must be in **Python scheduler / engine code** that is NOT
covered by CUDA APIs. Specifically: between two consecutive CUDA
launches, the engine subprocess runs Python code (scheduler step,
block allocation, batch building, etc.). nsys doesn't time this
Python code without explicit NVTX ranges or CPU sampling.

## What we can be confident about

1. The GPU itself isn't doing more work in partial mode (less, in
   fact).
2. The regression isn't in launch overhead (CUDA API time is lower).
3. The regression must be in Python-side engine code between kernel
   launches — specifically, code that runs MORE OFTEN or MORE
   EXPENSIVELY when partial cache hits occur.
4. Our perf_counter timer in `cache_partial_block` showed only ~10 ms
   of work across the whole workload. So the cost isn't in our
   M.4/M.5 partial-cache scaffolding code itself.

## What we can hypothesize but haven't confirmed

The CPU-side cost is probably in vLLM's `gpu_model_runner` /
scheduler / input_batch code paths that handle a request with a
cached partial block in its block_table:

- **slot_mapping construction**: when block_table[K] is a "borrowed"
  partial block, slot mappings for new decode tokens land at offsets
  [R, R+1, ...] of that block instead of [0, 1, ...]. The Python-
  side code that builds the slot_mapping tensor may have a different
  cost (e.g., extra branches, less SIMD-friendly access patterns
  during numpy operations).
- **block_table tensor builds**: input_batch's block_table tensor is
  rebuilt or appended-to per step. When the K-th block is "old"
  (allocated long ago), the block_id integer value is different,
  affecting some hash/sort/scan within the metadata builder.
- **attention metadata's `kv_seqlen` array**: with partial-cache
  hits the per-request kv_seqlen is longer (K*block_size + R) than
  baseline (K*block_size for ~91.6% of turns, computed full for the
  rest). The Triton kernel for slot_mapping might have shape-
  dependent perf cliffs.

These are all engine-side / Python-side effects that would need
either real CPU sampling (run as root or with perf_event_paranoid=1)
or explicit NVTX instrumentation to localize precisely.

## Recommendation

The remaining ~14% wall-clock regression on cc workload comes from
**engine-side CPU work that increases when applying partial-cache
hits, not from GPU kernels**. The right next step is:

1. Get CPU sampling working (set `kernel.perf_event_paranoid=1` or
   run nsys as root).
2. Wrap key engine functions (`scheduler.schedule`, `model_runner.
   _prepare_inputs`, `_build_attn_group_metadata`) with NVTX ranges.
3. Re-profile and identify the specific Python function whose
   per-call cost goes up in apply mode.

For the immediate user impact: the regression is real but the
TTFT win (-16%) and bubble elimination (-88%) still make partial
cache a net win for TTFT-sensitive interactive workloads. Bulk-
generation workloads should leave the env var unset (default).

The +51% per-call FlashAttn time at grid 132x1x1 is suggestive —
something about the cached-block memory layout is changing the
kernel's actual work even though grid is the same. Worth a deeper
dive with `nsys-ui` or `ncu` (Nsight Compute, kernel-level) in a
future investigation.
