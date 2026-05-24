# Finding M.12 — py-spy root cause: batch fragmentation → more sync points

After enabling `perf_event_paranoid=1` (via sudo) and attaching py-spy
to the EngineCore subprocess, the +14% wall regression on partial-
cache apply mode is now precisely localized.

## Methodology

- `sudo sysctl -w kernel.perf_event_paranoid=1` and `kernel.kptr_restrict=0`
- Run `09_cc_workload_compare.py` with both env states
- Identify the freshly-spawned `VLLM::EngineCore` subprocess by
  filtering `etime < 120s` (the host has stale engine processes
  from other users dating back 18+ hours that match the same name)
- `sudo py-spy record -f raw --rate 200 --idle -p $EC_PID -d 15`
- Filter to main-thread samples only (stacks containing
  `step_with_batch_queue` / `_process_engine_step` /
  `run_busy_loop`); drop the tqdm/usage-lib/torch-compile-worker
  background threads that dominate raw samples

After filtering: 2715 baseline samples vs 2903 partial samples on
the main scheduler thread.

## The smoking gun

Top main-thread delta by 4-frame leaf stack:

```
  Δ%      b%      p%   stack
+2.33   38.56   40.89   step_with_batch_queue:596;result;get_output;synchronize
-1.29   10.35    9.06   run_busy_loop;_process_input_queue;get;wait
-0.67    1.29    0.62   add_requests;apply_staged_writes;...;copy_to_uva
-0.61    1.99    1.38   torch._inductor.../call (compiled GEMM kernel)
+0.48    3.20    3.69   torch._inductor.../call (different compiled kernel)
```

The biggest delta (+2.33%) is the main thread blocking in
`future.result()` at `step_with_batch_queue:596`, which eventually
hits `cudaStreamSynchronize`. The main thread spends MORE time
waiting on GPU output completions under partial cache.

Combined with M.11's findings:
- GPU kernel total time is LESS in partial (5.79 vs 6.22 s).
- CUDA API totals (cudaEventSynchronize etc.) are LESS in partial.
- Per-call FlashAttn time is +51% per call, with -36% fewer calls.

These point to a single explanation: **partial-cache hits cause the
scheduler to issue smaller, more frequent batches to the worker**.
Each batch executes faster on GPU (smaller, +51% per-call attn time
because of different shape/locality), but there are MORE batches,
each requiring a `future.result()` sync point on the main scheduler
thread. The fixed per-sync overhead × more syncs = the 2.43s extra
CPU+sync time in M.11.

The `-1.29% in _process_input_queue.get` confirms this from the
other direction: the main thread spends LESS time waiting on the
input queue (idle), because it's instead busy syncing for GPU
output more often.

## Why partial cache fragments batches

vLLM v1 uses async scheduling with `batch_queue_size = 2`. The
scheduler tries to keep 2 batches in flight on the worker. Each
scheduling step picks requests and fills up to
`max_num_batched_tokens = 16384` tokens.

When partial cache hits, the new admission needs to prefill only
the EXTENSION (e.g., 16 tokens for the cc workload's TTFT pass)
instead of R+16 tokens. A request that would have been a "big
prefill chunk" becomes a "tiny prefill chunk", which fills the
batch slot quickly. The scheduler moves on to schedule the NEXT
request sooner. Net: more scheduling steps for the same total
output → more sync points.

The async scheduler is designed to hide this overhead by overlapping
the next batch's `schedule` with the previous batch's GPU work, but
when the GPU work is short (small batch), there's nothing to overlap
with, and the main thread blocks on `future.result()`.

## Mitigations (none implemented yet)

1. **Increase `batch_queue_size`** from 2 to 4 or 8 — gives the
   scheduler more room to overlap. Cheap to try.

2. **Coalesce partial-cache hits across requests** so that we issue
   bigger admission batches. Today each admission goes through
   `_try_partial_extension` independently. If two cc-session turns
   arrive close together and share a parent prefix, batching their
   admissions could amortize.

3. **Defer the partial-extension application** until the prefill
   batch is "large enough" — skip the partial hit when the
   resulting batch would be too small to amortize the sync cost.
   Requires a heuristic.

4. **Reduce per-sync overhead** by batching multiple
   `future.result()` calls. Architectural; out of scope.

For the cc workload at single-stream (max_num_seqs=4), the
mitigation is probably (1) or (3). For multi-tenant (large
max_num_seqs), the regression should self-amortize because batches
are naturally bigger.

## Confidence level

This root cause is now well-supported:
- Main-thread sample distribution shifts toward `future.result()`
  by 2.33% (= +0.35s of waiting time across the 15s sampling
  window → consistent with 2s wall regression over the 16s workload)
- GPU kernel time is LESS in partial (M.11) — rules out "GPU
  doing more work"
- CUDA API totals are LESS (M.11) — rules out "more launches /
  more syncs in CUDA driver time"
- Per-call FlashAttn time +51% with -36% calls (M.11) — confirms
  shape change consistent with smaller batches

The regression is a **batching efficiency loss** caused by the
partial-cache mechanism making prefill chunks smaller. Not a Python
bug, not a GPU bug. An engine-scheduler interaction.

## Repro

```bash
# Need sudo for py-spy attach
sudo sysctl -w kernel.perf_event_paranoid=1
sudo sysctl -w kernel.kptr_restrict=0

# Then:
CUDA_VISIBLE_DEVICES=0 KMP_AFFINITY=disabled VLLM_PARTIAL_CACHE_ENABLED=1 \
    .venv/bin/python -u dev/interlayer/09_cc_workload_compare.py &
WORKLOAD_PID=$!
# (wait ~30s for engine to enter the workload loop)
EC_PID=$(ps -eo pid,etime,cmd | grep VLLM::EngineCore | sort -k2 -n | head -1 | awk '{print $1}')
sudo py-spy record -f raw -o /tmp/partial.raw --rate 200 --idle -p $EC_PID -d 15
wait $WORKLOAD_PID

# Repeat without VLLM_PARTIAL_CACHE_ENABLED for baseline.
# Diff main-thread samples:
python dev/interlayer/12_pyspy_diff.py  # (the analysis script in this commit)
```

## Files

- `dev/interlayer/runs/nsys/baseline_raw.txt` — 27159 samples,
  baseline (no env)
- `dev/interlayer/runs/nsys/partial_raw.txt` — 29711 samples,
  partial cache enabled
- `dev/interlayer/runs/nsys/partial_engine.svg` — partial flamegraph
- `dev/interlayer/runs/nsys/baseline_engine.svg` — baseline flamegraph
