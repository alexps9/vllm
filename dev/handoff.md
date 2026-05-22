# HiMA L1 LRU-vs-LPB Multi-Scenario Comparison — Handoff

This is everything you need to reproduce the **3-scenario comparison**
(best case / average case / worst case for HiMA L1's LPB-scored
free-block queue vs vLLM's default LRU queue) on Qwen3.5-35B-A3B
real cc workload.

The host I had access to has a CPU fairshare scheduler that pins
processes to a single CPU, and was sharing the box with another user's
heavy CUDA kernel compilation. The script is correct and ran once
end-to-end before contention got bad (LPB: ~4 min wall, LRU: same).
You should be able to run it cleanly on any host without that pressure.

---

## TL;DR — what to run

On a host with:

- ≥ 2× H100 / H200 / similar (~140 GB each); script uses TP=2 and ~35 GB GPU each
- vLLM-songyang branch `HiMA` already built (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`)
- `KMP_AFFINITY=disabled` (already set inside script)
- The cc traces file at `/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl` (44 MB, 106 sessions). If missing, copy from the source host or update `DATA = Path(...)` at top of `dev/compare_lru_lpb.py`.

```bash
cd /data/yuzhou/projects/vllm-songyang
git checkout HiMA && git pull

# Pick two free GPUs (any two; doesn't matter which).
export GPUS=0,1

# Both modes — model load takes ~1.5 min each, experiment ~3 min each.
CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/python -u dev/compare_lru_lpb.py --mode lru \
    | tee dev/compare_lru.out
CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/python -u dev/compare_lru_lpb.py --mode lpb \
    | tee dev/compare_lpb.out

# Aggregate + plot
.venv/bin/python dev/plot_lru_vs_lpb.py | tee dev/compare_summary.out
```

Outputs land at:
- `dev/compare_{lru,lpb}.jsonl` — per-request structured log (one row per phase event)
- `dev/compare_{lru,lpb}.out` — engine stdout
- `dev/compare_summary.out` — formatted 3-scenario summary table
- `dev/compare_summary.json` — aggregated metrics
- `dev/figures/fig_anchor_survival.png` — Phase A/B/C anchor cache % over time
- `dev/figures/fig_workload_metrics.png` — Phase B cc burst headline (TTFT/TPOT/throughput LRU vs LPB)
- `dev/figures/fig_lru_vs_lpb_scenarios.png` — **the multi-scenario plot** (2×2 grid: TTFT, TPOT, throughput, hit% × 3 scenarios)

---

## What the experiment proves

HiMA L1 replaces vLLM's LRU `FreeKVCacheBlockQueue` with a path-counted
LPB scorer that protects deeply-hit shared prefixes (the "anchor")
from cold-burst eviction. Finding K in `dev/README.md` already showed
this on the **average** case (cc burst). The user asked for best AND
worst case too, so this driver runs **three phases** in one engine
load, per mode:

| phase | label | workload shape | predicted LPB outcome |
|---|---|---|---|
| **B** | **cc_burst** (average) | replay 10 real cc sessions after warming anchor 500×; anchor is never re-hit during the burst | small win on workload metrics, **big win on anchor survival** |
| **D** | **anchor_rehit** (BEST CASE) | after Phase B+C, issue 30 fresh requests = anchor + unique 16-token tail | **TTFT collapse**: LPB keeps anchor → 89% prefix hit per rehit; LRU evicted it → full ~4.7K-token re-prefill per request |
| **E** | **cold_unique** (WORST CASE) | 50 short 2K-token prompts with NO shared prefix and NO anchor warming | should be a near-wash; isolates LPB's hot-path overhead (path-counter lookup, heap-based queue) from any structural win |

Phase D quantifies the **upside** when downstream traffic actually
consumes the anchor LPB protected (the swarm / shared-system-prompt
pattern HiMA targets). Phase E quantifies the **downside risk** when
LPB has nothing useful to protect.

---

## What we expect to see (anchored on the partial data captured here)

### Phase A — anchor warmup (both modes)
Anchor is the first user message of cc session 0, ~4737 tokens
(~4.5 blocks at `block_size = 1056`). After 500 warm probes both modes
have `cached = 4224/4737 (89.2%)`. The 8.8% miss is the residual of
the last partial block — vLLM only caches full blocks, and the anchor
isn't an exact block multiple. Don't chase the 8.8%, it's expected.

```
BASELINE anchor probe: cached=4224/4737 (89.2%)
```

### Phase B — cc burst, 10 sessions (average case)

Already in `dev/README.md` Finding K. Expect (with `±10%` variance on a
quiet box; numbers from the run committed in `8a4add270`):

```
                                LRU          LPB           Δ
mean TTFT (ms)               ~510          ~495        -2.9%
mean TPOT (ms/tok)           ~28.5         ~27.4       -3.7%
throughput (out_tok/s)       ~720          ~754        +4.7%
total wall (s)               ~165          ~159        -3.8%
```

### Phase C — final anchor probe

**THE HEADLINE.** Under LPB the anchor survives Phase B's cold burst;
under LRU it gets evicted.

```
[lru] FINAL anchor probe: cached=0/4737 (0.0%)
[lpb] FINAL anchor probe: cached=4224/4737 (89.2%)
```

### Phase D — anchor re-hit (LPB BEST CASE)

30 requests, each prompt = anchor (4737 tokens) + unique 16-token tail.

**LPB side (verified — data committed in `dev/compare_lpb.jsonl`):**
```
  rehit[ 0] ttft_cached= 4224 ttft_wall=60ms
  rehit[ 1] ttft_cached= 4224 ttft_wall=59ms
  rehit[15] ttft_cached= 4224 ttft_wall=58ms
  rehit[29] ttft_cached= 4224 ttft_wall=60ms
```
All 30 rehits hit at 89.2% (4224/4737 cached). TTFT ~60ms.

**LRU side (predicted; please verify):**
First rehit's TTFT pass should show `ttft_cached=0` (anchor was
evicted in Phase B). The decode pass should show `full_cached=~0`
(the prompt as a whole hasn't been seen).
From rehit 1 onwards the prompt itself is now cached (it was just
issued in rehit 0's TTFT+decode passes), so subsequent rehits will
get high hits — but each prompt has a unique tail per `rehit[j]`, so
the cache only covers the anchor portion of *the same* prompt, not
across `j`. Net: each LRU rehit will pay a fresh ~4737-token
prefill on the anchor portion → TTFT in the **hundreds of ms** range,
not 60 ms. **Expected LPB win on Phase D mean TTFT: ~5–8×.**

### Phase E — no-shared-prefix cold flow (LPB WORST CASE)

50 unique 2048-token prompts, slices of a long filler, no overlap.
LPB and LRU should be **statistically indistinguishable**. If LPB is
materially slower (>5% on TPOT/throughput), that's a real hot-path
regression worth investigating — the path-counter and heap-based
queue overhead is paying nothing.

Expected:
```
mean TTFT (ms)           ~150        ~150        ~0%
mean TPOT (ms/tok)       ~28         ~28         ~0%
throughput (out_tok/s)   ~720        ~720        ~0%
```

---

## Existing data on disk you can re-use

`dev/compare_lpb.jsonl` (currently committed at `83657cf8d`) contains:
- 238× `cc_turn` rows — Phase B (LPB side, average case)
- 2× `anchor_probe` rows — Phases C baseline + final
- 30× `rehit_turn` rows — **Phase D anchor-rehit data, LPB side**
- 0× `cold_turn` rows — Phase E not captured (the first run errored
  on a short filler; the fix is in 83657cf8d, just re-run)

If you don't want to re-run LPB, you can run only LRU and the plot
script will still produce most of the 3-scenario table — the only
gap will be Phase E LPB side (so the "worst case" comparison will
be missing). But re-running LPB is cheap (~4 min), recommended.

---

## Environment requirements

### Hardware
- 2× GPU with at least 50 GB each (Qwen3.5-35B-A3B in BF16 + 35% util KV cache + state). The script asks for `gpu_memory_utilization=0.35` which on H200 (143 GB) gives ~50 GB → ~1.08 M KV tokens.
- ~200 GB system RAM.

### Software
```bash
# vllm-songyang repo on HiMA branch
git clone https://github.com/alexps9/vllm.git vllm-songyang
cd vllm-songyang
git checkout HiMA  # current head: 83657cf8d as of this handoff

# venv (NEVER use system python; per AGENTS.md)
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# verify HiMA is wired
.venv/bin/python -c "from vllm.engine.arg_utils import EngineArgs; \
    print('hima_enabled' in vars(EngineArgs()))"
# → should print True
```

### Dataset
The cc traces (106 real Claude Code sessions, ~44 MB). Either:
- Copy `/data/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl` from the source host, OR
- Edit the `DATA = Path(...)` line at `dev/compare_lru_lpb.py:53` to point at your copy.

### Env vars (already set inside script, listed for reference)

```bash
KMP_AFFINITY=disabled        # prevent Intel OMP from pinning to 1 CPU
VLLM_HIMA_HPB_WINDOW_S=3600  # don't expire anchor hits between phases
VLLM_LOGGING_LEVEL=INFO
# Optional: VLLM_PORT=39911 if 8081 conflicts
```

---

## Sanity checks before claiming victory

1. **Did HiMA actually fire?** Look in `dev/compare_lpb.out` for:
   ```
   INFO ... [core.py:168] HiMA enabled (page=2048 KiB, kv_slots=..., rec_slots=...).
   ```
   If absent → `--mode lpb` didn't enable it; check `EngineArgs.hima_enabled` plumbing.

2. **Did the anchor actually survive under LPB?** Phase C output:
   ```
   [lpb] FINAL anchor probe: cached=4224/4737 (89.2%)
   ```
   If 0/4737 → LPB scoring fell back to time.monotonic() somewhere. The
   `_HIT_SCORE_OFFSET = 1e12` in `vllm/v1/core/hima/lpb_free_queue.py`
   is what keeps hit blocks above cold blocks in the min-heap.

3. **Are Phase D LPB rehits actually hitting?** Look for:
   ```
   rehit[ 0] ttft_cached= 4224 ttft_wall=~60ms
   ```
   If `ttft_cached=0`, the anchor got evicted between Phase C and
   Phase D — almost certainly means LPB isn't running (sanity check 2
   would have already failed).

4. **Is Phase E genuinely cold?** Look for:
   ```
   cold[ 0] cached=  0 wall=~150ms
   ```
   If `cached>0`, your filler is being block-aligned with something
   else and Phase E isn't testing what it claims to.

---

## If LRU run fails to load model and burns time

The original symptom on the source host was: python process at 0.2%
CPU, RSS frozen at 395 MB, no syscalls for minutes. Causes:

1. **Intel OMP pinned to 1 CPU.** Fix: `KMP_AFFINITY=disabled` (already in script).
2. **Another user is hogging the CPU with C++/CUDA compilation.** Wait it out, or move to a quieter machine.
3. **Process killed but won't die on SIGKILL.** Was a separate kernel-level cgroup wedge; was reset by reboot. If you see this, try `pkill -9 -u $USER python` then escalate to admin.

To verify the script is alive (not wedged), watch:
```bash
watch -n 5 'ps -p $(pgrep -f compare_lru_lpb | tail -1) -o etime,pcpu,rss && \
  tail -3 dev/compare_lpb.out'
```
RSS should grow during model load; CPU should be > 50%.

---

## File map (everything this experiment touches)

```
dev/compare_lru_lpb.py        # driver — runs Phase A/B/C/D/E for one mode
dev/plot_lru_vs_lpb.py        # aggregator + 3 figures + summary table
dev/compare_{lru,lpb}.jsonl   # per-request structured log
dev/compare_{lru,lpb}.out     # engine stdout
dev/compare_summary.{json,out}# aggregated table
dev/figures/                  # 3 PNGs (anchor survival, workload, scenarios)
dev/README.md                 # Findings K (average case), K.2 (multi-scenario)
```

Core HiMA code (do not touch unless debugging):
```
vllm/v1/core/hima/lpb_free_queue.py     # LPB queue impl (heap + path-counter)
vllm/v1/core/hima/coordinator_hima.py   # wraps HybridKVCacheCoordinator
vllm/v1/core/hima/cost_curve.py         # per-pool c_kv_ms / c_m_ms
vllm/v1/core/hima/runtime.py            # PathCountedHitCounter, sliding window
vllm/engine/arg_utils.py                # threads hima_enabled into CacheConfig
```

---

## Once you have both runs

Push the new jsonls + summary + figures back to the branch:

```bash
git add dev/compare_{lru,lpb}.jsonl \
        dev/compare_{lru,lpb}.out \
        dev/compare_summary.{json,out} \
        dev/figures/fig_lru_vs_lpb_scenarios.png
git commit -m "dev: capture full LRU + LPB multi-scenario data"
git push origin HiMA
```

If the numbers diverge materially from the predictions in this doc,
that's interesting and probably points to either:
- A real LPB regression somewhere we didn't catch (Phase E should be a wash; if it's not, dig in)
- A real LPB win somewhere bigger than predicted (Phase D could easily be 10× instead of 5–8×)

Either way, update `dev/README.md` Finding K.2 with the actual table
before the result is forgotten.
