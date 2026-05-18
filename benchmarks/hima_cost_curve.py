# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiMA cost-curve calibration script.

Estimates the polynomial coefficients of

    c_KV(L) = alpha_kv * L^2 + beta_kv * L + gamma_kv     [microseconds]
    c_M(L)  = alpha_m  * L   + beta_m                     [microseconds]

by measuring prefill latency vs prefix length on the local GPU. The
output JSON can be fed back to HiMA via the ``VLLM_HIMA_CSIGMA_JSON``
environment variable, so the LPB scoring and the Cross-Pool Planner
both reflect the actual cost characteristics of your model + hardware
combination -- including RTX PRO 6000 Blackwell, where the H200
defaults in ``LEGACY_DEFAULT`` are *not* representative.

Usage::

    python benchmarks/hima_cost_curve.py \\
        --model Qwen/Qwen3-Next-7B-Instruct \\
        --lengths 256,1024,4096,16384 \\
        --output csigma_blackwell.json

Notes:
    * Requires a real GPU and torch. We import torch lazily so this
      module is still importable in CPU-only contexts (e.g. lint CI).
    * The mamba-pool curve (c_M) requires a hybrid model; for
      pure-attention models the script reports only c_KV and emits a
      ``"m_alpha": null`` so the JSON loader knows to keep the default.
    * The script does NOT depend on vLLM internals -- it drives the
      model directly via ``transformers``. This keeps it usable as a
      one-shot calibration step before HiMA is enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class _Sample:
    length: int
    latency_us: float


@dataclass
class _Curve:
    points: list[_Sample]
    alpha: float
    beta: float
    gamma: float

    def predict_us(self, length: int) -> float:
        return self.alpha * (length**2) + self.beta * length + self.gamma


def _fit_quadratic(samples: list[_Sample]) -> _Curve:
    """Least-squares fit of ``alpha*L^2 + beta*L + gamma``.

    Implemented with plain Python so we don't pull NumPy. With <= 10
    samples this is fine.
    """
    if len(samples) < 3:
        raise ValueError("need at least 3 samples to fit a quadratic")
    # Build the normal equations: A^T A x = A^T y where each row of A is
    # [L^2, L, 1]. Inverse 3x3 manually.
    n = len(samples)
    sx0 = float(n)
    sx1 = sum(s.length for s in samples)
    sx2 = sum(s.length**2 for s in samples)
    sx3 = sum(s.length**3 for s in samples)
    sx4 = sum(s.length**4 for s in samples)
    sy0 = sum(s.latency_us for s in samples)
    sy1 = sum(s.length * s.latency_us for s in samples)
    sy2 = sum((s.length**2) * s.latency_us for s in samples)
    # Solve [[sx4, sx3, sx2], [sx3, sx2, sx1], [sx2, sx1, sx0]] * [a, b, c]
    #     = [sy2, sy1, sy0]
    m = [[sx4, sx3, sx2, sy2], [sx3, sx2, sx1, sy1], [sx2, sx1, sx0, sy0]]
    # Gaussian elimination
    for i in range(3):
        pivot = m[i][i]
        if pivot == 0:
            raise ValueError("singular system; cannot fit")
        for k in range(i + 1, 3):
            factor = m[k][i] / pivot
            for j in range(i, 4):
                m[k][j] -= factor * m[i][j]
    # Back-substitution
    a = m[0][3] / m[0][0] if m[0][0] != 0 else 0.0
    # update vector after back-sub: m[i][3] holds residual
    c = m[2][3] / m[2][2]
    b = (m[1][3] - m[1][2] * c) / m[1][1]
    a = (m[0][3] - m[0][1] * b - m[0][2] * c) / m[0][0]
    return _Curve(points=samples, alpha=a, beta=b, gamma=c)


def _fit_linear(samples: list[_Sample]) -> _Curve:
    """Least-squares fit of ``alpha * L + beta``."""
    if len(samples) < 2:
        raise ValueError("need at least 2 samples for linear fit")
    n = len(samples)
    sx = sum(s.length for s in samples)
    sy = sum(s.latency_us for s in samples)
    sxy = sum(s.length * s.latency_us for s in samples)
    sxx = sum(s.length**2 for s in samples)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("singular system; degenerate samples")
    alpha = (n * sxy - sx * sy) / denom
    beta = (sy - alpha * sx) / n
    return _Curve(points=samples, alpha=alpha, beta=beta, gamma=0.0)


def _bench_prefill(
    model: str,
    lengths: list[int],
    warmup: int,
    iters: int,
    device: str,
    dtype: str,
) -> list[_Sample]:
    """Drive the model through prefill at several lengths.

    Each iteration creates a random input of the target length and
    measures ``model(...)`` wall time on the GPU (after a CUDA sync).
    """
    import torch  # noqa: PLC0415 -- intentional lazy import
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"Loading model {model!r} (dtype={dtype}) on {device}...")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    mdl.eval()

    vocab = config.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    samples: list[_Sample] = []
    with torch.inference_mode():
        for L in lengths:
            input_ids = torch.randint(
                low=1, high=vocab, size=(1, L), device=device, dtype=torch.long
            )
            mask = torch.ones_like(input_ids)
            # Warm up
            for _ in range(warmup):
                _ = mdl(input_ids=input_ids, attention_mask=mask, use_cache=False)
            torch.accelerator.synchronize()
            ts = []
            for _ in range(iters):
                torch.accelerator.synchronize()
                t0 = time.perf_counter()
                _ = mdl(input_ids=input_ids, attention_mask=mask, use_cache=False)
                torch.accelerator.synchronize()
                t1 = time.perf_counter()
                ts.append((t1 - t0) * 1_000_000.0)
            med = statistics.median(ts)
            samples.append(_Sample(length=L, latency_us=med))
            print(f"  L={L:>6}  median={med:>9.2f} us  (n={iters}, pad_id={pad_id})")
    del mdl
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--lengths",
        default="256,1024,4096,16384",
        help="Comma-separated prefix lengths (tokens) to probe.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--output", default="hima_csigma.json")
    parser.add_argument(
        "--skip-mamba",
        action="store_true",
        help="Skip the linear (mamba) fit even if it would succeed. "
        "Use when the model has no recurrent layers.",
    )
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: torch is required. Install via `uv pip install -e . "
            "--torch-backend=auto` or similar.",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is unavailable; cost calibration requires a GPU.",
            file=sys.stderr,
        )
        return 3

    lengths = sorted({int(x) for x in args.lengths.split(",") if x.strip()})
    if len(lengths) < 3:
        print("ERROR: need >= 3 distinct lengths for a quadratic fit.", file=sys.stderr)
        return 4

    print("=" * 60)
    print(" HiMA cost-curve calibration")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Device:   {args.device}")
    print(f"Lengths:  {lengths}")
    print(f"Iters:    warmup={args.warmup}, timed={args.iters}")
    samples = _bench_prefill(
        args.model, lengths, args.warmup, args.iters, args.device, args.dtype
    )

    kv_curve = _fit_quadratic(samples)
    out: dict[str, object] = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "samples": [asdict(s) for s in samples],
        "kv_alpha": kv_curve.alpha,
        "kv_beta": kv_curve.beta,
        "kv_gamma": kv_curve.gamma,
    }

    if not args.skip_mamba:
        try:
            m_curve = _fit_linear(samples)
            out["m_alpha"] = m_curve.alpha
            out["m_beta"] = m_curve.beta
        except ValueError:
            out["m_alpha"] = None
            out["m_beta"] = None

    # Cross-over (L*): where c_KV(L) = c_M(L)
    if out.get("m_alpha") is not None:
        # Solve: a*L^2 + b*L + c = a_m*L + b_m
        a, b, c = kv_curve.alpha, kv_curve.beta, kv_curve.gamma
        a_m, b_m = out["m_alpha"], out["m_beta"]
        disc = (b - a_m) ** 2 - 4 * a * (c - b_m)
        if disc >= 0 and a > 0:
            l_star = (-(b - a_m) + math.sqrt(disc)) / (2 * a)
            out["L_star"] = max(1.0, l_star)

    print("\nFit results:")
    print(
        f"  c_KV(L) = {kv_curve.alpha:.3e} * L^2 "
        f"+ {kv_curve.beta:.3e} * L "
        f"+ {kv_curve.gamma:.3e}"
    )
    if out.get("m_alpha") is not None:
        print(f"  c_M (L) = {out['m_alpha']:.3e} * L + {out['m_beta']:.3e}")
    if out.get("L_star") is not None:
        print(f"  L* (cross-over) = {out['L_star']:.0f} tokens")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.output)}")
    print("To activate, set:")
    print(f"  export VLLM_HIMA_CSIGMA_JSON={os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
