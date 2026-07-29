#!/usr/bin/env python3
"""Benchmark Odyssey RMSNorm forward / backward throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import NormConfig, OdysseyRMSNorm
from odyssey.config import REPO_ROOT


def _measure(fn, *, warmup: int, repeats: int, sync_cuda: bool) -> float:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        if sync_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "ODY-0005" / "metrics.json",
    )
    args = parser.parse_args()

    cfg = NormConfig(
        hidden_size=args.hidden_size,
        epsilon=args.eps,
        device=args.device,
        dtype=args.dtype,
    )
    norm = OdysseyRMSNorm(cfg)
    dtype = cfg.torch_dtype
    x = torch.randn(
        args.batch, args.seq, args.hidden_size, device=args.device, dtype=dtype
    )
    sync = args.device.startswith("cuda")

    fwd_s = _measure(
        lambda: norm(x),
        warmup=args.warmup,
        repeats=args.repeats,
        sync_cuda=sync,
    )

    x_b = x.detach().requires_grad_(True)

    def _bwd() -> None:
        if x_b.grad is not None:
            x_b.grad = None
        if norm.weight.grad is not None:
            norm.weight.grad = None
        y = norm(x_b)
        y.pow(2).mean().backward()

    bwd_s = _measure(
        _bwd, warmup=args.warmup, repeats=args.repeats, sync_cuda=sync
    )

    elems = args.batch * args.seq * args.hidden_size
    metrics = {
        "hidden_size": args.hidden_size,
        "epsilon": args.eps,
        "dtype": args.dtype,
        "device": args.device,
        "shape": [args.batch, args.seq, args.hidden_size],
        "parameter_count": norm.parameter_count(),
        "memory_bytes": norm.memory_bytes(),
        "forward_mean_seconds": round(fwd_s, 8),
        "backward_mean_seconds": round(bwd_s, 8),
        "forward_elems_per_second": round(elems / fwd_s, 2),
        "backward_elems_per_second": round(elems / bwd_s, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
