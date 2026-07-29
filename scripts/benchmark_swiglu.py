#!/usr/bin/env python3
"""Benchmark Odyssey SwiGLU forward throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import FeedForwardConfig, OdysseySwiGLU
from odyssey.config import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "ODY-0006" / "metrics.json",
    )
    args = parser.parse_args()

    cfg = FeedForwardConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        device=args.device,
        dtype=args.dtype,
    )
    ffn = OdysseySwiGLU(cfg)
    x = torch.randn(
        args.batch,
        args.seq,
        args.hidden_size,
        device=args.device,
        dtype=cfg.torch_dtype,
    )
    sync = args.device.startswith("cuda")
    for _ in range(args.warmup):
        _ = ffn(x)
    if sync:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(args.repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ffn(x)
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    metrics = {
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "dtype": args.dtype,
        "device": args.device,
        "shape": list(x.shape),
        "parameter_count": ffn.parameter_count(),
        "memory_bytes": ffn.memory_bytes(),
        "forward_mean_seconds": round(mean_s, 8),
        "tokens_per_second": round(args.batch * args.seq / mean_s, 2),
        "inspect": ffn.inspect(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
