#!/usr/bin/env python3
"""Benchmark Odyssey RoPE cache build + rotate throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import OdysseyRoPE, RopeConfig
from model.rope_visualizer import export_rope_assets
from odyssey.config import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--max-position", type=int, default=4096)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "ODY-0004" / "metrics.json",
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    cfg = RopeConfig(
        theta=10000.0,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        max_position_embeddings=args.max_position,
        device=args.device,
        dtype="float32",
    )

    t0 = time.perf_counter()
    rope = OdysseyRoPE(cfg)
    # Force full cache materialization
    _ = rope(torch.zeros(1, 1, args.head_dim, device=args.device), position_offset=0)
    cache_s = time.perf_counter() - t0

    x = torch.randn(args.batch, args.seq, args.heads, args.head_dim, device=args.device)
    for _ in range(args.warmup):
        _ = rope(x, position_offset=0)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(args.repeats):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = rope(x, position_offset=0)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_s = sum(times) / len(times)
    elems = args.batch * args.seq * args.heads * args.head_dim
    metrics = {
        "theta": 10000.0,
        "rotary_dim": args.rotary_dim,
        "head_dim": args.head_dim,
        "max_position_embeddings": args.max_position,
        "cache_build_seconds": round(cache_s, 6),
        "cache_memory_bytes": rope.cache_memory_bytes(),
        "rotate_mean_seconds": round(mean_s, 8),
        "rotate_elems_per_second": round(elems / mean_s, 2),
        "device": args.device,
        "shape": list(x.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.visualize:
        paths = export_rope_assets(cfg)
        print("visualizations:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
