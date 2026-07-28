#!/usr/bin/env python3
"""Benchmark OdysseyEmbedding lookup latency, memory, and init time."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import torch

from model import EmbeddingConfig, OdysseyEmbedding
from model.embedding_visualizer import export_embedding_preview
from odyssey.config import REPO_ROOT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    *,
    vocab_size: int,
    hidden_size: int,
    batch: int,
    seq: int,
    warmup: int,
    repeats: int,
    device: str,
    init_strategy: str,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    cfg = EmbeddingConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        padding_idx=0,
        init_strategy=init_strategy,  # type: ignore[arg-type]
        device=device,
        dtype="float32",
    )

    t0 = time.perf_counter()
    emb = OdysseyEmbedding(cfg)
    init_s = time.perf_counter() - t0

    ids = torch.randint(0, vocab_size, (batch, seq), device=cfg.torch_device)

    for _ in range(warmup):
        _ = emb(ids)
    _sync(cfg.torch_device)

    times: list[float] = []
    for _ in range(repeats):
        _sync(cfg.torch_device)
        start = time.perf_counter()
        out = emb(ids)
        _sync(cfg.torch_device)
        times.append(time.perf_counter() - start)
        assert out.shape == (batch, seq, hidden_size)

    mean_s = sum(times) / len(times)
    tokens = batch * seq
    tok_per_s = tokens / mean_s if mean_s > 0 else float("inf")

    tracemalloc.start()
    _ = emb(ids)
    _sync(cfg.torch_device)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    info = emb.inspect()
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "batch": batch,
        "sequence": seq,
        "device": device,
        "init_strategy": init_strategy,
        "init_seconds": round(init_s, 6),
        "lookup_mean_seconds": round(mean_s, 8),
        "lookup_tokens_per_second": round(tok_per_s, 2),
        "parameter_count": info.parameter_count,
        "weight_memory_bytes": info.memory_bytes,
        "tracemalloc_peak_bytes": peak,
        "cuda_available": torch.cuda.is_available(),
        "output_shape": list(out.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-strategy", default="xavier_uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "ODY-0003" / "metrics.json",
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    metrics = benchmark(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        batch=args.batch,
        seq=args.seq,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        init_strategy=args.init_strategy,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.visualize:
        torch.manual_seed(args.seed)
        emb = OdysseyEmbedding(
            EmbeddingConfig(
                vocab_size=min(args.vocab_size, 2048),
                hidden_size=min(args.hidden_size, 128),
                padding_idx=0,
                init_strategy=args.init_strategy,  # type: ignore[arg-type]
            )
        )
        paths = export_embedding_preview(emb)
        print("visualizations:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
