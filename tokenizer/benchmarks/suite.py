"""Benchmark suite for OdysseyTokenizer."""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Any

from odyssey_tokenizer import OdysseyTokenizer
from odyssey_tokenizer.statistics import timed_call


def run_benchmark(
    model_path: str | Path,
    corpus_path: str | Path,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    """Measure encode/decode speed, compression, and peak memory."""
    tokenizer = OdysseyTokenizer.load(model_path)
    lines = [
        line.strip()
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    if not lines:
        raise ValueError("Benchmark corpus is empty")

    tracemalloc.start()
    ids_batch, encode_seconds = timed_call(
        lambda: [tokenizer.encode(line) for line in lines]
    )
    _, decode_seconds = timed_call(lambda: [tokenizer.decode(ids) for ids in ids_batch])
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = tokenizer.compute_stats(lines)
    total_tokens = sum(len(ids) for ids in ids_batch)
    return {
        "model": str(model_path),
        "lines": len(lines),
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "total_tokens": total_tokens,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "encoding_tokens_per_second": total_tokens / max(encode_seconds, 1e-9),
        "decoding_tokens_per_second": total_tokens / max(decode_seconds, 1e-9),
        "compression_ratio": stats.compression_ratio,
        "compression_percent": stats.compression_percent,
        "unknown_token_frequency": stats.unknown_token_frequency,
        "memory_current_bytes": current,
        "memory_peak_bytes": peak,
        "stats": stats.to_dict(),
    }
