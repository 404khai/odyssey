"""Shared helpers for the SentencePiece tokenizer pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create parent directories for a target file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty-printed UTF-8 JSON."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def timed(fn_name: str = "operation") -> Any:
    """Tiny helper returning (result, elapsed_seconds) for callables."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    _ = fn_name
    return decorator


def average_token_length(pieces: Sequence[str]) -> float:
    """Mean character length of token surface forms."""
    if not pieces:
        return 0.0
    return sum(len(piece) for piece in pieces) / len(pieces)


def compression_ratio(text: str, token_count: int) -> float:
    """Characters per token — higher means better compression."""
    if token_count <= 0:
        return 0.0
    return len(text) / token_count


def count_lines(path: Path) -> int:
    """Count newline-terminated records in a text file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield fixed-size slices from a sequence."""
    if size <= 0:
        raise ValueError("size must be positive")
    for index in range(0, len(items), size):
        yield items[index : index + size]
