"""Compression / quality statistics for OdysseyTokenizer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TokenizerStats:
    vocab_size: int
    merge_count: int
    average_token_length: float
    compression_ratio: float
    compression_percent: float
    unknown_token_frequency: float
    encoding_tokens_per_second: float
    decoding_tokens_per_second: float
    characters: int
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "merge_count": self.merge_count,
            "average_token_length": self.average_token_length,
            "compression_ratio": self.compression_ratio,
            "compression_percent": self.compression_percent,
            "unknown_token_frequency": self.unknown_token_frequency,
            "encoding_tokens_per_second": self.encoding_tokens_per_second,
            "decoding_tokens_per_second": self.decoding_tokens_per_second,
            "characters": self.characters,
            "tokens": self.tokens,
        }


def compression_summary(characters: int, tokens: int) -> tuple[float, float]:
    """Return (chars_per_token, percent_reduction)."""
    if tokens <= 0:
        return 0.0, 0.0
    ratio = characters / tokens
    percent = (1.0 - (tokens / characters)) * 100.0 if characters else 0.0
    return ratio, percent


def timed_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
