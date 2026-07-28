"""Byte-level BPE trainer implemented from first principles.

Performance notes:
    Pair counts are maintained incrementally when a merge is applied so training
    stays practical on multi-thousand-line corpora without sacrificing
    deterministic greedy BPE semantics.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from odyssey_tokenizer.config import BPEConfig, load_bpe_config
from odyssey_tokenizer.merges import MergeTable
from odyssey_tokenizer.normalizer import TextNormalizer
from odyssey_tokenizer.vocabulary import Vocabulary


@dataclass(slots=True)
class TrainResult:
    vocabulary: Vocabulary
    merges: MergeTable
    training_seconds: float
    corpus_lines: int
    corpus_bytes: int
    vocab_size: int
    merge_count: int
    pair_stats: list[tuple[bytes, bytes, int]]


def _iter_corpus_lines(
    path: Path,
    *,
    max_lines: int,
    shuffle: bool,
    seed: int,
) -> list[str]:
    lines = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_lines > 0:
        lines = lines[:max_lines]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)
    return lines


def _bytes_to_symbols(data: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([value]) for value in data)


def _count_pairs_from_words(
    words: list[list[bytes]], freqs: list[int]
) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for symbols, freq in zip(words, freqs, strict=True):
        if len(symbols) < 2:
            continue
        for index in range(len(symbols) - 1):
            counts[(symbols[index], symbols[index + 1])] += freq
    return counts


def _merge_word(symbols: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    left, right = pair
    merged = left + right
    output: list[bytes] = []
    index = 0
    while index < len(symbols):
        if (
            index < len(symbols) - 1
            and symbols[index] == left
            and symbols[index + 1] == right
        ):
            output.append(merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return output


class BPETrainer:
    """Train a deterministic byte-level BPE vocabulary from a text corpus."""

    def __init__(self, config: BPEConfig | None = None) -> None:
        self.config = config or load_bpe_config()
        self.normalizer = TextNormalizer(self.config)

    def train(self, corpus_path: str | Path) -> TrainResult:
        path = Path(corpus_path)
        if not path.is_file():
            raise FileNotFoundError(f"Training corpus not found: {path}")

        start = time.perf_counter()
        lines = _iter_corpus_lines(
            path,
            max_lines=self.config.training.max_lines,
            shuffle=self.config.training.shuffle_input_sentence,
            seed=self.config.training.seed,
        )
        if not lines:
            raise ValueError(f"Training corpus is empty: {path}")

        # Collapse identical normalized lines into a frequency dictionary.
        line_freqs: Counter[tuple[bytes, ...]] = Counter()
        corpus_bytes = 0
        for line in lines:
            normalized = self.normalizer.normalize(line)
            if not normalized:
                continue
            encoded = normalized.encode("utf-8")
            corpus_bytes += len(encoded)
            line_freqs[_bytes_to_symbols(encoded)] += 1

        words: list[list[bytes]] = []
        freqs: list[int] = []
        for symbols, freq in line_freqs.items():
            words.append(list(symbols))
            freqs.append(freq)

        vocabulary = Vocabulary.from_specials_and_bytes(self.config.special_tokens)
        merges = MergeTable()
        pair_stats: list[tuple[bytes, bytes, int]] = []
        pair_counts = _count_pairs_from_words(words, freqs)

        target_merges = self.config.vocab_size - len(vocabulary)
        if target_merges <= 0:
            raise ValueError("vocab_size too small for specials + byte alphabet")

        for _ in range(target_merges):
            if not pair_counts:
                break
            best_pair, best_freq = max(
                pair_counts.items(),
                key=lambda item: (item[1], item[0][0], item[0][1]),
            )
            if best_freq < self.config.min_frequency:
                break

            merges.add(best_pair[0], best_pair[1], frequency=best_freq)
            vocabulary.add(best_pair[0] + best_pair[1])
            pair_stats.append((best_pair[0], best_pair[1], best_freq))

            # Apply merge and refresh pair counts only for affected words.
            touched: list[int] = []
            left, right = best_pair
            for word_index, word_symbols in enumerate(words):
                if len(word_symbols) < 2:
                    continue
                if any(
                    word_symbols[i] == left and word_symbols[i + 1] == right
                    for i in range(len(word_symbols) - 1)
                ):
                    touched.append(word_index)

            for word_index in touched:
                old = words[word_index]
                freq = freqs[word_index]
                # Remove old adjacent pairs.
                for index in range(len(old) - 1):
                    pair = (old[index], old[index + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                new = _merge_word(old, best_pair)
                words[word_index] = new
                for index in range(len(new) - 1):
                    pair_counts[(new[index], new[index + 1])] += freq

        elapsed = time.perf_counter() - start
        return TrainResult(
            vocabulary=vocabulary,
            merges=merges,
            training_seconds=elapsed,
            corpus_lines=len(lines),
            corpus_bytes=corpus_bytes,
            vocab_size=len(vocabulary),
            merge_count=len(merges),
            pair_stats=pair_stats,
        )


def train_bpe(corpus_path: str | Path, config: BPEConfig | None = None) -> TrainResult:
    """Convenience wrapper around :class:`BPETrainer`."""
    return BPETrainer(config).train(corpus_path)


def get_pairs(symbols: Iterable[bytes]) -> set[tuple[bytes, bytes]]:
    """Return adjacent pairs in a symbol sequence (useful for tests / demos)."""
    sequence = list(symbols)
    return {
        (sequence[index], sequence[index + 1]) for index in range(len(sequence) - 1)
    }
