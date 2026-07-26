"""SentencePiece training orchestration."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from tokenizer.sentencepiece.config import TokenizerConfig, load_tokenizer_config
from tokenizer.sentencepiece.normalizer import TextNormalizer
from tokenizer.sentencepiece.utils import ensure_parent, write_json


@dataclass(slots=True)
class TrainResult:
    """Artifacts produced by a successful tokenizer training run."""

    model_path: Path
    vocab_path: Path
    metadata_path: Path
    vocab_size: int
    training_seconds: float
    corpus_path: Path
    corpus_lines: int


class SentencePieceTrainer:
    """Train a SentencePiece model from a raw text corpus."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or load_tokenizer_config()
        self.normalizer = TextNormalizer(self.config)

    def train(
        self,
        input_path: str | Path,
        *,
        model_prefix: str | Path | None = None,
        vocab_size: int | None = None,
        normalize_corpus: bool = True,
    ) -> TrainResult:
        """Train SentencePiece and write model/vocab/metadata artifacts."""
        corpus = Path(input_path)
        if not corpus.is_file():
            raise FileNotFoundError(f"Training corpus not found: {corpus}")

        prefix = Path(model_prefix) if model_prefix else self.config.model_prefix_path()
        if not prefix.is_absolute():
            prefix = self.config.resolve(prefix)
        ensure_parent(Path(str(prefix) + ".model"))

        effective_vocab = vocab_size or self.config.vocab_size
        train_input = corpus
        temp_dir: tempfile.TemporaryDirectory[str] | None = None

        if normalize_corpus:
            temp_dir = tempfile.TemporaryDirectory(prefix="odyssey-spm-")
            normalized = Path(temp_dir.name) / "corpus.normalized.txt"
            self.normalizer.normalize_file(str(corpus), str(normalized))
            train_input = normalized

        line_count = sum(1 for _ in train_input.open("r", encoding="utf-8"))
        if line_count == 0:
            raise ValueError(f"Training corpus is empty after normalization: {corpus}")

        # SentencePiece refuses vocab sizes that exceed what the corpus can support.
        # Keep a small safety margin above character inventory + special symbols.
        user_symbols = list(self.config.user_defined_symbols)
        train_kwargs: dict[str, Any] = {
            "input": str(train_input),
            "model_prefix": str(prefix),
            "vocab_size": effective_vocab,
            "character_coverage": self.config.character_coverage,
            "model_type": self.config.model_type,
            "pad_id": self.config.pad_id,
            "bos_id": self.config.bos_id,
            "eos_id": self.config.eos_id,
            "unk_id": self.config.unk_id,
            "unk_piece": self.config.special_tokens["unk"],
            "bos_piece": self.config.special_tokens["bos"],
            "eos_piece": self.config.special_tokens["eos"],
            "pad_piece": self.config.special_tokens["pad"],
            "user_defined_symbols": user_symbols,
            "num_threads": self.config.training.num_threads,
            "shuffle_input_sentence": self.config.training.shuffle_input_sentence,
            "max_sentence_length": self.config.training.max_sentence_length,
            "hard_vocab_limit": self.config.training.hard_vocab_limit,
        }
        if self.config.training.input_sentence_size > 0:
            train_kwargs["input_sentence_size"] = (
                self.config.training.input_sentence_size
            )

        start = time.perf_counter()
        try:
            self._run_sentencepiece_train(train_kwargs)
        except RuntimeError as exc:
            # Tiny / homogeneous corpora cannot always materialize the requested
            # vocab size when hard_vocab_limit=true. Retry softly so research
            # experiments still produce a usable reference tokenizer.
            message = str(exc)
            if (
                self.config.training.hard_vocab_limit
                and "Vocabulary size too high" in message
            ):
                train_kwargs["hard_vocab_limit"] = False
                self._run_sentencepiece_train(train_kwargs)
            else:
                raise
        elapsed = time.perf_counter() - start

        model_path = Path(str(prefix) + ".model")
        vocab_path = Path(str(prefix) + ".vocab")
        if not model_path.is_file():
            raise RuntimeError(f"SentencePiece did not write model file: {model_path}")

        # Align configured path names when using the default prefix.
        configured_model = self.config.model_path()
        configured_vocab = self.config.vocab_path()
        if prefix == self.config.model_prefix_path():
            if model_path != configured_model:
                ensure_parent(configured_model)
                model_path.replace(configured_model)
                model_path = configured_model
            if vocab_path != configured_vocab and vocab_path.is_file():
                ensure_parent(configured_vocab)
                vocab_path.replace(configured_vocab)
                vocab_path = configured_vocab

        processor = spm.SentencePieceProcessor()
        processor.load(str(model_path))
        actual_vocab = int(processor.get_piece_size())

        metadata_path = self.config.metadata_path()
        if prefix != self.config.model_prefix_path():
            metadata_path = prefix.parent / "metadata.json"

        metadata = {
            "backend": "sentencepiece",
            "phase": 1,
            "experiment": "ODY-0001",
            "vocab_size_requested": effective_vocab,
            "vocab_size_actual": actual_vocab,
            "model_type": self.config.model_type,
            "character_coverage": self.config.character_coverage,
            "training_seconds": elapsed,
            "corpus_path": str(corpus),
            "corpus_lines": line_count,
            "model_file": str(model_path),
            "vocab_file": str(vocab_path),
            "special_token_ids": {
                "pad": self.config.pad_id,
                "bos": self.config.bos_id,
                "eos": self.config.eos_id,
                "unk": self.config.unk_id,
            },
            "user_defined_symbols": user_symbols,
            "config": self.config.to_dict(),
        }
        write_json(metadata_path, metadata)

        if temp_dir is not None:
            temp_dir.cleanup()

        return TrainResult(
            model_path=model_path,
            vocab_path=vocab_path,
            metadata_path=metadata_path,
            vocab_size=actual_vocab,
            training_seconds=elapsed,
            corpus_path=corpus,
            corpus_lines=line_count,
        )

    @staticmethod
    def _run_sentencepiece_train(train_kwargs: dict[str, Any]) -> None:
        """Invoke SentencePiece training with a loosely typed kwargs map."""
        spm.SentencePieceTrainer.train(**train_kwargs)  # type: ignore[arg-type]
