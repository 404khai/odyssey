"""Public SentencePiece tokenizer API for Odyssey Phase 1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

from tokenizer.sentencepiece.config import TokenizerConfig, load_tokenizer_config
from tokenizer.sentencepiece.decoder import TokenizerDecoder
from tokenizer.sentencepiece.encoder import TokenizerEncoder
from tokenizer.sentencepiece.normalizer import TextNormalizer
from tokenizer.sentencepiece.special_tokens import special_token_surfaces
from tokenizer.sentencepiece.trainer import SentencePieceTrainer, TrainResult
from tokenizer.sentencepiece.utils import average_token_length, read_json, write_json


@dataclass(slots=True)
class TokenizerStats:
    """Aggregate tokenizer quality / performance metrics."""

    vocab_size: int
    average_token_length: float
    compression_ratio: float
    unknown_token_frequency: float
    encoding_tokens_per_second: float
    decoding_tokens_per_second: float
    model_file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "average_token_length": self.average_token_length,
            "compression_ratio": self.compression_ratio,
            "unknown_token_frequency": self.unknown_token_frequency,
            "encoding_tokens_per_second": self.encoding_tokens_per_second,
            "decoding_tokens_per_second": self.decoding_tokens_per_second,
            "model_file_size_bytes": self.model_file_size_bytes,
        }


@dataclass(slots=True)
class InspectionResult:
    """Structured output for the tokenizer inspector."""

    input_text: str
    normalized_text: str
    pieces: list[str]
    ids: list[int]
    decoded_text: str

    def render(self) -> str:
        piece_line = " ".join(self.pieces)
        id_line = "[" + ", ".join(str(token_id) for token_id in self.ids) + "]"
        return "\n".join(
            [
                "Input",
                self.input_text,
                "",
                "Normalized",
                self.normalized_text,
                "",
                "Tokens",
                piece_line,
                "",
                "IDs",
                id_line,
                "",
                "Decoded Text",
                self.decoded_text,
            ]
        )


class OdysseySentencePieceTokenizer:
    """Reference SentencePiece tokenizer used throughout early Odyssey development.

    This is intentionally NOT the final Odyssey tokenizer. Phase 2 replaces the
    internals with a from-scratch BPE implementation while keeping a similar API.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or load_tokenizer_config()
        self.normalizer = TextNormalizer(self.config)
        self._processor: spm.SentencePieceProcessor | None = None
        self._encoder: TokenizerEncoder | None = None
        self._decoder: TokenizerDecoder | None = None
        self._model_path: Path | None = None

    @property
    def processor(self) -> spm.SentencePieceProcessor:
        if self._processor is None:
            raise RuntimeError("Tokenizer model is not loaded. Call load() or train().")
        return self._processor

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    def train(
        self,
        input_path: str | Path,
        *,
        model_prefix: str | Path | None = None,
        vocab_size: int | None = None,
        normalize_corpus: bool = True,
    ) -> TrainResult:
        """Train a new SentencePiece model and load it into this instance."""
        trainer = SentencePieceTrainer(self.config)
        result = trainer.train(
            input_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            normalize_corpus=normalize_corpus,
        )
        self.load(result.model_path)
        return result

    def load(self, model_path: str | Path | None = None) -> None:
        """Load a SentencePiece ``.model`` file."""
        path = Path(model_path) if model_path else self.config.model_path()
        if not path.is_absolute():
            path = self.config.resolve(path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer model not found: {path}")

        processor = spm.SentencePieceProcessor()
        if not processor.load(str(path)):
            raise RuntimeError(f"Failed to load SentencePiece model: {path}")

        self._processor = processor
        self._model_path = path
        self._encoder = TokenizerEncoder(processor, self.config, self.normalizer)
        self._decoder = TokenizerDecoder(processor, self.config)

    def save(
        self,
        model_prefix: str | Path | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Copy the loaded model/vocab to ``model_prefix`` and write metadata.

        SentencePiece models are immutable binary artifacts. Saving therefore
        copies the on-disk model produced by ``train()`` / ``load()``.
        """
        if self._processor is None or self._model_path is None:
            raise RuntimeError("Cannot save before load()/train().")

        source_model = self._model_path
        if not source_model.is_file():
            raise FileNotFoundError(
                f"No source model available to save from: {source_model}"
            )

        prefix = Path(model_prefix) if model_prefix else self.config.model_prefix_path()
        if not prefix.is_absolute():
            prefix = self.config.resolve(prefix)

        dest_model = Path(str(prefix) + ".model")
        dest_vocab = Path(str(prefix) + ".vocab")
        dest_model.parent.mkdir(parents=True, exist_ok=True)

        dest_model.write_bytes(source_model.read_bytes())
        source_vocab = source_model.with_suffix(".vocab")
        if source_vocab.is_file():
            dest_vocab.write_bytes(source_vocab.read_bytes())
        else:
            self._export_vocab(dest_vocab)

        payload = {
            "backend": "sentencepiece",
            "vocab_size": self.vocab_size,
            "config": self.config.to_dict(),
            "special_token_ids": self.special_token_ids(),
        }
        if metadata:
            payload.update(metadata)
        write_json(dest_model.parent / "metadata.json", payload)
        return dest_model, dest_vocab

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        """Encode text to token IDs."""
        return self._require_encoder().encode(
            text, add_bos=add_bos, add_eos=add_eos, normalize=normalize
        )

    def decode(self, ids: list[int], *, skip_special_ids: bool = False) -> str:
        """Decode token IDs to text."""
        return self._require_decoder().decode(ids, skip_special_ids=skip_special_ids)

    def encode_as_pieces(self, text: str, *, normalize: bool = True) -> list[str]:
        """Encode text to token surface forms."""
        return self._require_encoder().encode_as_pieces(text, normalize=normalize)

    def piece_to_id(self, piece: str) -> int:
        return int(self.processor.piece_to_id(piece))

    def id_to_piece(self, token_id: int) -> str:
        return str(self.processor.id_to_piece(token_id))

    def special_token_ids(self) -> dict[str, int]:
        """Resolve reserved token surfaces to their vocabulary IDs."""
        ids = {
            "pad": self.config.pad_id,
            "bos": self.config.bos_id,
            "eos": self.config.eos_id,
            "unk": self.config.unk_id,
        }
        for surface in self.config.user_defined_symbols:
            ids[surface.strip("<>")] = self.piece_to_id(surface)
        return ids

    def has_reserved_tokens(self) -> bool:
        """Return True when every configured reserved surface is in the vocab."""
        for surface in special_token_surfaces(self.config):
            token_id = self.piece_to_id(surface)
            if self.id_to_piece(token_id) != surface:
                return False
        return True

    def inspect(self, text: str) -> InspectionResult:
        """Return a structured Input → Tokens → IDs → Decoded view."""
        normalized = self.normalizer.normalize(text)
        pieces = self.encode_as_pieces(text)
        ids = self.encode(text)
        decoded = self.decode(ids)
        return InspectionResult(
            input_text=text,
            normalized_text=normalized,
            pieces=pieces,
            ids=ids,
            decoded_text=decoded,
        )

    def compute_stats(self, sample_texts: list[str]) -> TokenizerStats:
        """Compute vocabulary / compression / speed statistics on sample texts."""
        if not sample_texts:
            raise ValueError("sample_texts must be non-empty")

        all_pieces: list[str] = []
        total_chars = 0
        total_tokens = 0
        unk_count = 0

        encode_start = time.perf_counter()
        encoded_batch: list[list[int]] = []
        for text in sample_texts:
            ids = self.encode(text)
            pieces = self.encode_as_pieces(text)
            encoded_batch.append(ids)
            all_pieces.extend(pieces)
            total_chars += len(self.normalizer.normalize(text))
            total_tokens += len(ids)
            unk_count += sum(1 for token_id in ids if token_id == self.config.unk_id)
        encode_elapsed = max(time.perf_counter() - encode_start, 1e-9)

        decode_start = time.perf_counter()
        for ids in encoded_batch:
            _ = self.decode(ids)
        decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)

        model_path = self.config.model_path()
        model_size = model_path.stat().st_size if model_path.is_file() else 0

        return TokenizerStats(
            vocab_size=self.vocab_size,
            average_token_length=average_token_length(all_pieces),
            compression_ratio=(total_chars / total_tokens if total_tokens else 0.0),
            unknown_token_frequency=(unk_count / total_tokens if total_tokens else 0.0),
            encoding_tokens_per_second=total_tokens / encode_elapsed,
            decoding_tokens_per_second=total_tokens / decode_elapsed,
            model_file_size_bytes=model_size,
        )

    def load_metadata(self) -> dict[str, Any]:
        """Load tokenizer metadata JSON if present."""
        path = self.config.metadata_path()
        if not path.is_file():
            return {}
        return read_json(path)

    def _export_vocab(self, path: Path) -> None:
        lines = []
        for token_id in range(self.vocab_size):
            piece = self.id_to_piece(token_id)
            score = self.processor.get_score(token_id)
            lines.append(f"{piece}\t{score}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _require_encoder(self) -> TokenizerEncoder:
        if self._encoder is None:
            raise RuntimeError("Tokenizer model is not loaded. Call load() or train().")
        return self._encoder

    def _require_decoder(self) -> TokenizerDecoder:
        if self._decoder is None:
            raise RuntimeError("Tokenizer model is not loaded. Call load() or train().")
        return self._decoder


# Convenience alias used in docs / imports.
SentencePieceTokenizer = OdysseySentencePieceTokenizer
