"""Public OdysseyTokenizer API — reusable across training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from odyssey_tokenizer.config import BPEConfig, load_bpe_config
from odyssey_tokenizer.decoder import BPEDecoder
from odyssey_tokenizer.encoder import BPEEncoder
from odyssey_tokenizer.merges import MergeTable
from odyssey_tokenizer.normalizer import TextNormalizer
from odyssey_tokenizer.serialization import load_tokenizer_bundle, save_tokenizer_bundle
from odyssey_tokenizer.special_tokens import (
    build_special_tokens,
    core_special_id_map,
    describe_special_tokens,
)
from odyssey_tokenizer.statistics import (
    TokenizerStats,
    compression_summary,
    timed_call,
)
from odyssey_tokenizer.trainer import BPETrainer, TrainResult
from odyssey_tokenizer.vocabulary import Vocabulary


@dataclass(slots=True)
class InspectionResult:
    input_text: str
    normalized_text: str
    tokens: list[str]
    ids: list[int]
    decoded_text: str
    characters: int
    token_count: int
    compression_ratio: float
    compression_percent: float

    def render(self) -> str:
        ratio, percent = self.compression_ratio, self.compression_percent
        return "\n".join(
            [
                "Input",
                self.input_text,
                "",
                "Normalized",
                self.normalized_text,
                "",
                "Tokens",
                " ".join(self.tokens),
                "",
                "IDs",
                "[" + ", ".join(str(token_id) for token_id in self.ids) + "]",
                "",
                "Decoded Text",
                self.decoded_text,
                "",
                "Compression",
                f"Characters {self.characters}",
                f"Tokens {self.token_count}",
                f"Ratio {ratio:.3f} chars/token",
                f"Reduction {percent:.1f}%",
            ]
        )


class OdysseyTokenizer:
    """Owned Odyssey byte-level BPE tokenizer.

    Designed as a reusable library interface so Phalanx Runtime (and a future
    Rust port) can share identical encode/decode behavior with training:

        tokenizer = OdysseyTokenizer.load("path/to/odyssey.model")
        ids = tokenizer.encode(text)
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        merges: MergeTable,
        config: BPEConfig,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.merges = merges
        self.config = config
        self.metadata = metadata or {}
        self.normalizer = TextNormalizer(config)
        special_ids = core_special_id_map(config)
        self.pad_id = special_ids["pad"]
        self.bos_id = special_ids["bos"]
        self.eos_id = special_ids["eos"]
        self.unk_id = special_ids["unk"]
        special_bytes = {surface.encode("utf-8") for surface in config.special_tokens}
        self._encoder = BPEEncoder(
            vocabulary,
            merges,
            self.normalizer,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        self._decoder = BPEDecoder(
            vocabulary,
            special_token_bytes=special_bytes,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(
        cls,
        corpus_path: str | Path,
        config: BPEConfig | None = None,
        *,
        save_path: str | Path | None = None,
    ) -> tuple[OdysseyTokenizer, TrainResult]:
        """Train from a newline-delimited corpus and optionally save."""
        cfg = config or load_bpe_config()
        result = BPETrainer(cfg).train(corpus_path)
        tokenizer = cls(
            result.vocabulary,
            result.merges,
            cfg,
            metadata={
                "experiment": "ODY-0002",
                "training_seconds": result.training_seconds,
                "corpus_lines": result.corpus_lines,
                "corpus_bytes": result.corpus_bytes,
                "merge_count": result.merge_count,
            },
        )
        if save_path is not None:
            tokenizer.save(save_path)
        elif cfg.paths.model_dir:
            tokenizer.save(cfg.model_dir_path())
        return tokenizer, result

    @classmethod
    def load(cls, path: str | Path) -> OdysseyTokenizer:
        """Load a tokenizer model directory."""
        directory = Path(path)
        vocabulary, merges, config_payload, metadata = load_tokenizer_bundle(directory)
        config = _config_from_payload(config_payload)
        return cls(vocabulary, merges, config, metadata=metadata)

    def save(self, path: str | Path | None = None) -> Path:
        """Save vocabulary, merges, config, and metadata to a model directory."""
        directory = Path(path) if path is not None else self.config.model_dir_path()
        return save_tokenizer_bundle(
            directory,
            vocabulary=self.vocabulary,
            merges=self.merges,
            config=self.config,
            metadata=self.metadata,
        )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        return self._encoder.encode(
            text, add_bos=add_bos, add_eos=add_eos, normalize=normalize
        )

    def decode(self, ids: list[int], *, skip_special_ids: bool = False) -> str:
        return self._decoder.decode(ids, skip_special_ids=skip_special_ids)

    def encode_as_tokens(self, text: str, *, normalize: bool = True) -> list[str]:
        tokens = self._encoder.encode_as_tokens(text, normalize=normalize)
        return [token.decode("latin-1", errors="replace") for token in tokens]

    def inspect(self, text: str) -> InspectionResult:
        normalized = self.normalizer.normalize(text)
        tokens = self.encode_as_tokens(text)
        ids = self.encode(text)
        decoded = self.decode(ids)
        ratio, percent = compression_summary(len(normalized), len(ids))
        return InspectionResult(
            input_text=text,
            normalized_text=normalized,
            tokens=tokens,
            ids=ids,
            decoded_text=decoded,
            characters=len(normalized),
            token_count=len(ids),
            compression_ratio=ratio,
            compression_percent=percent,
        )

    def compute_stats(self, sample_texts: list[str]) -> TokenizerStats:
        if not sample_texts:
            raise ValueError("sample_texts must be non-empty")

        total_chars = 0
        total_tokens = 0
        unk_count = 0
        token_char_lens: list[int] = []

        encode_start_ids, encode_elapsed = timed_call(
            lambda: [self.encode(text) for text in sample_texts]
        )

        for text, ids in zip(sample_texts, encode_start_ids, strict=True):
            normalized = self.normalizer.normalize(text)
            total_chars += len(normalized)
            total_tokens += len(ids)
            unk_count += sum(1 for token_id in ids if token_id == self.unk_id)
            for token in self.encode_as_tokens(text):
                token_char_lens.append(len(token))

        _, decode_elapsed = timed_call(
            lambda: [self.decode(ids) for ids in encode_start_ids]
        )

        ratio, percent = compression_summary(total_chars, total_tokens)
        avg_len = (
            sum(token_char_lens) / len(token_char_lens) if token_char_lens else 0.0
        )
        return TokenizerStats(
            vocab_size=self.vocab_size,
            merge_count=len(self.merges),
            average_token_length=avg_len,
            compression_ratio=ratio,
            compression_percent=percent,
            unknown_token_frequency=(unk_count / total_tokens if total_tokens else 0.0),
            encoding_tokens_per_second=total_tokens / max(encode_elapsed, 1e-9),
            decoding_tokens_per_second=total_tokens / max(decode_elapsed, 1e-9),
            characters=total_chars,
            tokens=total_tokens,
        )

    def special_tokens_table(self) -> str:
        return describe_special_tokens(self.config)

    def has_reserved_tokens(self) -> bool:
        for token in build_special_tokens(self.config):
            if not self.vocabulary.contains(token.surface.encode("utf-8")):
                return False
        return True


def _config_from_payload(payload: dict[str, Any]) -> BPEConfig:
    """Rebuild BPEConfig from a serialized config.json payload."""
    from odyssey_tokenizer.config import NormalizationConfig, PathConfig, TrainingConfig

    norm = payload.get("normalization", {})
    train = payload.get("training", {})
    paths = payload.get("paths", {})
    return BPEConfig(
        algorithm=str(payload.get("algorithm", "bpe")),
        vocab_size=int(payload.get("vocab_size", 32000)),
        min_frequency=int(payload.get("min_frequency", 2)),
        byte_level=bool(payload.get("byte_level", True)),
        lowercase=bool(payload.get("lowercase", False)),
        pad_id=int(payload.get("pad_id", 0)),
        bos_id=int(payload.get("bos_id", 1)),
        eos_id=int(payload.get("eos_id", 2)),
        unk_id=int(payload.get("unk_id", 3)),
        special_tokens=list(payload.get("special_tokens", [])),
        normalization=NormalizationConfig(
            form=str(norm.get("form", "NFKC")),
            collapse_whitespace=bool(norm.get("collapse_whitespace", True)),
            preserve_newlines=bool(norm.get("preserve_newlines", True)),
            strip=bool(norm.get("strip", True)),
        ),
        training=TrainingConfig(
            input_sentence_size=int(train.get("input_sentence_size", 0)),
            shuffle_input_sentence=bool(train.get("shuffle_input_sentence", True)),
            seed=int(train.get("seed", 42)),
            max_lines=int(train.get("max_lines", 0)),
            progress_every=int(train.get("progress_every", 500)),
        ),
        paths=PathConfig(
            model_dir=str(paths.get("model_dir", "assets/tokenizer/bpe/odyssey.model")),
            merges_file=str(
                paths.get(
                    "merges_file", "assets/tokenizer/bpe/odyssey.model/merges.txt"
                )
            ),
            vocab_file=str(
                paths.get("vocab_file", "assets/tokenizer/bpe/odyssey.model/vocab.json")
            ),
            metadata_file=str(
                paths.get(
                    "metadata_file", "assets/tokenizer/bpe/odyssey.model/metadata.json"
                )
            ),
            visualization_dir=str(
                paths.get("visualization_dir", "assets/tokenizer/bpe")
            ),
        ),
    )
