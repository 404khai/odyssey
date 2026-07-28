"""Tokenizer configuration loading and validation.

Why this exists:
    Tokenizer hyperparameters are part of the model contract. Loading them from
    YAML keeps train/serve settings reproducible and avoids hardcoded IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from odyssey.config import REPO_ROOT

DEFAULT_TOKENIZER_CONFIG_PATH = REPO_ROOT / "configs" / "tokenizer_sentencepiece.yaml"


@dataclass(slots=True)
class NormalizationConfig:
    form: str = "NFKC"
    collapse_whitespace: bool = True
    preserve_newlines: bool = True
    strip: bool = True


@dataclass(slots=True)
class TrainingConfig:
    input_sentence_size: int = 0
    shuffle_input_sentence: bool = True
    num_threads: int = 4
    max_sentence_length: int = 8192
    hard_vocab_limit: bool = False


@dataclass(slots=True)
class PathConfig:
    model_prefix: str = "assets/tokenizer/odyssey"
    model_file: str = "assets/tokenizer/odyssey.model"
    vocab_file: str = "assets/tokenizer/odyssey.vocab"
    metadata_file: str = "assets/tokenizer/metadata.json"


@dataclass(slots=True)
class TokenizerConfig:
    """Strongly typed view over ``configs/tokenizer.yaml``."""

    vocab_size: int = 32000
    character_coverage: float = 0.9995
    model_type: str = "unigram"
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    user_defined_symbols: list[str] = field(
        default_factory=lambda: [
            "<mask>",
            "<system>",
            "<user>",
            "<assistant>",
            "<tool>",
            "<think>",
        ]
    )
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    special_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
            "mask": "<mask>",
            "system": "<system>",
            "user": "<user>",
            "assistant": "<assistant>",
            "tool": "<tool>",
            "think": "<think>",
        }
    )

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a path relative to the repository root when needed."""
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path

    def model_prefix_path(self) -> Path:
        return self.resolve(self.paths.model_prefix)

    def model_path(self) -> Path:
        return self.resolve(self.paths.model_file)

    def vocab_path(self) -> Path:
        return self.resolve(self.paths.vocab_file)

    def metadata_path(self) -> Path:
        return self.resolve(self.paths.metadata_file)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "user_defined_symbols": list(self.user_defined_symbols),
            "normalization": {
                "form": self.normalization.form,
                "collapse_whitespace": self.normalization.collapse_whitespace,
                "preserve_newlines": self.normalization.preserve_newlines,
                "strip": self.normalization.strip,
            },
            "training": {
                "input_sentence_size": self.training.input_sentence_size,
                "shuffle_input_sentence": self.training.shuffle_input_sentence,
                "num_threads": self.training.num_threads,
                "max_sentence_length": self.training.max_sentence_length,
                "hard_vocab_limit": self.training.hard_vocab_limit,
            },
            "paths": {
                "model_prefix": self.paths.model_prefix,
                "model_file": self.paths.model_file,
                "vocab_file": self.paths.vocab_file,
                "metadata_file": self.paths.metadata_file,
            },
            "special_tokens": dict(self.special_tokens),
        }


def _require_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Config key '{key}' must be int, got {type(value).__name__}")
    return value


def load_tokenizer_config(path: Path | str | None = None) -> TokenizerConfig:
    """Load and validate tokenizer configuration from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_TOKENIZER_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Tokenizer config root must be a mapping")

    norm_raw = raw.get("normalization", {}) or {}
    train_raw = raw.get("training", {}) or {}
    paths_raw = raw.get("paths", {}) or {}
    special_raw = raw.get("special_tokens", {}) or {}
    symbols = raw.get(
        "user_defined_symbols", ["<mask>", "<system>", "<user>", "<assistant>"]
    )

    if not isinstance(symbols, list) or not all(
        isinstance(item, str) for item in symbols
    ):
        raise ValueError("user_defined_symbols must be a list of strings")

    model_type = str(raw.get("model_type", "unigram")).lower()
    if model_type not in {"unigram", "bpe"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    config = TokenizerConfig(
        vocab_size=_require_int(raw, "vocab_size", 32000),
        character_coverage=float(raw.get("character_coverage", 0.9995)),
        model_type=model_type,
        pad_id=_require_int(raw, "pad_id", 0),
        bos_id=_require_int(raw, "bos_id", 1),
        eos_id=_require_int(raw, "eos_id", 2),
        unk_id=_require_int(raw, "unk_id", 3),
        user_defined_symbols=list(symbols),
        normalization=NormalizationConfig(
            form=str(norm_raw.get("form", "NFKC")),
            collapse_whitespace=bool(norm_raw.get("collapse_whitespace", True)),
            preserve_newlines=bool(norm_raw.get("preserve_newlines", True)),
            strip=bool(norm_raw.get("strip", True)),
        ),
        training=TrainingConfig(
            input_sentence_size=int(train_raw.get("input_sentence_size", 0)),
            shuffle_input_sentence=bool(train_raw.get("shuffle_input_sentence", True)),
            num_threads=int(train_raw.get("num_threads", 4)),
            max_sentence_length=int(train_raw.get("max_sentence_length", 8192)),
            hard_vocab_limit=bool(train_raw.get("hard_vocab_limit", False)),
        ),
        paths=PathConfig(
            model_prefix=str(paths_raw.get("model_prefix", "assets/tokenizer/odyssey")),
            model_file=str(
                paths_raw.get("model_file", "assets/tokenizer/odyssey.model")
            ),
            vocab_file=str(
                paths_raw.get("vocab_file", "assets/tokenizer/odyssey.vocab")
            ),
            metadata_file=str(
                paths_raw.get("metadata_file", "assets/tokenizer/metadata.json")
            ),
        ),
        special_tokens={
            "pad": str(special_raw.get("pad", "<pad>")),
            "bos": str(special_raw.get("bos", "<bos>")),
            "eos": str(special_raw.get("eos", "<eos>")),
            "unk": str(special_raw.get("unk", "<unk>")),
            "mask": str(special_raw.get("mask", "<mask>")),
            "system": str(special_raw.get("system", "<system>")),
            "user": str(special_raw.get("user", "<user>")),
            "assistant": str(special_raw.get("assistant", "<assistant>")),
            "tool": str(special_raw.get("tool", "<tool>")),
            "think": str(special_raw.get("think", "<think>")),
        },
    )

    if config.vocab_size < 16:
        raise ValueError("vocab_size must be >= 16")

    ids = [config.pad_id, config.bos_id, config.eos_id, config.unk_id]
    if len(set(ids)) != 4:
        raise ValueError("pad_id, bos_id, eos_id, and unk_id must be unique")

    return config
