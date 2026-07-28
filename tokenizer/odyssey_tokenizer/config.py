"""Configuration for the Odyssey BPE tokenizer library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from odyssey.config import REPO_ROOT

DEFAULT_BPE_CONFIG_PATH = REPO_ROOT / "configs" / "tokenizer.yaml"

DEFAULT_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<think>",
]


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
    seed: int = 42
    max_lines: int = 0
    progress_every: int = 500


@dataclass(slots=True)
class PathConfig:
    model_dir: str = "assets/tokenizer/bpe/odyssey.model"
    merges_file: str = "assets/tokenizer/bpe/odyssey.model/merges.txt"
    vocab_file: str = "assets/tokenizer/bpe/odyssey.model/vocab.json"
    metadata_file: str = "assets/tokenizer/bpe/odyssey.model/metadata.json"
    visualization_dir: str = "assets/tokenizer/bpe"


@dataclass(slots=True)
class BPEConfig:
    """Typed view over ``configs/tokenizer.yaml`` for the BPE backend."""

    algorithm: str = "bpe"
    vocab_size: int = 32000
    min_frequency: int = 2
    byte_level: bool = True
    lowercase: bool = False
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    special_tokens: list[str] = field(
        default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS)
    )
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def resolve(self, relative: str | Path) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path

    def model_dir_path(self) -> Path:
        return self.resolve(self.paths.model_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "byte_level": self.byte_level,
            "lowercase": self.lowercase,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "special_tokens": list(self.special_tokens),
            "normalization": {
                "form": self.normalization.form,
                "collapse_whitespace": self.normalization.collapse_whitespace,
                "preserve_newlines": self.normalization.preserve_newlines,
                "strip": self.normalization.strip,
            },
            "training": {
                "input_sentence_size": self.training.input_sentence_size,
                "shuffle_input_sentence": self.training.shuffle_input_sentence,
                "seed": self.training.seed,
                "max_lines": self.training.max_lines,
                "progress_every": self.training.progress_every,
            },
            "paths": {
                "model_dir": self.paths.model_dir,
                "merges_file": self.paths.merges_file,
                "vocab_file": self.paths.vocab_file,
                "metadata_file": self.paths.metadata_file,
                "visualization_dir": self.paths.visualization_dir,
            },
        }


def load_bpe_config(path: Path | str | None = None) -> BPEConfig:
    """Load BPE tokenizer configuration from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_BPE_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Tokenizer config root must be a mapping")

    norm_raw = raw.get("normalization", {}) or {}
    train_raw = raw.get("training", {}) or {}
    paths_raw = raw.get("paths", {}) or {}
    specials = raw.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
    if not isinstance(specials, list) or not all(
        isinstance(item, str) for item in specials
    ):
        raise ValueError("special_tokens must be a list of strings")

    # Ensure core controls exist and appear first in a stable order.
    ordered: list[str] = []
    for token in DEFAULT_SPECIAL_TOKENS:
        if token in specials and token not in ordered:
            ordered.append(token)
    for token in specials:
        if token not in ordered:
            ordered.append(token)

    config = BPEConfig(
        algorithm=str(raw.get("algorithm", "bpe")).lower(),
        vocab_size=int(raw.get("vocab_size", 32000)),
        min_frequency=int(raw.get("min_frequency", 2)),
        byte_level=bool(raw.get("byte_level", True)),
        lowercase=bool(raw.get("lowercase", False)),
        pad_id=int(raw.get("pad_id", 0)),
        bos_id=int(raw.get("bos_id", 1)),
        eos_id=int(raw.get("eos_id", 2)),
        unk_id=int(raw.get("unk_id", 3)),
        special_tokens=ordered,
        normalization=NormalizationConfig(
            form=str(norm_raw.get("form", "NFKC")),
            collapse_whitespace=bool(norm_raw.get("collapse_whitespace", True)),
            preserve_newlines=bool(norm_raw.get("preserve_newlines", True)),
            strip=bool(norm_raw.get("strip", True)),
        ),
        training=TrainingConfig(
            input_sentence_size=int(train_raw.get("input_sentence_size", 0)),
            shuffle_input_sentence=bool(train_raw.get("shuffle_input_sentence", True)),
            seed=int(train_raw.get("seed", 42)),
            max_lines=int(train_raw.get("max_lines", 0)),
            progress_every=int(train_raw.get("progress_every", 500)),
        ),
        paths=PathConfig(
            model_dir=str(
                paths_raw.get("model_dir", "assets/tokenizer/bpe/odyssey.model")
            ),
            merges_file=str(
                paths_raw.get(
                    "merges_file", "assets/tokenizer/bpe/odyssey.model/merges.txt"
                )
            ),
            vocab_file=str(
                paths_raw.get(
                    "vocab_file", "assets/tokenizer/bpe/odyssey.model/vocab.json"
                )
            ),
            metadata_file=str(
                paths_raw.get(
                    "metadata_file", "assets/tokenizer/bpe/odyssey.model/metadata.json"
                )
            ),
            visualization_dir=str(
                paths_raw.get("visualization_dir", "assets/tokenizer/bpe")
            ),
        ),
    )

    if config.vocab_size < len(config.special_tokens) + 256:
        raise ValueError(
            "vocab_size must be >= len(special_tokens) + 256 for byte-level BPE"
        )
    if config.min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    return config
