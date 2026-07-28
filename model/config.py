"""Configuration for Odyssey token embeddings.

Why this exists:
    Embedding hyperparameters (vocab size, hidden size, init, device, dtype)
    must stay out of scattered script constants so experiments are reproducible
    and Phalanx Runtime can mirror the same shapes later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from odyssey.config import REPO_ROOT

DEFAULT_EMBEDDING_CONFIG_PATH = REPO_ROOT / "configs" / "embedding.yaml"

InitStrategy = Literal[
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
]

VALID_INIT_STRATEGIES: tuple[str, ...] = (
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
)

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(slots=True)
class EmbeddingConfig:
    """Typed configuration for :class:`~model.embeddings.OdysseyEmbedding`."""

    vocab_size: int = 32000
    hidden_size: int = 768
    padding_idx: int | None = 0
    init_strategy: InitStrategy = "xavier_uniform"
    init_std: float = 0.02
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.init_strategy not in VALID_INIT_STRATEGIES:
            raise ValueError(
                f"init_strategy must be one of {VALID_INIT_STRATEGIES}, "
                f"got {self.init_strategy!r}"
            )
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if self.dtype not in DTYPE_MAP:
            raise ValueError(
                f"dtype must be one of {tuple(DTYPE_MAP)}, got {self.dtype!r}"
            )
        if self.padding_idx is not None:
            if self.padding_idx < 0 or self.padding_idx >= self.vocab_size:
                raise ValueError(
                    f"padding_idx {self.padding_idx} out of range for "
                    f"vocab_size {self.vocab_size}"
                )

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def parameter_count(self) -> int:
        """Number of embedding parameters (vocab × hidden)."""
        return self.vocab_size * self.hidden_size

    def memory_bytes(self, dtype: torch.dtype | None = None) -> int:
        """Bytes for the embedding matrix at the given (or config) dtype."""
        resolved = dtype if dtype is not None else self.torch_dtype
        return self.parameter_count * resolved.itemsize

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingConfig:
        padding = data.get("padding_idx", 0)
        if padding is False or padding == "none":
            padding = None
        return cls(
            vocab_size=int(data.get("vocab_size", 32000)),
            hidden_size=int(data.get("hidden_size", 768)),
            padding_idx=None if padding is None else int(padding),
            init_strategy=str(data.get("init_strategy", "xavier_uniform")),  # type: ignore[arg-type]
            init_std=float(data.get("init_std", 0.02)),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
        )


def load_embedding_config(path: Path | str | None = None) -> EmbeddingConfig:
    """Load embedding configuration from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_EMBEDDING_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Embedding config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Embedding config root must be a mapping")

    # Support nested `embedding:` or flat keys.
    section = raw.get("embedding", raw)
    if not isinstance(section, dict):
        raise ValueError("embedding section must be a mapping")
    return EmbeddingConfig.from_dict(section)
