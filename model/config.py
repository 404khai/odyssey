"""Configuration for Odyssey model components (embeddings + RoPE + RMSNorm + Attention).

Why this exists:
    Embedding, RoPE, and normalization hyperparameters must stay out of
    scattered script constants so experiments are reproducible and Phalanx
    Runtime can mirror the same shapes / θ / ε / scaling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from odyssey.config import REPO_ROOT

DEFAULT_EMBEDDING_CONFIG_PATH = REPO_ROOT / "configs" / "embedding.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"

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

RopeScalingType = Literal["none", "linear"]
NormType = Literal["rmsnorm"]
FeedForwardType = Literal["swiglu"]
ActivationType = Literal["silu", "swish", "swiglu"]
AttentionType = Literal["gqa", "mha", "mqa"]

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
        return self.vocab_size * self.hidden_size

    def memory_bytes(self, dtype: torch.dtype | None = None) -> int:
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


@dataclass(slots=True)
class RopeConfig:
    """RoPE hyperparameters — must match Phalanx ``RopeConfig`` semantics."""

    theta: float = 10000.0
    head_dim: int = 64
    rotary_dim: int = 64
    max_position_embeddings: int = 2048
    scaling: RopeScalingType = "none"
    scaling_factor: float = 1.0
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not (self.theta > 0):
            raise ValueError("theta must be > 0")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.rotary_dim < 2 or self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even and >= 2")
        if self.rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim ({self.rotary_dim}) exceeds head_dim ({self.head_dim})"
            )
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be >= 1")
        if self.scaling not in ("none", "linear"):
            raise ValueError(
                f"scaling must be 'none' or 'linear' (NTK/YaRN deferred), got {self.scaling!r}"
            )
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be > 0")
        if self.dtype not in DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(DTYPE_MAP)}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def effective_scale(self) -> float:
        return self.scaling_factor if self.scaling == "linear" else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RopeConfig:
        scaling = str(data.get("scaling", "none")).lower()
        if scaling in ("", "null", "none"):
            scaling = "none"
        head_dim = int(data.get("head_dim", 64))
        return cls(
            theta=float(data.get("theta", data.get("freq_base", 10000.0))),
            head_dim=head_dim,
            rotary_dim=int(
                data.get("rotary_dim", data.get("dimension_count", head_dim))
            ),
            max_position_embeddings=int(
                data.get(
                    "max_position_embeddings",
                    data.get("max_position", data.get("context_length", 2048)),
                )
            ),
            scaling=scaling,  # type: ignore[arg-type]
            scaling_factor=float(data.get("scaling_factor", data.get("factor", 1.0))),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
        )


@dataclass(slots=True)
class NormConfig:
    """RMSNorm hyperparameters — must match Phalanx ``RmsNorm`` / ``rms_norm_eps``."""

    type: NormType = "rmsnorm"
    hidden_size: int = 768
    epsilon: float = 1e-6
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.type != "rmsnorm":
            raise ValueError(
                f"norm type must be 'rmsnorm' (LayerNorm / post-norm forbidden), "
                f"got {self.type!r}"
            )
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.dtype not in DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(DTYPE_MAP)}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormConfig:
        return cls(
            type=str(data.get("type", "rmsnorm")).lower(),  # type: ignore[arg-type]
            hidden_size=int(data.get("hidden_size", 768)),
            epsilon=float(
                data.get("epsilon", data.get("eps", data.get("rms_norm_eps", 1e-6)))
            ),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
        )


@dataclass(slots=True)
class AttentionConfig:
    """Grouped-query / multi-head attention hyperparameters.

    Must match Phalanx ``AttentionConfig`` + ``layers::Attention`` semantics.
    Default Odyssey path is GQA (``num_kv_heads <= num_heads``).
    """

    num_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int = 64
    # 0 → auto ``num_heads * head_dim`` in ``__post_init__`` (keeps type ``int`` for mypy).
    hidden_size: int = 0
    dropout: float = 0.0
    causal: bool = True
    bias: bool = False
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.num_kv_heads < 1:
            raise ValueError("num_kv_heads must be >= 1")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.bias:
            raise ValueError("attention bias must be False (Odyssey Spec v1.0.0)")
        if self.dtype not in DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(DTYPE_MAP)}")
        expected_hidden = self.num_heads * self.head_dim
        if self.hidden_size == 0:
            self.hidden_size = expected_hidden
        elif self.hidden_size != expected_hidden:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_heads * head_dim ({expected_hidden})"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def query_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

    @property
    def is_mqa(self) -> bool:
        return self.num_kv_heads == 1 and self.num_heads > 1

    @property
    def is_mha(self) -> bool:
        return self.num_kv_heads == self.num_heads

    @property
    def attention_type(self) -> AttentionType:
        if self.is_mha:
            return "mha"
        if self.is_mqa:
            return "mqa"
        return "gqa"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttentionConfig:
        num_heads = int(data.get("num_heads", 12))
        head_dim = int(data.get("head_dim", 64))
        hidden = data.get("hidden_size")
        return cls(
            num_heads=num_heads,
            num_kv_heads=int(data.get("num_kv_heads", num_heads)),
            head_dim=head_dim,
            hidden_size=0 if hidden is None else int(hidden),
            dropout=float(data.get("dropout", 0.0)),
            causal=bool(data.get("causal", True)),
            bias=bool(data.get("bias", False)),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
        )


@dataclass(slots=True)
class FeedForwardConfig:
    """SwiGLU FFN hyperparameters — must match Phalanx ``SwiGlu``."""

    type: FeedForwardType = "swiglu"
    hidden_size: int = 768
    intermediate_size: int = 2048
    activation: ActivationType = "silu"
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.type != "swiglu":
            raise ValueError(
                f"feed_forward type must be 'swiglu' (Spec), got {self.type!r}"
            )
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.intermediate_size < 1:
            raise ValueError("intermediate_size must be >= 1")
        if self.activation not in ("silu", "swish", "swiglu"):
            raise ValueError(
                f"activation must be silu/swish/swiglu, got {self.activation!r}"
            )
        if self.dtype not in DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(DTYPE_MAP)}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def expansion_ratio(self) -> float:
        return self.intermediate_size / self.hidden_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedForwardConfig:
        return cls(
            type=str(data.get("type", "swiglu")).lower(),  # type: ignore[arg-type]
            hidden_size=int(data.get("hidden_size", 768)),
            intermediate_size=int(data.get("intermediate_size", 2048)),
            activation=str(data.get("activation", "silu")).lower(),  # type: ignore[arg-type]
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
        )


@dataclass(slots=True)
class ModelConfig:
    """Top-level model hyperparameters used by configs/model.yaml."""

    name: str = "odyssey-tiny"
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    context_length: int = 2048
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rope: RopeConfig = field(default_factory=RopeConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    feed_forward: FeedForwardConfig = field(default_factory=FeedForwardConfig)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.norm.hidden_size != self.hidden_size:
            self.norm = NormConfig(
                type=self.norm.type,
                hidden_size=self.hidden_size,
                epsilon=self.norm.epsilon,
                device=self.norm.device,
                dtype=self.norm.dtype,
            )

        # Keep nested attention aligned with top-level head layout.
        expected_head_dim = self.hidden_size // self.num_heads
        if (
            self.attention.num_heads != self.num_heads
            or self.attention.num_kv_heads != self.num_kv_heads
            or self.attention.head_dim != expected_head_dim
            or self.attention.hidden_size != self.hidden_size
        ):
            self.attention = AttentionConfig(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=expected_head_dim,
                hidden_size=self.hidden_size,
                dropout=self.attention.dropout,
                causal=self.attention.causal,
                bias=self.attention.bias,
                device=self.attention.device,
                dtype=self.attention.dtype,
            )
        if (
            self.feed_forward.hidden_size != self.hidden_size
            or self.feed_forward.intermediate_size != self.intermediate_size
        ):
            self.feed_forward = FeedForwardConfig(
                type=self.feed_forward.type,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=self.feed_forward.activation,
                device=self.feed_forward.device,
                dtype=self.feed_forward.dtype,
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "context_length": self.context_length,
            "embedding": self.embedding.to_dict(),
            "rope": self.rope.to_dict(),
            "norm": self.norm.to_dict(),
            "attention": self.attention.to_dict(),
            "feed_forward": self.feed_forward.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        emb_raw = data.get("embedding", {}) or {}
        rope_raw = dict(data.get("rope", {}) or {})
        norm_raw = dict(data.get("norm", {}) or {})
        ff_raw = dict(data.get("feed_forward", {}) or {})
        attn_raw = dict(data.get("attention", {}) or {})
        hidden = int(data.get("hidden_size", 768))
        intermediate = int(data.get("intermediate_size", 2048))
        heads = int(data.get("num_heads", 12))
        head_dim = hidden // heads
        rope_raw.setdefault("head_dim", head_dim)
        rope_raw.setdefault("rotary_dim", rope_raw.get("rotary_dim", head_dim))
        rope_raw.setdefault(
            "max_position_embeddings",
            int(data.get("context_length", 2048)),
        )
        norm_raw.setdefault("hidden_size", hidden)
        ff_raw.setdefault("hidden_size", hidden)
        ff_raw.setdefault("intermediate_size", intermediate)
        attn_raw.setdefault("num_heads", heads)
        attn_raw.setdefault("num_kv_heads", int(data.get("num_kv_heads", heads)))
        attn_raw.setdefault("head_dim", head_dim)
        attn_raw.setdefault("hidden_size", hidden)
        return cls(
            name=str(data.get("name", "odyssey-tiny")),
            vocab_size=int(data.get("vocab_size", data.get("vocabulary_size", 32000))),
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_layers=int(data.get("num_layers", 12)),
            num_heads=heads,
            num_kv_heads=int(data.get("num_kv_heads", heads)),
            context_length=int(data.get("context_length", 2048)),
            embedding=EmbeddingConfig.from_dict(emb_raw),
            rope=RopeConfig.from_dict(rope_raw),
            norm=NormConfig.from_dict(norm_raw),
            attention=AttentionConfig.from_dict(attn_raw),
            feed_forward=FeedForwardConfig.from_dict(ff_raw),
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

    section = raw.get("embedding", raw)
    if not isinstance(section, dict):
        raise ValueError("embedding section must be a mapping")
    return EmbeddingConfig.from_dict(section)


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    """Load model configuration (including RoPE) from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_MODEL_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Model config root must be a mapping")

    section = raw.get("model", raw)
    if not isinstance(section, dict):
        raise ValueError("model section must be a mapping")
    return ModelConfig.from_dict(section)


def load_rope_config(path: Path | str | None = None) -> RopeConfig:
    """Load RoPE config from ``configs/model.yaml``."""
    return load_model_config(path).rope


def load_norm_config(path: Path | str | None = None) -> NormConfig:
    """Load RMSNorm config from ``configs/model.yaml``."""
    return load_model_config(path).norm


def load_feed_forward_config(path: Path | str | None = None) -> FeedForwardConfig:
    """Load SwiGLU FFN config from ``configs/model.yaml``."""
    return load_model_config(path).feed_forward


def load_attention_config(path: Path | str | None = None) -> AttentionConfig:
    """Load attention / GQA config from ``configs/model.yaml``."""
    return load_model_config(path).attention
