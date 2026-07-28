"""Token embedding layer for Odyssey.

Maps integer token IDs to dense vectors via a trainable lookup table
``E ∈ ℝ^(V × D)``. Uses :class:`torch.nn.Embedding` for correctness; custom
CUDA kernels are out of scope for Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from model.config import EmbeddingConfig
from model.initialization import initialize_embedding


@dataclass(frozen=True, slots=True)
class EmbeddingInspection:
    """Snapshot of embedding layer metadata for logging and CLI inspect."""

    vocab_size: int
    hidden_size: int
    embedding_shape: tuple[int, int]
    parameter_count: int
    memory_bytes: int
    trainable_parameters: int
    padding_idx: int | None
    init_strategy: str
    device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "embedding_shape": list(self.embedding_shape),
            "parameter_count": self.parameter_count,
            "memory_bytes": self.memory_bytes,
            "trainable_parameters": self.trainable_parameters,
            "padding_idx": self.padding_idx,
            "init_strategy": self.init_strategy,
            "device": self.device,
            "dtype": self.dtype,
        }

    def format(self) -> str:
        mb = self.memory_bytes / (1024 * 1024)
        return (
            "OdysseyEmbedding\n"
            f"  vocab_size:           {self.vocab_size:,}\n"
            f"  hidden_size:          {self.hidden_size:,}\n"
            f"  embedding_shape:      {self.embedding_shape}\n"
            f"  parameter_count:      {self.parameter_count:,}\n"
            f"  trainable_parameters:  {self.trainable_parameters:,}\n"
            f"  memory (weights):     {self.memory_bytes:,} bytes ({mb:.3f} MiB)\n"
            f"  padding_idx:          {self.padding_idx}\n"
            f"  init_strategy:        {self.init_strategy}\n"
            f"  device / dtype:       {self.device} / {self.dtype}\n"
        )


class OdysseyEmbedding(nn.Module):
    """Configurable token embedding lookup.

    Input shape: ``(batch, sequence)`` of integer token IDs.
    Output shape: ``(batch, sequence, hidden_size)``.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.padding_idx,
            device=config.torch_device,
            dtype=config.torch_dtype,
        )
        initialize_embedding(
            self.embedding.weight,
            config.init_strategy,
            std=config.init_std,
            padding_idx=config.padding_idx,
        )

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> OdysseyEmbedding:
        return cls(config)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def parameter_count(self) -> int:
        return int(self.embedding.weight.numel())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def memory_bytes(self) -> int:
        return int(self.embedding.weight.nbytes)

    def validate_input(self, token_ids: torch.Tensor) -> None:
        """Validate token ID tensor rank, dtype family, and ID range."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (batch, sequence), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (
            torch.int32,
            torch.int64,
            torch.long,
        ):
            raise TypeError(f"token_ids must be integer dtype, got {token_ids.dtype}")
        if token_ids.numel() == 0:
            return
        min_id = int(token_ids.min().item())
        max_id = int(token_ids.max().item())
        if min_id < 0:
            raise ValueError(f"token id {min_id} is negative")
        if max_id >= self.config.vocab_size:
            raise ValueError(
                f"token id {max_id} >= vocab_size {self.config.vocab_size}"
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for ``token_ids``.

        Args:
            token_ids: Long/int tensor of shape ``(batch, sequence)``.

        Returns:
            Float tensor of shape ``(batch, sequence, hidden_size)``.
        """
        self.validate_input(token_ids)
        output: torch.Tensor = self.embedding(token_ids)
        expected = (
            token_ids.shape[0],
            token_ids.shape[1],
            self.config.hidden_size,
        )
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"embedding output shape {tuple(output.shape)} != expected {expected}"
            )
        return output

    def inspect(self) -> EmbeddingInspection:
        return EmbeddingInspection(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            embedding_shape=tuple(self.embedding.weight.shape),  # type: ignore[arg-type]
            parameter_count=self.parameter_count(),
            memory_bytes=self.memory_bytes(),
            trainable_parameters=self.trainable_parameter_count(),
            padding_idx=self.config.padding_idx,
            init_strategy=self.config.init_strategy,
            device=str(self.embedding.weight.device),
            dtype=str(self.embedding.weight.dtype).removeprefix("torch."),
        )
