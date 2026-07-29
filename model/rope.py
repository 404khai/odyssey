"""LLaMA-style Rotary Positional Embeddings for Odyssey.

Canonical training-side RoPE. Must stay numerically identical to
``phalanx::layers::Rope`` (see ``scripts/validate_rope.py``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.config import RopeConfig
from model.rope_cache import RopeCacheManager
from model.rope_math import rotate_adjacent_pairs


class OdysseyRoPE(nn.Module):
    """Apply rotary embeddings to query / key tensors.

    Accepted layouts (last dim = ``head_dim``):

    - ``(batch, seq, heads, head_dim)`` — Odyssey training default
    - ``(seq, heads, head_dim)`` — Phalanx multi-head layout
    - ``(seq, head_dim)`` — single-head / packed

    Shape is preserved. Values (V) are never rotated.
    """

    def __init__(self, config: RopeConfig) -> None:
        super().__init__()
        self.config = config
        self._manager = RopeCacheManager(
            rotary_dim=config.rotary_dim,
            theta=config.theta,
            scale=config.scaling_factor if config.scaling == "linear" else 1.0,
            initial_max_position=config.max_position_embeddings,
            device=config.device,
            dtype=config.torch_dtype,
        )

    @classmethod
    def from_config(cls, config: RopeConfig) -> OdysseyRoPE:
        return cls(config)

    @property
    def cache_max_position(self) -> int:
        return self._manager.cache.max_position

    def cache_memory_bytes(self) -> int:
        return self._manager.cache.memory_bytes()

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Rotate ``x`` starting at absolute position ``position_offset``."""
        self._validate_input(x)
        if position_offset < 0:
            raise ValueError("position_offset must be >= 0")

        seq = self._seq_len(x)
        if seq == 0:
            return x

        required = position_offset + seq
        cache = self._manager.get(required)
        positions = torch.arange(
            position_offset,
            position_offset + seq,
            device=x.device,
            dtype=torch.long,
        )
        cos = cache.cos[positions]  # (seq, n_pairs)
        sin = cache.sin[positions]

        # Align cos/sin to x layout: insert head axis when needed.
        cos_b, sin_b = self._broadcast_cis(cos, sin, x.ndim)
        # Compute in float32 for numerical parity with Phalanx, cast back.
        x_f = x.float()
        out = rotate_adjacent_pairs(x_f, cos_b.float(), sin_b.float())
        return out.to(dtype=x.dtype)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K; leave V to the caller untouched."""
        return self.forward(q, position_offset=position_offset), self.forward(
            k, position_offset=position_offset
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim not in (2, 3, 4):
            raise ValueError(
                "expected rank 2/3/4 with last dim = head_dim, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.config.head_dim:
            raise ValueError(
                f"last dim {x.shape[-1]} != configured head_dim {self.config.head_dim}"
            )
        if self.config.rotary_dim > self.config.head_dim:
            raise ValueError("rotary_dim exceeds head_dim")

    @staticmethod
    def _seq_len(x: torch.Tensor) -> int:
        if x.ndim == 4:
            return int(x.shape[1])
        return int(x.shape[0])

    @staticmethod
    def _broadcast_cis(
        cos: torch.Tensor, sin: torch.Tensor, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand ``(seq, n_pairs)`` for broadcasting onto ``x`` pair dims."""
        if rank == 2:
            # x: (seq, head_dim) — cos already (seq, n_pairs)
            return cos, sin
        if rank == 3:
            # x: (seq, heads, head_dim) → cos (seq, 1, n_pairs)
            return cos.unsqueeze(1), sin.unsqueeze(1)
        # rank 4: (batch, seq, heads, head_dim) → (1, seq, 1, n_pairs)
        return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)

    def inspect(self) -> dict[str, Any]:
        cache = self._manager.cache
        return {
            "theta": self.config.theta,
            "rotary_dim": self.config.rotary_dim,
            "head_dim": self.config.head_dim,
            "scaling": self.config.scaling,
            "scaling_factor": self.config.scaling_factor,
            "max_position_embeddings": self.config.max_position_embeddings,
            "cache_max_position": cache.max_position,
            "cache_memory_bytes": cache.memory_bytes(),
            "device": str(cache.device),
            "dtype": str(cache.dtype).removeprefix("torch."),
        }
