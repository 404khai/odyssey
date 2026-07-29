"""Cosine / sine cache for Rotary Positional Embeddings.

Layout (matches Phalanx Runtime ``layers::Rope``):

    cos[pos * n_pairs + pair]
    sin[pos * n_pairs + pair]

Logical view exposed to PyTorch: ``(max_position, n_pairs)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.rope_math import apply_linear_position_scale, inverse_frequencies


@dataclass(slots=True)
class RopeCache:
    """Precomputed cos/sin tables for absolute positions ``0 .. max_position-1``."""

    cos: torch.Tensor  # (max_position, n_pairs)
    sin: torch.Tensor  # (max_position, n_pairs)
    inv_freq: torch.Tensor  # (n_pairs,)
    rotary_dim: int
    theta: float
    scale: float
    max_position: int

    @property
    def n_pairs(self) -> int:
        return self.rotary_dim // 2

    @property
    def device(self) -> torch.device:
        return self.cos.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cos.dtype

    def memory_bytes(self) -> int:
        return int(self.cos.nbytes + self.sin.nbytes + self.inv_freq.nbytes)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> RopeCache:
        return RopeCache(
            cos=self.cos.to(device=device, dtype=dtype),
            sin=self.sin.to(device=device, dtype=dtype),
            inv_freq=self.inv_freq.to(device=device, dtype=dtype),
            rotary_dim=self.rotary_dim,
            theta=self.theta,
            scale=self.scale,
            max_position=self.max_position,
        )


def build_rope_cache(
    *,
    rotary_dim: int,
    max_position: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> RopeCache:
    """Build a dense cos/sin cache for ``max_position`` absolute indices."""
    if max_position < 1:
        raise ValueError("max_position must be >= 1")

    inv_freq = inverse_frequencies(
        rotary_dim, theta, dtype=torch.float32, device=device
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    positions = apply_linear_position_scale(positions, scale)
    # angles: (max_position, n_pairs)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)
    return RopeCache(
        cos=cos,
        sin=sin,
        inv_freq=inv_freq.to(dtype=dtype),
        rotary_dim=rotary_dim,
        theta=theta,
        scale=scale,
        max_position=max_position,
    )


class RopeCacheManager:
    """Lazy cos/sin cache with growth and device/dtype moves.

    The cache is never rebuilt when a request fits the existing table on the
    same device/dtype/hyperparameters.
    """

    def __init__(
        self,
        *,
        rotary_dim: int,
        theta: float = 10000.0,
        scale: float = 1.0,
        initial_max_position: int = 2048,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.rotary_dim = rotary_dim
        self.theta = theta
        self.scale = scale
        self.device = torch.device(device)
        self.dtype = dtype
        self._cache: RopeCache | None = None
        self._ensure(initial_max_position)

    @property
    def cache(self) -> RopeCache:
        if self._cache is None:
            raise RuntimeError("RoPE cache not initialized")
        return self._cache

    def _ensure(self, required_positions: int) -> RopeCache:
        need = max(1, required_positions)
        if (
            self._cache is not None
            and self._cache.max_position >= need
            and self._cache.device == self.device
            and self._cache.dtype == self.dtype
            and self._cache.rotary_dim == self.rotary_dim
            and self._cache.theta == self.theta
            and self._cache.scale == self.scale
        ):
            return self._cache

        # Grow geometrically to avoid frequent rebuilds.
        current = 0 if self._cache is None else self._cache.max_position
        target = max(need, current)
        if self._cache is not None and need > current:
            target = max(need, current * 2)

        self._cache = build_rope_cache(
            rotary_dim=self.rotary_dim,
            max_position=target,
            theta=self.theta,
            scale=self.scale,
            device=self.device,
            dtype=self.dtype,
        )
        return self._cache

    def get(self, required_positions: int) -> RopeCache:
        """Return a cache covering absolute positions ``0 .. required_positions-1``."""
        return self._ensure(required_positions)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> None:
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if self._cache is not None:
            self._cache = self._cache.to(device=self.device, dtype=self.dtype)
