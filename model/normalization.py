"""Shared normalization helpers for Odyssey.

RMSNorm is cheaper than LayerNorm because it skips mean centering.
These helpers keep the float32 sum-of-squares path explicit so fp16/bf16
activations stay numerically stable and match Phalanx Runtime.
"""

from __future__ import annotations

import torch


def rms(x: torch.Tensor, eps: float, *, dim: int = -1) -> torch.Tensor:
    """Root-mean-square along ``dim`` with float64 sum, float32 result.

    Computes ``sqrt(mean(x²) + eps)`` and broadcasts the result so callers
    can divide ``x`` elementwise without reshaping.

    Float64 accumulation keeps Odyssey aligned with Phalanx ``RmsNorm``
    (Rule 6 / Principle 8) for large ``hidden_size`` reductions.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    x_f = x.double()
    mean_sq = x_f.pow(2).mean(dim=dim, keepdim=True)
    return torch.sqrt(mean_sq + eps).float()


def rms_normalize(x: torch.Tensor, eps: float, *, dim: int = -1) -> torch.Tensor:
    """Scale ``x`` by its RMS (no affine γ). Output dtype matches ``x``."""
    scale = rms(x, eps, dim=dim)
    return (x.float() / scale).to(dtype=x.dtype)


def apply_scale(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Elementwise affine scale ``γ ⊙ x`` along the last dimension."""
    if gamma.ndim != 1:
        raise ValueError(f"gamma must be rank-1, got shape {tuple(gamma.shape)}")
    if x.shape[-1] != gamma.shape[0]:
        raise ValueError(f"last dim {x.shape[-1]} != gamma length {gamma.shape[0]}")
    return x * gamma.to(dtype=x.dtype, device=x.device)
