"""Rotary embedding mathematics (LLaMA / Phalanx-compatible).

Frequency schedule
------------------
For rotary width ``d_r`` (even) and base ``θ`` (default 10000):

    inv_freq[i] = θ ^ (-2i / d_r)     for i = 0 .. d_r/2 - 1

Why θ = 10000: inherited from the original Transformer sinusoids and
RoFormer / LLaMA; it spaces wavelengths from ~2π to cover short and long
offsets without trainable parameters.

Adjacent-pair rotation
----------------------
For each pair ``(x0, x1)`` at absolute position ``m`` (after optional linear
scale ``m' = m / factor``):

    x0' = x0 * cos(m'·ω) - x1 * sin(m'·ω)
    x1' = x0 * sin(m'·ω) + x1 * cos(m'·ω)

This is the real form of multiplying ``(x0 + i x1)`` by ``e^{i m' ω}``.
Rotation preserves L2 norm of each pair (and thus the full rotary subspace).

V is never rotated: values carry content, not relative-position phases in the
RoPE attention derivation.
"""

from __future__ import annotations

import math

import torch


def inverse_frequencies(
    rotary_dim: int,
    theta: float = 10000.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``inv_freq`` of shape ``(rotary_dim // 2,)``."""
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even and > 0, got {rotary_dim}")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError(f"theta must be finite and > 0, got {theta}")

    n_pairs = rotary_dim // 2
    # Match Phalanx: exponent = (2 * i) / rotary_dim in f32-style arithmetic.
    indices = torch.arange(n_pairs, device=device, dtype=torch.float32)
    inv_freq = theta ** (-(2.0 * indices) / float(rotary_dim))
    return inv_freq.to(dtype=dtype)


def rotate_adjacent_pairs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the leading even-width slice of ``x`` with broadcastable cos/sin.

    Args:
        x: ``(..., head_dim)`` — only the first ``2 * cos.shape[-1]`` dims rotate.
        cos: ``(..., n_pairs)`` broadcastable to the pair axis.
        sin: same shape as ``cos``.

    Returns:
        Tensor with the same shape as ``x``.
    """
    if x.shape[-1] < 2:
        return x

    n_pairs = cos.shape[-1]
    rotary_dim = n_pairs * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary width {rotary_dim} exceeds head_dim {x.shape[-1]}")

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Adjacent pairs: (x0, x1, x2, x3, ...) → rotate (x0,x1), (x2,x3), ...
    x0 = x_rot[..., 0::2]
    x1 = x_rot[..., 1::2]
    # Broadcast cos/sin over batch/head dims as needed.
    while cos.ndim < x0.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    rotated = torch.stack((out0, out1), dim=-1).flatten(-2)

    if x_pass.numel() == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


def apply_linear_position_scale(positions: torch.Tensor, factor: float) -> torch.Tensor:
    """Return ``m' = m / factor`` for linear RoPE scaling (factor=1 → identity)."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"scaling factor must be finite and > 0, got {factor}")
    return positions.to(dtype=torch.float32) / float(factor)
