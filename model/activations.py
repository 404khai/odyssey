"""Activation functions for Odyssey feed-forward networks.

SiLU / Swish is preferred over GELU in LLaMA-style models: it is smoother
than ReLU, cheaper than GELU's erf, and pairs naturally with GLU gating
(Shazeer, *GLU Variants Improve Transformer*).
"""

from __future__ import annotations

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid via ``torch.sigmoid`` (float32 promote)."""
    return torch.sigmoid(x.float()).to(dtype=x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish: ``x · σ(x)``.

    Uses the same ``1 / (1 + e^{-x})`` form as Phalanx ``layers::SwiGlu``
    (not ``torch.nn.functional.silu``) for Spec parity.
    """
    x_f = x.float()
    return (x_f * (1.0 / (1.0 + (-x_f).exp()))).to(dtype=x.dtype)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Alias for :func:`silu` (Shazeer / Spec naming)."""
    return silu(x)
