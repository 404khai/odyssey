"""Numerically stable Softmax for attention weights.

Subtracting the per-row maximum before ``exp`` prevents overflow when scores
are large positive (common after unscaled or poorly scaled QKᵀ). Softmax is
translation-invariant, so the result is exact in exact arithmetic and far more
stable in float32.
"""

from __future__ import annotations

import torch


def stable_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along ``dim`` with max-subtraction.

    Computes in float32 for Spec / Phalanx parity, then casts back to the
    input dtype.
    """
    x = scores.float()
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    denom = exp.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return (exp / denom).to(dtype=scores.dtype)
