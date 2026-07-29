"""Pre-norm residual pathway helpers for Odyssey.

Spec v1.0.0 freezes residual ordering as:

    x → RMSNorm → sub-layer → residual add → output

Never post-norm. Residuals create gradient highways: the Jacobian of an
identity skip is 1, so deep stacks remain trainable even when the sub-layer
Jacobian is small.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def residual_add(x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
    """Compute ``x + sublayer_out`` with strict shape validation.

    Shapes must match exactly. Broadcasting is intentionally rejected so
    silent shape bugs cannot hide inside a Transformer block.
    """
    if x.shape != sublayer_out.shape:
        raise ValueError(
            f"residual shapes must match: x={tuple(x.shape)} vs "
            f"sublayer={tuple(sublayer_out.shape)}"
        )
    if x.dtype != sublayer_out.dtype:
        raise ValueError(
            f"residual dtypes must match: x={x.dtype} vs sublayer={sublayer_out.dtype}"
        )
    return x + sublayer_out


def pre_norm_residual(
    x: torch.Tensor,
    *,
    norm: Callable[[torch.Tensor], torch.Tensor],
    sublayer: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply the frozen Spec pre-norm pattern: ``x + sublayer(norm(x))``."""
    return residual_add(x, sublayer(norm(x)))


def describe_residual_flow(
    *,
    hidden_size: int,
    batch: int = 1,
    seq_len: int = 1,
) -> dict[str, Any]:
    """Describe tensor flow through a pre-norm residual block (for inspectors)."""
    shape = (batch, seq_len, hidden_size)
    return {
        "ordering": "pre-norm",
        "steps": [
            {"stage": "input", "shape": shape},
            {"stage": "rmsnorm", "shape": shape},
            {"stage": "sublayer", "shape": shape},
            {"stage": "residual_add", "op": "x + f(norm(x))", "shape": shape},
            {"stage": "output", "shape": shape},
        ],
        "notes": (
            "Identity skip preserves gradient magnitude; Spec forbids post-norm."
        ),
    }
