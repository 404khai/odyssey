"""Weight initialization strategies for Odyssey modules.

Why each strategy exists
------------------------
``normal``
    GPT-style default (often σ≈0.02). Simple, scale-agnostic; can explode or
    vanish if fan-in / fan-out grows without care.

``xavier_uniform`` / ``xavier_normal`` (Glorot)
    Keeps activation variance roughly stable through linear layers by scaling
    with fan-in and fan-out. Good default for embeddings and dense projections
    when using saturating or linear-ish activations.

``kaiming_uniform`` / ``kaiming_normal`` (He)
    Tuned for ReLU-family nonlinearities (half of units fire). Prefer for
    layers that feed ReLU/SwiGLU-style FFNs more than for raw embedding tables.

Embeddings are a lookup table, not a matrix multiply at the input, but
initialization still matters: it sets the starting scale of token vectors and
therefore the early gradient magnitudes into the rest of the network.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

InitStrategy = Literal[
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
]


def initialize_embedding(
    weight: torch.Tensor,
    strategy: InitStrategy = "xavier_uniform",
    *,
    std: float = 0.02,
    padding_idx: int | None = None,
) -> None:
    """In-place initialize an embedding weight matrix.

    Args:
        weight: Tensor of shape ``(vocab_size, hidden_size)``.
        strategy: Initialization scheme name.
        std: Standard deviation for ``normal`` strategy.
        padding_idx: If set, zero the corresponding row after init.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"embedding weight must be 2D, got shape {tuple(weight.shape)}"
        )

    with torch.no_grad():
        if strategy == "normal":
            nn.init.normal_(weight, mean=0.0, std=std)
        elif strategy == "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        elif strategy == "xavier_normal":
            nn.init.xavier_normal_(weight)
        elif strategy == "kaiming_uniform":
            nn.init.kaiming_uniform_(weight, a=0.0, nonlinearity="relu")
        elif strategy == "kaiming_normal":
            nn.init.kaiming_normal_(weight, a=0.0, nonlinearity="relu")
        else:
            raise ValueError(f"unknown init strategy: {strategy!r}")

        if padding_idx is not None:
            weight[padding_idx].zero_()


def describe_strategy(strategy: InitStrategy) -> str:
    """Human-readable rationale for an initialization strategy."""
    descriptions: dict[str, str] = {
        "normal": (
            "Independent N(0, σ²) draws. Common in GPT-style models (σ≈0.02). "
            "Does not adapt to fan-in/fan-out."
        ),
        "xavier_uniform": (
            "Glorot uniform: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)). "
            "Stabilizes variance through linear maps; Odyssey default for embeddings."
        ),
        "xavier_normal": (
            "Glorot normal: N(0, 2 / (fan_in + fan_out)). Same goal as xavier_uniform "
            "with Gaussian tails."
        ),
        "kaiming_uniform": (
            "He uniform for ReLU-family nets. Scales by fan-in only; typically "
            "larger than Xavier for the same shape."
        ),
        "kaiming_normal": (
            "He normal for ReLU-family nets. Prefer when the next nonlinearity "
            "is ReLU-like rather than embedding-only pipelines."
        ),
    }
    if strategy not in descriptions:
        raise ValueError(f"unknown init strategy: {strategy!r}")
    return descriptions[strategy]
