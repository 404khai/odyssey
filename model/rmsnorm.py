"""LLaMA-style RMSNorm for Odyssey.

Canonical training-side normalization. Must stay numerically identical to
``phalanx::layers::RmsNorm`` (see ``scripts/validate_rmsnorm.py``).

Unlike LayerNorm, RMSNorm does **not** subtract the mean — it only scales by
the root-mean-square. That removes an unnecessary centering pass while
keeping activations in a healthy magnitude range for deep residual stacks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.config import NormConfig
from model.normalization import apply_scale, rms_normalize


class OdysseyRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Accepted layouts: any tensor whose last dimension equals ``hidden_size``.
    Typical training shape is ``(batch, seq, hidden_size)``.

    Parameters
    ----------
    config:
        Hidden size, epsilon, device, and dtype. ``gamma`` is initialized to
        ones — Spec v1 / LLaMA convention so the layer starts as near-identity
        scaling and learns magnitude corrections during training.
    """

    def __init__(self, config: NormConfig) -> None:
        super().__init__()
        self.config = config
        # Ones init: at step 0 the network sees pure RMS scaling without an
        # arbitrary learned stretch that would fight early optimization.
        self.weight = nn.Parameter(
            torch.ones(
                config.hidden_size,
                device=config.torch_device,
                dtype=config.torch_dtype,
            )
        )

    @classmethod
    def from_config(cls, config: NormConfig) -> OdysseyRMSNorm:
        return cls(config)

    @property
    def gamma(self) -> torch.Tensor:
        """Alias for the learnable scale parameter (Spec name ``γ``)."""
        return self.weight

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def parameter_count(self) -> int:
        return int(self.weight.numel())

    def memory_bytes(self) -> int:
        return self.parameter_count() * self.weight.element_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``γ ⊙ x / RMS(x)`` along the last dimension."""
        self.validate_input(x)
        normalized = rms_normalize(x, self.config.epsilon, dim=-1)
        return apply_scale(normalized, self.weight)

    def validate_input(self, x: torch.Tensor) -> None:
        if x.ndim < 1:
            raise ValueError(f"expected at least rank-1 input, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"last dim {x.shape[-1]} != configured hidden_size "
                f"{self.config.hidden_size}"
            )

    def inspect(self) -> dict[str, Any]:
        """Human-readable layer summary for residual inspectors / logs."""
        return {
            "type": self.config.type,
            "hidden_size": self.config.hidden_size,
            "epsilon": self.config.epsilon,
            "parameter_count": self.parameter_count(),
            "memory_bytes": self.memory_bytes(),
            "device": str(self.weight.device),
            "dtype": str(self.weight.dtype),
            "gamma_mean": float(self.weight.detach().float().mean()),
            "gamma_std": float(self.weight.detach().float().std(unbiased=False)),
        }

    def format_inspect(self) -> str:
        info = self.inspect()
        return (
            f"OdysseyRMSNorm(type={info['type']}, D={info['hidden_size']}, "
            f"eps={info['epsilon']}, params={info['parameter_count']}, "
            f"mem={info['memory_bytes']} B)"
        )
