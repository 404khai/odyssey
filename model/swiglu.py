"""LLaMA-style SwiGLU feed-forward block for Odyssey.

Canonical formula (Odyssey Spec v1.0.0 / Phalanx ``layers::SwiGlu``):

    FFN(x) = (SiLU(x W1ᵀ) ⊙ (x W3ᵀ)) W2ᵀ

Weight shapes: ``w1,w3 = (I, D)``, ``w2 = (D, I)``. No biases.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.activations import silu
from model.config import FeedForwardConfig
from model.parameter_counter import (
    count_parameters,
    memory_bytes,
    projection_breakdown,
)


class OdysseySwiGLU(nn.Module):
    """Gated SwiGLU MLP applied position-wise on the last dimension.

    Module names map to Spec / GGUF:

    | Module       | Spec | GGUF        |
    |--------------|------|-------------|
    | ``gate_proj``| w1   | ffn_gate    |
    | ``up_proj``  | w3   | ffn_up      |
    | ``down_proj``| w2   | ffn_down    |
    """

    def __init__(self, config: FeedForwardConfig) -> None:
        super().__init__()
        if config.type != "swiglu":
            raise ValueError(f"feed_forward.type must be 'swiglu', got {config.type!r}")
        if config.activation not in ("silu", "swish", "swiglu"):
            raise ValueError(
                f"activation must be silu/swish/swiglu, got {config.activation!r}"
            )
        self.config = config
        d = config.hidden_size
        i = config.intermediate_size
        # bias=False — Spec forbids FFN biases.
        self.gate_proj = nn.Linear(
            d, i, bias=False, device=config.torch_device, dtype=config.torch_dtype
        )
        self.up_proj = nn.Linear(
            d, i, bias=False, device=config.torch_device, dtype=config.torch_dtype
        )
        self.down_proj = nn.Linear(
            i, d, bias=False, device=config.torch_device, dtype=config.torch_dtype
        )

    @classmethod
    def from_config(cls, config: FeedForwardConfig) -> OdysseySwiGLU:
        return cls(config)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.config.intermediate_size

    def parameter_count(self) -> int:
        return count_parameters(self)

    def memory_bytes(self) -> int:
        return memory_bytes(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN. Shape ``(..., D)`` → ``(..., D)``.

        Mirrors Phalanx: float64 GEMM accumulators, float32 SiLU + Hadamard.
        """
        self.validate_input(x)
        x_f = x.float()
        gate = self._linear(x_f, self.gate_proj.weight)
        up = self._linear(x_f, self.up_proj.weight)
        gated = silu(gate) * up
        out = self._linear(gated, self.down_proj.weight)
        return out.to(dtype=x.dtype)

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``x @ Wᵀ`` with float64 accumulation (Phalanx ``Tensor::matmul``)."""
        return (x.double() @ weight.double().T).to(dtype=torch.float32)

    def validate_input(self, x: torch.Tensor) -> None:
        if x.ndim < 1:
            raise ValueError(f"expected rank >= 1, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"last dim {x.shape[-1]} != hidden_size {self.config.hidden_size}"
            )

    def inspect(self) -> dict[str, Any]:
        breakdown = projection_breakdown(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
        )
        return {
            "type": self.config.type,
            "activation": self.config.activation,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "expansion_ratio": breakdown["expansion_ratio"],
            "parameter_count": self.parameter_count(),
            "memory_bytes": self.memory_bytes(),
            "projections": {
                "gate_proj (w1)": breakdown["gate_proj_params"],
                "up_proj (w3)": breakdown["up_proj_params"],
                "down_proj (w2)": breakdown["down_proj_params"],
            },
            "shapes": {
                "input": f"(..., {self.config.hidden_size})",
                "gate/up": f"(..., {self.config.intermediate_size})",
                "output": f"(..., {self.config.hidden_size})",
            },
            "device": str(self.gate_proj.weight.device),
            "dtype": str(self.gate_proj.weight.dtype),
        }

    def format_inspect(self) -> str:
        info = self.inspect()
        return (
            f"OdysseySwiGLU(D={info['hidden_size']}, I={info['intermediate_size']}, "
            f"ratio={info['expansion_ratio']:.3f}, params={info['parameter_count']:,})"
        )
