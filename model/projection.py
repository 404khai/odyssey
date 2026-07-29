"""Q / K / V / O linear projections for attention.

Weights use the Spec layout ``(out_features, in_features)`` matching
``nn.Linear`` / Phalanx ``y = x @ Wᵀ``. No biases (Odyssey Spec v1.0.0).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.config import AttentionConfig


class AttentionProjections(nn.Module):
    """Four bias-free projections: query, key, value, output."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        q_out = config.query_dim
        kv_out = config.kv_dim
        device = config.torch_device
        dtype = config.torch_dtype
        self.q_proj = nn.Linear(d, q_out, bias=config.bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d, kv_out, bias=config.bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d, kv_out, bias=config.bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(q_out, d, bias=config.bias, device=device, dtype=dtype)

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q, K, V with float64 GEMM accumulators (Phalanx parity)."""
        x_f = x.float()
        q = self._linear(x_f, self.q_proj.weight)
        k = self._linear(x_f, self.k_proj.weight)
        v = self._linear(x_f, self.v_proj.weight)
        return q, k, v

    def project_output(self, x: torch.Tensor) -> torch.Tensor:
        """Merge-head tensor → residual stream width."""
        return self._linear(x.float(), self.o_proj.weight)

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (x.double() @ weight.double().T).to(dtype=torch.float32)

    def inspect(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "q_proj": list(self.q_proj.weight.shape),
            "k_proj": list(self.k_proj.weight.shape),
            "v_proj": list(self.v_proj.weight.shape),
            "o_proj": list(self.o_proj.weight.shape),
            "bias": cfg.bias,
            "query_dim": cfg.query_dim,
            "kv_dim": cfg.kv_dim,
        }
