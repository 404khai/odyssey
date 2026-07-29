"""Gradient propagation tests for OdysseyRMSNorm + residual add."""

from __future__ import annotations

import torch

from model import NormConfig, OdysseyRMSNorm, residual_add


def test_rmsnorm_gradients_flow() -> None:
    norm = OdysseyRMSNorm(NormConfig(hidden_size=32))
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = norm(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_residual_gradients_preserve_identity_path() -> None:
    x = torch.randn(2, 4, 16, requires_grad=True)
    f = torch.randn(2, 4, 16, requires_grad=True)
    y = residual_add(x, f)
    y.sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    assert f.grad is not None and torch.allclose(f.grad, torch.ones_like(f))
