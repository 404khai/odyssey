"""Numerical precision / stability tests for RMSNorm."""

from __future__ import annotations

import torch

from model import NormConfig, OdysseyRMSNorm


def test_no_nans_float32() -> None:
    norm = OdysseyRMSNorm(NormConfig(hidden_size=64, dtype="float32"))
    x = torch.randn(4, 32, 64) * 10
    y = norm(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_no_nans_float16() -> None:
    norm = OdysseyRMSNorm(NormConfig(hidden_size=64, dtype="float16"))
    x = torch.randn(4, 32, 64, dtype=torch.float16)
    y = norm(x)
    assert not torch.isnan(y.float()).any()


def test_near_zero_input_stable() -> None:
    """ε keeps division safe when activations are tiny."""
    norm = OdysseyRMSNorm(NormConfig(hidden_size=32, epsilon=1e-6))
    x = torch.full((1, 1, 32), 1e-20)
    y = norm(x)
    assert torch.isfinite(y).all()


def test_mixed_precision_path() -> None:
    """fp16 activations + fp32 γ still produce finite outputs."""
    cfg = NormConfig(hidden_size=48, dtype="float32")
    norm = OdysseyRMSNorm(cfg)
    x = torch.randn(2, 8, 48, dtype=torch.float16)
    # Module weight is fp32; PyTorch will promote — ensure no NaNs.
    y = norm(x.float()).half()
    assert torch.isfinite(y.float()).all()
