"""Unit tests for residual pathway helpers."""

from __future__ import annotations

import torch

from model import NormConfig, OdysseyRMSNorm, describe_residual_flow, pre_norm_residual, residual_add


def test_residual_add_matches_sum() -> None:
    x = torch.randn(2, 4, 8)
    f = torch.randn(2, 4, 8)
    y = residual_add(x, f)
    assert torch.equal(y, x + f)


def test_residual_rejects_shape_mismatch() -> None:
    try:
        residual_add(torch.zeros(2, 4, 8), torch.zeros(2, 4, 4))
        raise AssertionError("expected ValueError")
    except ValueError as err:
        assert "shapes must match" in str(err)


def test_pre_norm_residual_identity_sublayer() -> None:
    cfg = NormConfig(hidden_size=16, epsilon=1e-6)
    norm = OdysseyRMSNorm(cfg)
    x = torch.randn(1, 3, 16)
    y = pre_norm_residual(x, norm=norm, sublayer=lambda t: t)
    # y = x + norm(x)
    assert torch.allclose(y, x + norm(x))


def test_describe_residual_flow() -> None:
    flow = describe_residual_flow(hidden_size=768, batch=2, seq_len=16)
    assert flow["ordering"] == "pre-norm"
    assert flow["steps"][0]["stage"] == "input"
    assert flow["steps"][-1]["shape"] == (2, 16, 768)
