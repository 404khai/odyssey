"""Activation unit tests."""

from __future__ import annotations

import torch

from model.activations import sigmoid, silu, swish


def test_sigmoid_bounds() -> None:
    x = torch.linspace(-5, 5, 21)
    y = sigmoid(x)
    assert torch.all(y > 0) and torch.all(y < 1)


def test_silu_zero_at_zero() -> None:
    assert silu(torch.tensor(0.0)).item() == 0.0


def test_swish_alias() -> None:
    x = torch.randn(8)
    assert torch.equal(swish(x), silu(x))
