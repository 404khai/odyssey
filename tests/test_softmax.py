"""Tests for stable Softmax."""

from __future__ import annotations

import torch

from model.softmax import stable_softmax


def test_sums_to_one() -> None:
    x = torch.randn(2, 4, 8)
    y = stable_softmax(x, dim=-1)
    assert torch.allclose(y.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_max_invariance() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]])
    a = stable_softmax(x)
    b = stable_softmax(x + 1000.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_blocks_neg_inf() -> None:
    x = torch.tensor([[0.0, float("-inf"), 0.0]])
    y = stable_softmax(x)
    assert y[0, 1].item() == 0.0
    assert abs(y.sum().item() - 1.0) < 1e-6
