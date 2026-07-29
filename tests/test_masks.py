"""Tests for causal masking."""

from __future__ import annotations

import torch

from model.causal_mask import apply_causal_mask, make_causal_mask
from model.softmax import stable_softmax


def test_lower_triangular() -> None:
    m = make_causal_mask(4)
    assert m.shape == (4, 4)
    for s in range(4):
        for t in range(4):
            if t <= s:
                assert m[s, t].item() == 0.0
            else:
                assert m[s, t].item() == float("-inf")


def test_blocks_future_after_softmax() -> None:
    scores = torch.ones(1, 1, 4, 4)
    masked = apply_causal_mask(scores)
    w = stable_softmax(masked, dim=-1)
    # Upper triangle must be ~0
    for s in range(4):
        for t in range(s + 1, 4):
            assert w[0, 0, s, t].item() < 1e-6
