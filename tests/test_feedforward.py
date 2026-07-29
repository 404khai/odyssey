"""Feed-forward public API tests."""

from __future__ import annotations

import torch

from model import FeedForwardConfig, OdysseyFeedForward, build_feed_forward


def test_build_feed_forward() -> None:
    cfg = FeedForwardConfig(hidden_size=16, intermediate_size=32)
    ffn = build_feed_forward(cfg)
    assert isinstance(ffn, OdysseyFeedForward)
    y = ffn(torch.randn(2, 4, 16))
    assert y.shape == (2, 4, 16)


def test_rejects_non_swiglu_type() -> None:
    try:
        FeedForwardConfig(type="gelu")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError as err:
        assert "swiglu" in str(err)
