"""Shape tests for SwiGLU / FFN."""

from __future__ import annotations

import torch

from model import FeedForwardConfig, OdysseySwiGLU


def test_rank2_and_rank3() -> None:
    ffn = OdysseySwiGLU(FeedForwardConfig(hidden_size=24, intermediate_size=48))
    assert ffn(torch.randn(5, 24)).shape == (5, 24)
    assert ffn(torch.randn(2, 5, 24)).shape == (2, 5, 24)


def test_rejects_bad_hidden() -> None:
    ffn = OdysseySwiGLU(FeedForwardConfig(hidden_size=24, intermediate_size=48))
    try:
        ffn(torch.randn(2, 16))
        raise AssertionError("expected ValueError")
    except ValueError as err:
        assert "hidden_size" in str(err)
