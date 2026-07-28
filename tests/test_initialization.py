"""Tests for embedding weight initialization."""

from __future__ import annotations

import pytest
import torch

from model import (
    EmbeddingConfig,
    OdysseyEmbedding,
    describe_strategy,
    initialize_embedding,
)
from model.config import VALID_INIT_STRATEGIES


@pytest.mark.parametrize("strategy", VALID_INIT_STRATEGIES)
def test_all_strategies_run(strategy: str) -> None:
    weight = torch.empty(50, 16)
    initialize_embedding(weight, strategy, std=0.02, padding_idx=0)  # type: ignore[arg-type]
    assert weight.shape == (50, 16)
    assert torch.count_nonzero(weight[0]) == 0
    assert torch.isfinite(weight).all()


def test_normal_respects_std() -> None:
    torch.manual_seed(0)
    weight = torch.empty(10_000, 32)
    initialize_embedding(weight, "normal", std=0.05, padding_idx=None)
    # Sample std should be near 0.05 (loose bound).
    assert 0.03 < float(weight.std()) < 0.07


def test_describe_strategy_covers_all() -> None:
    for strategy in VALID_INIT_STRATEGIES:
        text = describe_strategy(strategy)  # type: ignore[arg-type]
        assert len(text) > 20


def test_unknown_strategy_raises() -> None:
    weight = torch.empty(4, 4)
    with pytest.raises(ValueError, match="unknown"):
        initialize_embedding(weight, "magic")  # type: ignore[arg-type]


def test_config_rejects_bad_strategy() -> None:
    with pytest.raises(ValueError, match="init_strategy"):
        EmbeddingConfig(init_strategy="nope")  # type: ignore[arg-type]


def test_odyssey_embedding_uses_strategy() -> None:
    torch.manual_seed(1)
    a = OdysseyEmbedding(
        EmbeddingConfig(
            vocab_size=40,
            hidden_size=8,
            padding_idx=None,
            init_strategy="normal",
            init_std=0.02,
        )
    )
    torch.manual_seed(1)
    b = OdysseyEmbedding(
        EmbeddingConfig(
            vocab_size=40,
            hidden_size=8,
            padding_idx=None,
            init_strategy="normal",
            init_std=0.02,
        )
    )
    assert torch.allclose(a.weight, b.weight)
