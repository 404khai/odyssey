"""Unit tests for Odyssey RoPE."""

from __future__ import annotations

import pytest
import torch

from model import EmbeddingConfig, OdysseyEmbedding, OdysseyRoPE, RopeConfig


@pytest.fixture
def rope() -> OdysseyRoPE:
    return OdysseyRoPE(
        RopeConfig(
            theta=10000.0,
            head_dim=8,
            rotary_dim=8,
            max_position_embeddings=64,
        )
    )


def test_shape_preserved_rank4(rope: OdysseyRoPE) -> None:
    q = torch.randn(2, 5, 3, 8)
    out = rope(q)
    assert out.shape == q.shape


def test_shape_preserved_phalanx_layout(rope: OdysseyRoPE) -> None:
    x = torch.randn(5, 3, 8)
    assert rope(x).shape == x.shape


def test_apply_rotary_returns_pair(rope: OdysseyRoPE) -> None:
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 2, 8)
    q2, k2 = rope.apply_rotary(q, k, position_offset=1)
    assert q2.shape == q.shape and k2.shape == k.shape


def test_position_zero_near_identity(rope: OdysseyRoPE) -> None:
    x = torch.randn(1, 2, 8)
    y = rope(x, position_offset=0)
    assert torch.allclose(x, y, atol=1e-5)


def test_deterministic(rope: OdysseyRoPE) -> None:
    x = torch.randn(3, 2, 8)
    a = rope(x, position_offset=2)
    b = rope(x, position_offset=2)
    assert torch.equal(a, b)


def test_embedding_then_rope_shapes() -> None:
    emb = OdysseyEmbedding(
        EmbeddingConfig(vocab_size=50, hidden_size=32, padding_idx=0)
    )
    # Reshape embedding to fake multi-head: D=32 → H=4, d=8
    ids = torch.randint(0, 50, (1, 6))
    h = emb(ids).view(1, 6, 4, 8)
    rope = OdysseyRoPE(RopeConfig(head_dim=8, rotary_dim=8, max_position_embeddings=32))
    out = rope(h)
    assert out.shape == (1, 6, 4, 8)


def test_rejects_bad_head_dim(rope: OdysseyRoPE) -> None:
    with pytest.raises(ValueError, match="head_dim"):
        rope(torch.randn(2, 2, 4))
