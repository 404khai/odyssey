"""Property tests for RoPE rotations."""

from __future__ import annotations

import torch

from model import OdysseyRoPE, RopeConfig
from model.rope_math import inverse_frequencies, rotate_adjacent_pairs


def test_inverse_freq_length_and_positive() -> None:
    inv = inverse_frequencies(16, theta=10000.0)
    assert inv.shape == (8,)
    assert torch.all(inv > 0)
    # First freq is 1.0 when θ^{-0}=1
    assert torch.isclose(inv[0], torch.tensor(1.0))


def test_rotation_preserves_norm() -> None:
    rope = OdysseyRoPE(
        RopeConfig(head_dim=16, rotary_dim=16, max_position_embeddings=128)
    )
    x = torch.randn(7, 3, 16)
    y = rope(x, position_offset=11)
    nx = torch.linalg.vector_norm(x, dim=-1)
    ny = torch.linalg.vector_norm(y, dim=-1)
    assert torch.allclose(nx, ny, atol=1e-5)


def test_partial_rotary_leaves_tail() -> None:
    rope = OdysseyRoPE(RopeConfig(head_dim=8, rotary_dim=4, max_position_embeddings=32))
    x = torch.randn(5, 2, 8)
    y = rope(x, position_offset=4)
    assert torch.equal(x[..., 4:], y[..., 4:])
    assert not torch.allclose(x[..., :4], y[..., :4])


def test_adjacent_pair_formula() -> None:
    x = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    cos = torch.tensor([[0.0, 1.0]])  # 90° on pair0, 0° on pair1
    sin = torch.tensor([[1.0, 0.0]])
    y = rotate_adjacent_pairs(x, cos, sin)
    # pair0: (1,0) → (0,1); pair1: (0,1) → (0,1)
    assert torch.allclose(y, torch.tensor([[0.0, 1.0, 0.0, 1.0]]), atol=1e-6)


def test_different_positions_change_output() -> None:
    rope = OdysseyRoPE(RopeConfig(head_dim=4, rotary_dim=4, max_position_embeddings=64))
    x = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    y0 = rope(x, position_offset=0)
    y7 = rope(x, position_offset=7)
    assert (y0 - y7).abs().sum() > 1e-3
