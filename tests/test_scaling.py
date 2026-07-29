"""RoPE scaling tests."""

from __future__ import annotations

import pytest
import torch

from model import OdysseyRoPE, RopeConfig


def test_linear_scaling_changes_angles() -> None:
    base = RopeConfig(
        head_dim=8,
        rotary_dim=8,
        max_position_embeddings=64,
        scaling="none",
        scaling_factor=1.0,
    )
    scaled = RopeConfig(
        head_dim=8,
        rotary_dim=8,
        max_position_embeddings=64,
        scaling="linear",
        scaling_factor=2.0,
    )
    x = torch.randn(1, 1, 8)
    y1 = OdysseyRoPE(base)(x, position_offset=10)
    y2 = OdysseyRoPE(scaled)(x, position_offset=10)
    assert not torch.allclose(y1, y2)


def test_linear_scale_matches_half_position() -> None:
    """m'=m/2 at offset 10 ≈ unscaled offset 5 for the same x."""
    cfg_s = RopeConfig(
        head_dim=8,
        rotary_dim=8,
        max_position_embeddings=64,
        scaling="linear",
        scaling_factor=2.0,
    )
    cfg = RopeConfig(
        head_dim=8,
        rotary_dim=8,
        max_position_embeddings=64,
        scaling="none",
    )
    x = torch.randn(1, 2, 8)
    y_scaled = OdysseyRoPE(cfg_s)(x, position_offset=10)
    y_half = OdysseyRoPE(cfg)(x, position_offset=5)
    assert torch.allclose(y_scaled, y_half, atol=1e-5)


def test_rejects_yarn() -> None:
    with pytest.raises(ValueError, match="scaling"):
        RopeConfig(scaling="yarn")  # type: ignore[arg-type]
