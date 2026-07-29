"""Additional RoPE shape tests (batch layouts)."""

from __future__ import annotations

import pytest
import torch

from model import OdysseyRoPE, RopeConfig, load_model_config


def test_load_model_yaml_rope() -> None:
    cfg = load_model_config()
    assert cfg.rope.theta == 10000.0
    assert cfg.rope.rotary_dim == 64
    assert cfg.head_dim == 64


@pytest.mark.parametrize(
    "shape",
    [
        (8, 64),
        (8, 4, 64),
        (2, 8, 4, 64),
    ],
)
def test_layouts(shape: tuple[int, ...]) -> None:
    rope = OdysseyRoPE(
        RopeConfig(head_dim=64, rotary_dim=64, max_position_embeddings=128)
    )
    x = torch.randn(*shape)
    y = rope(x, position_offset=0)
    assert y.shape == shape
