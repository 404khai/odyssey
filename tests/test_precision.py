"""Precision / stability tests for attention Softmax path."""

from __future__ import annotations

import torch

from model.attention import OdysseyAttention
from model.config import AttentionConfig
from model.softmax import stable_softmax


def test_large_scores_stable() -> None:
    x = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0]])
    y = stable_softmax(x)
    assert torch.isfinite(y).all()
    assert abs(y.sum().item() - 1.0) < 1e-5


def test_float32_attention_finite() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    attn = OdysseyAttention(cfg)
    y = attn(torch.randn(2, 8, 32) * 10)
    assert torch.isfinite(y).all()
