"""Attention tensor shape tests."""

from __future__ import annotations

import torch

from model.attention import OdysseyAttention
from model.attention_math import expand_kv_heads, merge_heads, reshape_to_heads
from model.config import AttentionConfig


def test_reshape_roundtrip() -> None:
    x = torch.randn(2, 5, 32)  # H=4, d=8
    h = reshape_to_heads(x, 4)
    assert h.shape == (2, 4, 5, 8)
    assert torch.allclose(merge_heads(h), x)


def test_gqa_expand() -> None:
    kv = torch.randn(1, 2, 4, 8)
    expanded = expand_kv_heads(kv, 6)
    assert expanded.shape == (1, 6, 4, 8)
    # Heads 0-2 share KV0, 3-5 share KV1
    assert torch.equal(expanded[:, 0], expanded[:, 1])
    assert torch.equal(expanded[:, 0], kv[:, 0])
    assert torch.equal(expanded[:, 3], kv[:, 1])


def test_attention_output_shape_gqa() -> None:
    cfg = AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=8)
    attn = OdysseyAttention(cfg)
    x = torch.randn(2, 7, 48)
    y = attn(x)
    assert y.shape == x.shape


def test_attention_output_shape_mha() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=4, head_dim=8)
    attn = OdysseyAttention(cfg)
    x = torch.randn(1, 5, 32)
    assert attn(x).shape == x.shape
