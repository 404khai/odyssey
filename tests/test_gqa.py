"""GQA-specific tests."""

from __future__ import annotations

import torch

from model.attention import OdysseyAttention, OdysseyMultiHeadAttention
from model.config import AttentionConfig, load_attention_config
from model.gqa import OdysseyGQA


def test_config_load_gqa() -> None:
    cfg = load_attention_config()
    assert cfg.num_heads == 12
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 64
    assert cfg.is_gqa
    assert cfg.gqa_groups == 3
    assert not cfg.bias


def test_mha_wrapper_rejects_gqa() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    try:
        OdysseyMultiHeadAttention(cfg)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_weights_sum_to_one() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    attn = OdysseyAttention(cfg)
    attn.eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        _, w = attn(x, return_weights=True)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gqa_module_shapes() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=1, head_dim=8)
    gqa = OdysseyGQA(cfg)
    q = torch.randn(1, 5, 32)
    k = torch.randn(1, 5, 8)
    v = torch.randn(1, 5, 8)
    out = gqa(q, k, v)
    assert out.shape == (1, 5, 32)
