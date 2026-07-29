"""Unit / integration tests for OdysseyAttention."""

from __future__ import annotations

import torch

from model import (
    AttentionConfig,
    EmbeddingConfig,
    NormConfig,
    OdysseyAttention,
    OdysseyEmbedding,
    OdysseyMultiHeadAttention,
    OdysseyRMSNorm,
    OdysseyRoPE,
    RopeConfig,
)


def test_parameter_count_gqa_vs_mha() -> None:
    gqa = OdysseyAttention(
        AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=16)
    )
    mha = OdysseyAttention(
        AttentionConfig(num_heads=8, num_kv_heads=8, head_dim=16)
    )
    # GQA saves on K/V projections: H_kv * d * D instead of H * d * D (×2 for K+V)
    assert gqa.parameter_count() < mha.parameter_count()


def test_causal_no_future_leak() -> None:
    cfg = AttentionConfig(num_heads=2, num_kv_heads=2, head_dim=4)
    attn = OdysseyAttention(cfg)
    attn.eval()
    with torch.no_grad():
        # Zero all weights except identity-like so output depends on attended values.
        for p in attn.parameters():
            p.zero_()
        # q,k near zero → uniform over allowed; set V and O so we can detect positions.
        attn.projections.v_proj.weight.copy_(
            torch.eye(cfg.kv_dim, cfg.hidden_size)
        )
        attn.projections.o_proj.weight.copy_(
            torch.eye(cfg.hidden_size, cfg.query_dim)
        )
        # Distinct token embeddings along seq
        x = torch.zeros(1, 4, cfg.hidden_size)
        for t in range(4):
            x[0, t, t % cfg.hidden_size] = float(t + 1)
        _, w = attn(x, return_weights=True)
        for s in range(4):
            for t in range(s + 1, 4):
                assert w[0, 0, s, t].item() < 1e-6


def test_with_rope() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    rope = OdysseyRoPE(
        RopeConfig(head_dim=8, rotary_dim=8, max_position_embeddings=32)
    )
    attn = OdysseyAttention(cfg, rope=rope)
    x = torch.randn(2, 5, 32)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_float16() -> None:
    cfg = AttentionConfig(
        num_heads=4, num_kv_heads=2, head_dim=8, dtype="float16"
    )
    attn = OdysseyAttention(cfg)
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    y = attn(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()


def test_gradients() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    attn = OdysseyAttention(cfg)
    x = torch.randn(2, 3, 32, requires_grad=True)
    attn(x).pow(2).mean().backward()
    assert x.grad is not None
    assert attn.projections.q_proj.weight.grad is not None


def test_deterministic() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32)
    wq = torch.randn(32, 32)
    wk = torch.randn(16, 32)
    wv = torch.randn(16, 32)
    wo = torch.randn(32, 32)

    def run() -> torch.Tensor:
        a = OdysseyAttention(cfg)
        with torch.no_grad():
            a.projections.q_proj.weight.copy_(wq)
            a.projections.k_proj.weight.copy_(wk)
            a.projections.v_proj.weight.copy_(wv)
            a.projections.o_proj.weight.copy_(wo)
        return a(x)

    assert torch.equal(run(), run())


def test_inspect() -> None:
    cfg = AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=8)
    info = OdysseyAttention(cfg).inspect()
    assert info["type"] == "gqa"
    assert info["parameter_count"] > 0
    assert "q" in info["shapes"]


def test_mha_reference() -> None:
    cfg = AttentionConfig(num_heads=4, num_kv_heads=4, head_dim=8)
    mha = OdysseyMultiHeadAttention(cfg)
    assert mha(torch.randn(1, 4, 32)).shape == (1, 4, 32)


def test_integration_stack() -> None:
    hidden = 32
    emb = OdysseyEmbedding(
        EmbeddingConfig(vocab_size=32, hidden_size=hidden, padding_idx=None)
    )
    rope = OdysseyRoPE(
        RopeConfig(head_dim=8, rotary_dim=8, max_position_embeddings=16)
    )
    norm = OdysseyRMSNorm(NormConfig(hidden_size=hidden))
    attn = OdysseyAttention(
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8),
        rope=rope,
    )
    ids = torch.randint(0, 32, (1, 4))
    h = emb(ids)
    h = attn(norm(h))
    assert h.shape == (1, 4, hidden)
