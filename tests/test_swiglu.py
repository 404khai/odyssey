"""Unit tests for OdysseySwiGLU."""

from __future__ import annotations

import torch

from model import (
    EmbeddingConfig,
    FeedForwardConfig,
    NormConfig,
    OdysseyEmbedding,
    OdysseyRMSNorm,
    OdysseyRoPE,
    OdysseySwiGLU,
    RopeConfig,
    load_feed_forward_config,
)
from model.activations import silu


def test_output_shape() -> None:
    cfg = FeedForwardConfig(hidden_size=32, intermediate_size=64)
    ffn = OdysseySwiGLU(cfg)
    x = torch.randn(2, 8, 32)
    assert ffn(x).shape == x.shape


def test_silu_formula() -> None:
    x = torch.tensor([-2.0, 0.0, 1.5])
    expected = x * torch.sigmoid(x)
    assert torch.allclose(silu(x), expected)


def test_gate_multiplication() -> None:
    cfg = FeedForwardConfig(hidden_size=4, intermediate_size=8)
    ffn = OdysseySwiGLU(cfg)
    with torch.no_grad():
        ffn.gate_proj.weight.zero_()
        ffn.up_proj.weight.fill_(1.0)
        ffn.down_proj.weight.zero_()
        ffn.down_proj.weight[0, 0] = 1.0
    x = torch.ones(1, 1, 4)
    # gate = silu(0)=0 → hidden=0 → output near 0
    y = ffn(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_config_load() -> None:
    cfg = load_feed_forward_config()
    assert cfg.type == "swiglu"
    assert cfg.hidden_size == 768
    assert cfg.intermediate_size == 2048
    assert cfg.activation == "silu"


def test_parameter_count() -> None:
    cfg = FeedForwardConfig(hidden_size=16, intermediate_size=32)
    ffn = OdysseySwiGLU(cfg)
    expected = 16 * 32 + 16 * 32 + 32 * 16
    assert ffn.parameter_count() == expected
    info = ffn.inspect()
    assert info["projections"]["gate_proj (w1)"] == 16 * 32


def test_float16_forward() -> None:
    cfg = FeedForwardConfig(hidden_size=16, intermediate_size=32, dtype="float16")
    ffn = OdysseySwiGLU(cfg)
    x = torch.randn(1, 4, 16, dtype=torch.float16)
    y = ffn(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()


def test_gradients() -> None:
    cfg = FeedForwardConfig(hidden_size=16, intermediate_size=32)
    ffn = OdysseySwiGLU(cfg)
    x = torch.randn(2, 3, 16, requires_grad=True)
    ffn(x).pow(2).mean().backward()
    assert x.grad is not None
    assert ffn.gate_proj.weight.grad is not None


def test_deterministic() -> None:
    cfg = FeedForwardConfig(hidden_size=8, intermediate_size=16)
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8)
    w1 = torch.randn(16, 8)
    w3 = torch.randn(16, 8)
    w2 = torch.randn(8, 16)

    def run() -> torch.Tensor:
        f = OdysseySwiGLU(cfg)
        with torch.no_grad():
            f.gate_proj.weight.copy_(w1)
            f.up_proj.weight.copy_(w3)
            f.down_proj.weight.copy_(w2)
        return f(x)

    assert torch.equal(run(), run())


def test_integration_stack() -> None:
    emb = OdysseyEmbedding(
        EmbeddingConfig(vocab_size=32, hidden_size=32, padding_idx=None)
    )
    rope = OdysseyRoPE(RopeConfig(head_dim=8, rotary_dim=8, max_position_embeddings=16))
    norm = OdysseyRMSNorm(NormConfig(hidden_size=32))
    ffn = OdysseySwiGLU(FeedForwardConfig(hidden_size=32, intermediate_size=64))
    ids = torch.randint(0, 32, (1, 4))
    h = emb(ids)
    q = h.view(1, 4, 4, 8)
    h = rope(q).reshape(1, 4, 32)
    h = ffn(norm(h))
    assert h.shape == (1, 4, 32)
