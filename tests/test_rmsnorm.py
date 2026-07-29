"""Unit tests for OdysseyRMSNorm."""

from __future__ import annotations

import torch

from model import NormConfig, OdysseyEmbedding, OdysseyRMSNorm, OdysseyRoPE, load_norm_config


def test_output_shape_matches_input() -> None:
    cfg = NormConfig(hidden_size=32, epsilon=1e-6)
    norm = OdysseyRMSNorm(cfg)
    x = torch.randn(2, 8, 32)
    y = norm(x)
    assert y.shape == x.shape


def test_gamma_initialized_to_ones() -> None:
    cfg = NormConfig(hidden_size=16)
    norm = OdysseyRMSNorm(cfg)
    assert torch.allclose(norm.weight, torch.ones(16))


def test_epsilon_from_config() -> None:
    cfg = load_norm_config()
    assert cfg.type == "rmsnorm"
    assert cfg.epsilon == 1e-6
    assert cfg.hidden_size == 768


def test_float32_forward_finite() -> None:
    cfg = NormConfig(hidden_size=64, dtype="float32")
    norm = OdysseyRMSNorm(cfg)
    y = norm(torch.randn(1, 4, 64))
    assert torch.isfinite(y).all()


def test_float16_forward_finite() -> None:
    cfg = NormConfig(hidden_size=64, dtype="float16")
    norm = OdysseyRMSNorm(cfg)
    x = torch.randn(1, 4, 64, dtype=torch.float16)
    y = norm(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()


def test_normalized_has_unit_rms() -> None:
    cfg = NormConfig(hidden_size=128, epsilon=1e-6)
    norm = OdysseyRMSNorm(cfg)
    x = torch.randn(3, 5, 128)
    y = norm(x)
    # With γ=1, RMS along last dim ≈ 1.
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_gamma_scales_output() -> None:
    cfg = NormConfig(hidden_size=8, epsilon=1e-6)
    norm = OdysseyRMSNorm(cfg)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(1, 2, 8)
    y = norm(x)
    # Rebuild ones-γ reference.
    ref = OdysseyRMSNorm(cfg)
    y1 = ref(x)
    assert torch.allclose(y, y1 * 2.0, atol=1e-5)


def test_rejects_wrong_hidden() -> None:
    cfg = NormConfig(hidden_size=16)
    norm = OdysseyRMSNorm(cfg)
    try:
        norm(torch.randn(1, 8))
        raise AssertionError("expected ValueError")
    except ValueError as err:
        assert "hidden_size" in str(err)


def test_inspect_reports_params() -> None:
    cfg = NormConfig(hidden_size=768)
    norm = OdysseyRMSNorm(cfg)
    info = norm.inspect()
    assert info["parameter_count"] == 768
    assert info["memory_bytes"] == 768 * 4
    assert "RMSNorm" in norm.format_inspect()


def test_integration_embedding_rope_rmsnorm() -> None:
    """Embedding → RoPE path on head dims, then RMSNorm on residual stream."""
    from model import EmbeddingConfig, RopeConfig

    emb = OdysseyEmbedding(EmbeddingConfig(vocab_size=64, hidden_size=32, padding_idx=None))
    rope = OdysseyRoPE(
        RopeConfig(head_dim=8, rotary_dim=8, max_position_embeddings=32)
    )
    norm = OdysseyRMSNorm(NormConfig(hidden_size=32))

    ids = torch.randint(0, 64, (2, 4))
    h = emb(ids)  # (2, 4, 32)
    # Treat as 4 heads of dim 8 for RoPE, then flatten back.
    q = h.view(2, 4, 4, 8)
    q_rot = rope(q, position_offset=0)
    stream = q_rot.reshape(2, 4, 32)
    out = norm(stream)
    assert out.shape == (2, 4, 32)
    assert torch.isfinite(out).all()


def test_deterministic_with_fixed_weights() -> None:
    cfg = NormConfig(hidden_size=16, epsilon=1e-6)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    gamma = torch.randn(16)

    def run() -> torch.Tensor:
        n = OdysseyRMSNorm(cfg)
        with torch.no_grad():
            n.weight.copy_(gamma)
        return n(x)

    assert torch.equal(run(), run())
