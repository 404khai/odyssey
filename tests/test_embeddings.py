"""Unit tests for OdysseyEmbedding."""

from __future__ import annotations

from pathlib import Path

import torch

from model import EmbeddingConfig, OdysseyEmbedding
from odyssey_tokenizer import OdysseyTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
BPE_MODEL = REPO_ROOT / "assets" / "tokenizer" / "bpe" / "odyssey.model"


def test_lookup_matches_weight_row() -> None:
    config = EmbeddingConfig(vocab_size=64, hidden_size=16, padding_idx=None)
    emb = OdysseyEmbedding(config)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(64 * 16, dtype=torch.float32).view(64, 16))

    ids = torch.tensor([[3, 7, 12]], dtype=torch.long)
    out = emb(ids)
    assert torch.equal(out[0, 0], emb.weight[3])
    assert torch.equal(out[0, 1], emb.weight[7])
    assert torch.equal(out[0, 2], emb.weight[12])


def test_padding_idx_is_zero_and_no_grad() -> None:
    config = EmbeddingConfig(
        vocab_size=32,
        hidden_size=8,
        padding_idx=0,
        init_strategy="normal",
        init_std=0.5,
    )
    emb = OdysseyEmbedding(config)
    assert torch.count_nonzero(emb.weight[0]) == 0

    ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = emb(ids)
    assert torch.equal(out[0, 0], torch.zeros(8))

    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    assert torch.count_nonzero(emb.weight.grad[0]) == 0


def test_gradient_flow_to_non_pad_rows() -> None:
    config = EmbeddingConfig(vocab_size=16, hidden_size=4, padding_idx=0)
    emb = OdysseyEmbedding(config)
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    out = emb(ids)
    out.sum().backward()
    assert emb.weight.grad is not None
    assert torch.count_nonzero(emb.weight.grad[1]) > 0
    assert torch.count_nonzero(emb.weight.grad[2]) > 0


def test_parameter_count() -> None:
    config = EmbeddingConfig(vocab_size=32000, hidden_size=768, padding_idx=0)
    emb = OdysseyEmbedding(config)
    assert emb.parameter_count() == 32000 * 768
    assert emb.inspect().parameter_count == 24_576_000
    assert emb.memory_bytes() == 24_576_000 * 4  # float32


def test_inspect_format() -> None:
    emb = OdysseyEmbedding(EmbeddingConfig(vocab_size=100, hidden_size=32))
    text = emb.inspect().format()
    assert "vocab_size" in text
    assert "100" in text
    assert "32" in text


def test_deterministic_lookup_with_fixed_weights() -> None:
    config = EmbeddingConfig(vocab_size=20, hidden_size=5, padding_idx=None)
    emb = OdysseyEmbedding(config)
    with torch.no_grad():
        emb.weight.copy_(torch.randn(20, 5))
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    a = emb(ids)
    b = emb(ids)
    assert torch.equal(a, b)


def test_rejects_out_of_range_ids() -> None:
    emb = OdysseyEmbedding(EmbeddingConfig(vocab_size=10, hidden_size=4))
    ids = torch.tensor([[0, 10]], dtype=torch.long)
    try:
        emb(ids)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "vocab_size" in str(exc)


def test_tokenizer_to_embedding_integration() -> None:
    """Tokenizer → embedding produces (1, seq, hidden)."""
    if BPE_MODEL.is_dir() and (BPE_MODEL / "vocab.json").is_file():
        tok = OdysseyTokenizer.load(BPE_MODEL)
        ids_list = tok.encode("Build authentication API")
        vocab = tok.vocab_size
    else:
        ids_list = [1, 2, 3, 4, 5]
        vocab = 64

    hidden = 32
    emb = OdysseyEmbedding(
        EmbeddingConfig(vocab_size=vocab, hidden_size=hidden, padding_idx=0)
    )
    ids = torch.tensor([ids_list], dtype=torch.long).clamp(max=vocab - 1)
    out = emb(ids)
    assert out.shape == (1, len(ids_list), hidden)


def test_load_embedding_config() -> None:
    from model import load_embedding_config

    config = load_embedding_config()
    assert config.vocab_size == 32000
    assert config.hidden_size == 768
    assert config.init_strategy == "xavier_uniform"
