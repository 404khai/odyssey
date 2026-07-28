"""Shape validation tests for the embedding layer."""

from __future__ import annotations

import pytest
import torch

from model import EmbeddingConfig, OdysseyEmbedding


@pytest.fixture
def emb() -> OdysseyEmbedding:
    return OdysseyEmbedding(
        EmbeddingConfig(vocab_size=128, hidden_size=64, padding_idx=0)
    )


def test_output_shape_batch_sequence_hidden(emb: OdysseyEmbedding) -> None:
    ids = torch.randint(0, 128, (4, 17), dtype=torch.long)
    out = emb(ids)
    assert out.shape == (4, 17, 64)


def test_phase3_example_shape() -> None:
    """AGENTS.md example: [512, 1284, 7] → (3, hidden) as a batch of length 1."""
    hidden = 768
    emb = OdysseyEmbedding(
        EmbeddingConfig(vocab_size=32000, hidden_size=hidden, padding_idx=0)
    )
    ids = torch.tensor([[512, 1284, 7]], dtype=torch.long)
    out = emb(ids)
    assert out.shape == (1, 3, hidden)


def test_rejects_1d_input(emb: OdysseyEmbedding) -> None:
    with pytest.raises(ValueError, match="batch, sequence"):
        emb(torch.tensor([1, 2, 3], dtype=torch.long))


def test_rejects_3d_input(emb: OdysseyEmbedding) -> None:
    with pytest.raises(ValueError, match="batch, sequence"):
        emb(torch.zeros(2, 3, 4, dtype=torch.long))


def test_rejects_float_ids(emb: OdysseyEmbedding) -> None:
    with pytest.raises(TypeError, match="integer"):
        emb(torch.zeros(2, 3, dtype=torch.float32))


def test_empty_sequence_shape(emb: OdysseyEmbedding) -> None:
    ids = torch.zeros(2, 0, dtype=torch.long)
    out = emb(ids)
    assert out.shape == (2, 0, 64)


def test_device_compatibility() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = OdysseyEmbedding(
        EmbeddingConfig(
            vocab_size=32,
            hidden_size=8,
            device=device,
            dtype="float32",
        )
    )
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    out = emb(ids)
    assert out.device.type == torch.device(device).type
    assert out.shape == (1, 3, 8)
