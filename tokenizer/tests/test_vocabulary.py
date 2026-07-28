"""Vocabulary builder tests."""

from __future__ import annotations

from odyssey_tokenizer.vocabulary import Vocabulary


def test_initial_vocabulary_has_specials_and_bytes() -> None:
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = Vocabulary.from_specials_and_bytes(specials)
    assert len(vocab) == 4 + 256
    assert vocab.get_id(b"<pad>") == 0
    assert vocab.get_id(bytes([0])) == 4
    assert vocab.get_id(bytes([255])) == 259


def test_vocabulary_serialization_roundtrip() -> None:
    vocab = Vocabulary.from_specials_and_bytes(["<pad>", "<unk>"])
    vocab.add(b"er")
    restored = Vocabulary.from_serializable(vocab.to_serializable())
    assert restored.get_id(b"er") == vocab.get_id(b"er")
    assert restored.get_token(0) == b"<pad>"
