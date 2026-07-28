"""Encoding tests for OdysseyTokenizer."""

from __future__ import annotations

from odyssey_tokenizer import OdysseyTokenizer


def test_encode_returns_ids(trained_tokenizer: OdysseyTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello world")
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_add_bos_eos(trained_tokenizer: OdysseyTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello", add_bos=True, add_eos=True)
    assert ids[0] == trained_tokenizer.bos_id
    assert ids[-1] == trained_tokenizer.eos_id


def test_inspect_shows_compression(trained_tokenizer: OdysseyTokenizer) -> None:
    result = trained_tokenizer.inspect("Build authentication API")
    assert result.token_count > 0
    assert result.characters >= result.token_count
    assert "Compression" in result.render()
