"""Decoding tests for OdysseyTokenizer."""

from __future__ import annotations

from odyssey_tokenizer import OdysseyTokenizer


def test_decode_skips_specials(trained_tokenizer: OdysseyTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello", add_bos=True, add_eos=True)
    decoded = trained_tokenizer.decode(ids, skip_special_ids=True)
    assert "<bos>" not in decoded
    assert "<eos>" not in decoded


def test_decode_batch_roundtrip(trained_tokenizer: OdysseyTokenizer) -> None:
    texts = ["Hello world", "Build authentication API"]
    for text in texts:
        ids = trained_tokenizer.encode(text)
        assert trained_tokenizer.decode(ids) == trained_tokenizer.normalizer.normalize(
            text
        )
