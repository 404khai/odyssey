"""End-to-end BPE trainer / round-trip tests."""

from __future__ import annotations

from pathlib import Path

from odyssey_tokenizer import OdysseyTokenizer
from odyssey_tokenizer.config import BPEConfig
from odyssey_tokenizer.trainer import get_pairs


def test_train_produces_vocab_and_merges(
    tiny_corpus: Path, bpe_config: BPEConfig
) -> None:
    tokenizer, result = OdysseyTokenizer.train(tiny_corpus, config=bpe_config)
    assert result.vocab_size == tokenizer.vocab_size
    assert result.merge_count > 0
    assert tokenizer.vocab_size >= 256 + len(bpe_config.special_tokens)
    assert tokenizer.has_reserved_tokens()


def test_encode_decode_roundtrip(trained_tokenizer: OdysseyTokenizer) -> None:
    text = "Build authentication API"
    normalized = trained_tokenizer.normalizer.normalize(text)
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids) == normalized


def test_get_pairs_counts_adjacent_symbols() -> None:
    symbols = [b"A", b"B", b"A", b"B", b"A", b"C"]
    pairs = get_pairs(symbols)
    assert (b"A", b"B") in pairs
    assert (b"B", b"A") in pairs
    assert (b"A", b"C") in pairs


def test_unknown_bytes_do_not_crash(trained_tokenizer: OdysseyTokenizer) -> None:
    # Byte-level BPE can represent any UTF-8 string; decode must still succeed.
    text = "🚀✨ completely novel emoji soup"
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    assert isinstance(ids, list)
    assert isinstance(decoded, str)
