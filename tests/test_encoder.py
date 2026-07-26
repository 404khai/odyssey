"""Encoder / round-trip tests for the SentencePiece tokenizer."""

from __future__ import annotations

from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def test_encode_returns_ids(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(token_id, int) for token_id in ids)


def test_encode_decode_roundtrip(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    text = "Build authentication API"
    normalized = trained_tokenizer.normalizer.normalize(text)
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    assert decoded == normalized


def test_unknown_words_do_not_crash(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    weird = "xyzzyquux_unknown_token_foobar_999"
    ids = trained_tokenizer.encode(weird)
    decoded = trained_tokenizer.decode(ids)
    assert isinstance(ids, list)
    assert isinstance(decoded, str)


def test_add_bos_eos(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello", add_bos=True, add_eos=True)
    assert ids[0] == trained_tokenizer.config.bos_id
    assert ids[-1] == trained_tokenizer.config.eos_id


def test_unicode_encoding(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    text = "café naïve 日本語"
    ids = trained_tokenizer.encode(text)
    # Accented Latin + Japanese appear in the tiny training corpus.
    assert trained_tokenizer.decode(ids) == trained_tokenizer.normalizer.normalize(text)
    assert trained_tokenizer.config.unk_id not in ids


def test_whitespace_normalization_roundtrip(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    # Odyssey normalizer preserves paragraph breaks...
    normalized = trained_tokenizer.normalizer.normalize("Hello    world\n\n\nagain")
    assert normalized == "Hello world\n\nagain"

    # ...while SentencePiece's internal normalizer collapses them during encode.
    # Round-trip fidelity is guaranteed for single-line normalized prose.
    text = "Hello    world again"
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    assert decoded == trained_tokenizer.normalizer.normalize(text)


def test_inspect_pipeline(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    result = trained_tokenizer.inspect("Build authentication API")
    assert result.pieces
    assert result.ids
    assert result.decoded_text
    rendered = result.render()
    assert "Input" in rendered
    assert "Tokens" in rendered
    assert "IDs" in rendered
