"""Decoder tests for the SentencePiece tokenizer."""

from __future__ import annotations

from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def test_decode_matches_normalized_text(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    text = "Hello world"
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids) == trained_tokenizer.normalizer.normalize(text)


def test_skip_special_ids(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    ids = trained_tokenizer.encode("Hello", add_bos=True, add_eos=True)
    decoded = trained_tokenizer.decode(ids, skip_special_ids=True)
    assert trained_tokenizer.config.special_tokens["bos"] not in decoded
    assert trained_tokenizer.config.special_tokens["eos"] not in decoded


def test_id_to_piece_roundtrip(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    ids = trained_tokenizer.encode("Build authentication API")
    pieces = [trained_tokenizer.id_to_piece(token_id) for token_id in ids]
    assert pieces == trained_tokenizer.encode_as_pieces("Build authentication API")
