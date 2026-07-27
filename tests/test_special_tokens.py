"""Special-token documentation and vocabulary presence tests."""

from __future__ import annotations

from tokenizer.sentencepiece.config import load_tokenizer_config
from tokenizer.sentencepiece.special_tokens import (
    build_special_tokens,
    describe_special_tokens,
    special_token_surfaces,
)
from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def test_special_token_surfaces_include_chat_roles() -> None:
    config = load_tokenizer_config()
    surfaces = special_token_surfaces(config)
    for expected in (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<mask>",
        "<system>",
        "<user>",
        "<assistant>",
    ):
        assert expected in surfaces


def test_core_special_ids_match_config() -> None:
    config = load_tokenizer_config()
    tokens = {token.name: token for token in build_special_tokens(config)}
    assert tokens["pad"].token_id == config.pad_id == 0
    assert tokens["bos"].token_id == config.bos_id == 1
    assert tokens["eos"].token_id == config.eos_id == 2
    assert tokens["unk"].token_id == config.unk_id == 3


def test_describe_special_tokens_is_readable() -> None:
    text = describe_special_tokens(load_tokenizer_config())
    assert "<pad>" in text
    assert "Purpose" in text


def test_vocabulary_contains_reserved_tokens(
    trained_tokenizer: OdysseySentencePieceTokenizer,
) -> None:
    assert trained_tokenizer.has_reserved_tokens()
    ids = trained_tokenizer.special_token_ids()
    assert ids["pad"] == 0
    assert ids["bos"] == 1
    assert ids["eos"] == 2
    assert ids["unk"] == 3
    assert "mask" in ids
    assert "system" in ids
    assert "user" in ids
    assert "assistant" in ids
