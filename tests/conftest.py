"""Shared fixtures for Odyssey tokenizer tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tokenizer.sentencepiece.config import TokenizerConfig, load_tokenizer_config
from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Create a small but diverse corpus for fast SentencePiece training."""
    stories = [
        "Once upon a time a little girl named Lily loved building APIs.",
        "Tom wanted to learn authentication, authorization, and databases.",
        "The engineer planned the migration carefully before writing code.",
        "Build authentication API for the software architecture review.",
        "Hello world from the Odyssey research tokenizer pipeline.",
        "She reasoned about tradeoffs, testing strategy, and deployment risks.",
        "Unicode café naïve — punctuation, numbers 12345, and newlines stay useful.",
        "日本語の文章もトークナイザ研究のために含めます。",
        "The assistant helped the user design a REST service with clear phases.",
        "Running runner runs; engineering engineers engineered systems daily.",
        "System prompts, user turns, and assistant replies need special tokens.",
    ]
    # Repeat to give SentencePiece enough signal for a modest vocab.
    lines = stories * 40
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tokenizer_config(tmp_path: Path) -> TokenizerConfig:
    """Tokenizer config pointing artifacts at a temporary directory."""
    base = load_tokenizer_config()
    prefix = tmp_path / "tok" / "odyssey"
    data = base.to_dict()
    data["vocab_size"] = 256
    # Tiny corpora cannot always satisfy a hard 256-piece limit.
    data["training"]["hard_vocab_limit"] = False
    data["paths"] = {
        "model_prefix": str(prefix),
        "model_file": str(prefix) + ".model",
        "vocab_file": str(prefix) + ".vocab",
        "metadata_file": str(tmp_path / "tok" / "metadata.json"),
    }
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_tokenizer_config(config_path)


@pytest.fixture
def trained_tokenizer(
    tiny_corpus: Path, tokenizer_config: TokenizerConfig
) -> OdysseySentencePieceTokenizer:
    tokenizer = OdysseySentencePieceTokenizer(tokenizer_config)
    tokenizer.train(tiny_corpus, vocab_size=256)
    return tokenizer
