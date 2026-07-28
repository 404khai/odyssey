"""Shared fixtures for Odyssey BPE tokenizer tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from odyssey_tokenizer import BPEConfig, OdysseyTokenizer, load_bpe_config


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    stories = [
        "Once upon a time a little girl named Lily loved building APIs.",
        "Tom wanted to learn authentication, authorization, and databases.",
        "The engineer planned the migration carefully before writing code.",
        "Build authentication API for the software architecture review.",
        "Hello world from the Odyssey research tokenizer pipeline.",
        "She reasoned about tradeoffs, testing strategy, and deployment risks.",
        "Unicode café naïve — punctuation, numbers 12345, stay useful.",
        "The assistant helped the user design a REST service with clear phases.",
        "Running runner runs; engineering engineers engineered systems daily.",
        "System prompts, user turns, and assistant replies need special tokens.",
        "Think before you build. Tool calls should be explicit and safe.",
    ]
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(stories * 20) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def bpe_config(tmp_path: Path) -> BPEConfig:
    base = load_bpe_config()
    data = base.to_dict()
    model_dir = tmp_path / "odyssey.model"
    data["vocab_size"] = 400  # specials + 256 bytes + ~134 merges
    data["min_frequency"] = 2
    data["training"]["max_lines"] = 0
    data["paths"] = {
        "model_dir": str(model_dir),
        "merges_file": str(model_dir / "merges.txt"),
        "vocab_file": str(model_dir / "vocab.json"),
        "metadata_file": str(model_dir / "metadata.json"),
        "visualization_dir": str(tmp_path / "viz"),
    }
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_bpe_config(config_path)


@pytest.fixture
def trained_tokenizer(tiny_corpus: Path, bpe_config: BPEConfig) -> OdysseyTokenizer:
    tokenizer, _ = OdysseyTokenizer.train(tiny_corpus, config=bpe_config)
    return tokenizer
