"""Phase 0 smoke tests: imports, config loading, and repository health."""

from __future__ import annotations

from pathlib import Path

import yaml

import evaluation
import model
import tokenizer
import training
from odyssey import __version__
from odyssey.config import DEFAULT_CONFIG_PATH, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version() -> None:
    assert __version__ == "0.3.0"


def test_package_imports() -> None:
    assert model.__doc__
    assert tokenizer.__doc__
    assert training.__doc__
    assert evaluation.__doc__


def test_default_config_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()


def test_config_loads() -> None:
    config = load_config()
    assert config["model"]["name"] == "odyssey-tiny"
    assert config["model"]["vocabulary_size"] == 32000
    assert config["model"]["context_length"] == 2048
    assert config["training"]["learning_rate"] == 3.0e-4
    assert config["training"]["optimizer"] == "adamw"
    assert config["training"]["batch_size"] == 8
    assert config["experiment"]["seed"] == 42
    assert config["tokenizer"]["path"] == "assets/tokenizer/bpe/odyssey.model"
    assert config["tokenizer"]["type"] == "bpe"
    assert config["tokenizer"]["config"] == "configs/tokenizer.yaml"
    assert config["experiment"]["id"] == "ODY-0003"
    assert config["model"]["embedding"]["init_strategy"] == "xavier_uniform"


def test_config_yaml_parses_directly() -> None:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict)
    assert "model" in raw


def test_required_directories_exist() -> None:
    required = [
        "configs",
        "datasets/raw",
        "datasets/processed",
        "docs/architecture",
        "docs/training",
        "docs/tokenizer",
        "docs/evaluation",
        "papers",
        "experiments",
        "model",
        "math",
        "spec",
        "tokenizer",
        "training",
        "evaluation",
        "tests",
        "assets",
        "scripts",
        ".github",
    ]
    for relative in required:
        assert (REPO_ROOT / relative).is_dir(), f"missing directory: {relative}"


def test_required_docs_exist() -> None:
    required_files = [
        "README.md",
        # AGENTS.md is intentionally gitignored (local agent instructions only).
        "ROADMAP.md",
        "MODEL_CARD.md",
        "CHANGELOG.md",
        "PAPERS.md",
        "EXPERIMENTS.md",
        "RESEARCH.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        "configs/default.yaml",
        "configs/tokenizer.yaml",
        "configs/tokenizer_sentencepiece.yaml",
        "experiments/README.md",
        "docs/tokenizer/architecture.md",
        "papers/sentencepiece.md",
        "papers/bpe.md",
        "papers/gpt2-tokenizer.md",
        "papers/tiktoken.md",
        "papers/transformer_embeddings.md",
        "papers/word2vec.md",
        "papers/glove.md",
        "docs/architecture/embeddings.md",
        "math/README.md",
        "math/embeddings.md",
        "spec/README.md",
        "spec/VERSION",
        "spec/architecture.md",
        "spec/weight_layout.md",
        "spec/gguf_mapping.md",
        "spec/runtime_contract.md",
        "configs/embedding.yaml",
        "tokenizer/README.md",
        "tokenizer/docs/bpe.md",
    ]
    for relative in required_files:
        assert (REPO_ROOT / relative).is_file(), f"missing file: {relative}"
