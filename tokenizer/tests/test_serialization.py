"""Serialization / reload regression tests."""

from __future__ import annotations

from pathlib import Path

from odyssey_tokenizer import OdysseyTokenizer
from odyssey_tokenizer.config import BPEConfig


def test_save_load_identical_encoding(
    tiny_corpus: Path, bpe_config: BPEConfig, tmp_path: Path
) -> None:
    tokenizer, _ = OdysseyTokenizer.train(tiny_corpus, config=bpe_config)
    text = "Build authentication API"
    original = tokenizer.encode(text)

    model_dir = tmp_path / "exported.model"
    tokenizer.save(model_dir)
    reloaded = OdysseyTokenizer.load(model_dir)

    assert reloaded.encode(text) == original
    assert reloaded.decode(original) == tokenizer.decode(original)
    assert reloaded.vocab_size == tokenizer.vocab_size
