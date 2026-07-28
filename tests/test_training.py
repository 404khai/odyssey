"""Training, save/load, config, and statistics tests."""

from __future__ import annotations

from pathlib import Path

from tokenizer.sentencepiece.config import TokenizerConfig, load_tokenizer_config
from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def test_load_tokenizer_config(repo_root: Path) -> None:
    config = load_tokenizer_config(
        repo_root / "configs" / "tokenizer_sentencepiece.yaml"
    )
    assert config.vocab_size == 32000
    assert config.model_type == "unigram"
    assert config.character_coverage == 0.9995
    assert config.pad_id == 0
    assert config.bos_id == 1
    assert config.eos_id == 2
    assert config.unk_id == 3


def test_train_save_load_identical_outputs(
    tiny_corpus: Path,
    tokenizer_config: TokenizerConfig,
    tmp_path: Path,
) -> None:
    tokenizer = OdysseySentencePieceTokenizer(tokenizer_config)
    result = tokenizer.train(tiny_corpus, vocab_size=256)

    assert result.model_path.is_file()
    assert result.vocab_path.is_file()
    assert result.metadata_path.is_file()
    assert result.vocab_size > 0

    text = "Build authentication API"
    original_ids = tokenizer.encode(text)

    save_prefix = tmp_path / "exported" / "odyssey"
    model_path, vocab_path = tokenizer.save(save_prefix)
    assert model_path.is_file()
    assert vocab_path.is_file()

    reloaded = OdysseySentencePieceTokenizer(tokenizer_config)
    reloaded.load(model_path)
    assert reloaded.encode(text) == original_ids
    assert reloaded.decode(original_ids) == tokenizer.decode(original_ids)


def test_vocab_size_positive(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    assert trained_tokenizer.vocab_size >= 16


def test_compute_stats(trained_tokenizer: OdysseySentencePieceTokenizer) -> None:
    stats = trained_tokenizer.compute_stats(
        [
            "Hello world",
            "Build authentication API",
            "The engineer planned carefully.",
        ]
    )
    assert stats.vocab_size == trained_tokenizer.vocab_size
    assert stats.compression_ratio > 0
    assert stats.average_token_length > 0
    assert stats.encoding_tokens_per_second > 0
    assert stats.decoding_tokens_per_second > 0
    assert 0.0 <= stats.unknown_token_frequency <= 1.0


def test_normalizer_file(tmp_path: Path, tokenizer_config: TokenizerConfig) -> None:
    from tokenizer.sentencepiece.normalizer import TextNormalizer

    src = tmp_path / "raw.txt"
    dst = tmp_path / "norm.txt"
    src.write_text("Hello    world\n\n\nNext   line\n", encoding="utf-8")
    count = TextNormalizer(tokenizer_config).normalize_file(str(src), str(dst))
    assert count == 2
    assert "Hello world" in dst.read_text(encoding="utf-8")
