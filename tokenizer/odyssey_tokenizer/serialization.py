"""Serialize / deserialize OdysseyTokenizer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odyssey_tokenizer.config import BPEConfig
from odyssey_tokenizer.merges import MergeTable
from odyssey_tokenizer.vocabulary import Vocabulary

MODEL_FORMAT_VERSION = 1


def save_tokenizer_bundle(
    directory: Path,
    *,
    vocabulary: Vocabulary,
    merges: MergeTable,
    config: BPEConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a loadable tokenizer directory (``*.model``)."""
    directory.mkdir(parents=True, exist_ok=True)

    vocab_path = directory / "vocab.json"
    merges_path = directory / "merges.txt"
    config_path = directory / "config.json"
    meta_path = directory / "metadata.json"

    vocab_path.write_text(
        json.dumps(vocabulary.to_serializable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merges_path.write_text("\n".join(merges.to_lines()) + "\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = {
        "format": "odyssey-bpe",
        "version": MODEL_FORMAT_VERSION,
        "vocab_size": len(vocabulary),
        "merge_count": len(merges),
        "config": config.to_dict(),
    }
    if metadata:
        payload.update(metadata)
    meta_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def load_tokenizer_bundle(
    directory: Path,
) -> tuple[Vocabulary, MergeTable, dict[str, Any], dict[str, Any]]:
    """Load vocabulary, merges, config dict, and metadata from a model dir."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Tokenizer model directory not found: {directory}")

    vocab_path = directory / "vocab.json"
    merges_path = directory / "merges.txt"
    config_path = directory / "config.json"
    meta_path = directory / "metadata.json"

    for required in (vocab_path, merges_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing tokenizer artifact: {required}")

    vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab_payload, dict):
        raise ValueError("vocab.json must be an object")
    vocabulary = Vocabulary.from_serializable(
        {str(key): int(value) for key, value in vocab_payload.items()}
    )

    merges = MergeTable.from_lines(merges_path.read_text(encoding="utf-8").splitlines())
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )
    if not isinstance(config_payload, dict):
        raise ValueError("config.json must be an object")
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must be an object")
    return vocabulary, merges, config_payload, metadata
