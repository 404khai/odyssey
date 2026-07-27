#!/usr/bin/env python3
"""Train the Odyssey SentencePiece reference tokenizer.

Example:
    python scripts/train.py \\
        --input datasets/raw/sample.txt \\
        --vocab-size 32000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from odyssey.config import REPO_ROOT
from tokenizer.sentencepiece.config import load_tokenizer_config
from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Odyssey SentencePiece tokenizer (Phase 1 reference)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a newline-delimited training corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size override (default: configs/tokenizer.yaml).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "tokenizer.yaml",
        help="Tokenizer YAML config path.",
    )
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=None,
        help="Output prefix for .model/.vocab (default: config paths.model_prefix).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip Odyssey normalization before SentencePiece training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_tokenizer_config(args.config)
    tokenizer = OdysseySentencePieceTokenizer(config)

    result = tokenizer.train(
        args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        normalize_corpus=not args.no_normalize,
    )

    sample = [
        "Build authentication API",
        "Hello world",
        "The engineer planned the migration carefully.",
    ]
    stats = tokenizer.compute_stats(sample)

    print("Training complete")
    print(f"  model:      {result.model_path}")
    print(f"  vocab:      {result.vocab_path}")
    print(f"  metadata:   {result.metadata_path}")
    print(f"  vocab_size: {result.vocab_size}")
    print(f"  seconds:    {result.training_seconds:.3f}")
    print(f"  corpus:     {result.corpus_lines} lines")
    print("Stats (sample texts)")
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
