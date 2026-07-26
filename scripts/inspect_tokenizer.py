#!/usr/bin/env python3
"""Inspect Odyssey SentencePiece tokenization.

Example:
    python scripts/inspect_tokenizer.py \\
        --model assets/tokenizer/odyssey.model \\
        --text "Build authentication API"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from odyssey.config import REPO_ROOT
from tokenizer.sentencepiece.config import load_tokenizer_config
from tokenizer.sentencepiece.special_tokens import describe_special_tokens
from tokenizer.sentencepiece.tokenizer import OdysseySentencePieceTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Odyssey tokenizer output.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .model file (default: configs/tokenizer.yaml paths.model_file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "tokenizer.yaml",
        help="Tokenizer YAML config path.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to tokenize.",
    )
    parser.add_argument(
        "--show-specials",
        action="store_true",
        help="Also print the reserved special-token table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_tokenizer_config(args.config)
    tokenizer = OdysseySentencePieceTokenizer(config)
    tokenizer.load(args.model)

    inspection = tokenizer.inspect(args.text)
    print(inspection.render())
    print()
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"special_token_ids: {tokenizer.special_token_ids()}")

    if args.show_specials:
        print()
        print(describe_special_tokens(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
