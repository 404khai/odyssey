#!/usr/bin/env python3
"""Download a TinyStories sample corpus for tokenizer training (ODY-0001)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from odyssey.config import REPO_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyStories sample corpus.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw" / "sample.txt",
        help="Output corpus path.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=5000,
        help="Number of TinyStories examples to download.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("datasets package is required") from exc

    print(f"Downloading TinyStories sample ({args.max_stories} stories)...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            # Keep stories as single lines for SentencePiece line-based training.
            handle.write(text.replace("\n", " ").strip() + "\n")
            written += 1
            if written >= args.max_stories:
                break

    print(f"Wrote {written} stories → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
