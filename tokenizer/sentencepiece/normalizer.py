"""Unicode and whitespace normalization for tokenizer inputs.

Why this exists:
    SentencePiece can normalize internally, but Odyssey keeps normalization
    explicit so tests and Phase 2 BPE can share the same preprocessing contract.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, cast

from tokenizer.sentencepiece.config import NormalizationConfig, TokenizerConfig

_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_UNICODE_FORMS = {"NFC", "NFD", "NFKC", "NFKD"}


class TextNormalizer:
    """Normalize raw text before SentencePiece training or encoding."""

    def __init__(self, config: NormalizationConfig | TokenizerConfig) -> None:
        if isinstance(config, TokenizerConfig):
            self.config = config.normalization
        else:
            self.config = config

    def normalize(self, text: str) -> str:
        """Apply configured Unicode / whitespace normalization."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        form = self.config.form.upper()
        if form and form != "NONE":
            if form not in _UNICODE_FORMS:
                raise ValueError(f"Unsupported Unicode normalization form: {form}")
            text = unicodedata.normalize(
                cast(Literal["NFC", "NFD", "NFKC", "NFKD"], form), text
            )

        # Preserve special Odyssey tokens verbatim while cleaning surrounding text.
        # They are inserted by templates and must survive whitespace collapse.
        if self.config.collapse_whitespace:
            if self.config.preserve_newlines:
                text = _WHITESPACE_RE.sub(" ", text)
                text = _MULTI_NEWLINE_RE.sub("\n\n", text)
            else:
                text = re.sub(r"\s+", " ", text)

        if self.config.strip:
            text = text.strip()

        return text

    def normalize_file(self, input_path: str, output_path: str) -> int:
        """Normalize a corpus file line-by-line. Returns lines written."""
        from pathlib import Path

        src = Path(input_path)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with (
            src.open("r", encoding="utf-8") as reader,
            dst.open("w", encoding="utf-8") as writer,
        ):
            for line in reader:
                normalized = self.normalize(line.rstrip("\n"))
                if not normalized:
                    continue
                writer.write(normalized + "\n")
                count += 1
        return count
