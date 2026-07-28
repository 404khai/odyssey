"""Unicode / whitespace normalization for OdysseyTokenizer."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, cast

from odyssey_tokenizer.config import BPEConfig, NormalizationConfig

_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_UNICODE_FORMS = {"NFC", "NFD", "NFKC", "NFKD"}


class TextNormalizer:
    """Deterministic text cleanup before byte encoding."""

    def __init__(self, config: NormalizationConfig | BPEConfig) -> None:
        if isinstance(config, BPEConfig):
            self.config = config.normalization
            self.lowercase = config.lowercase
        else:
            self.config = config
            self.lowercase = False

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        form = self.config.form.upper()
        if form and form != "NONE":
            if form not in _UNICODE_FORMS:
                raise ValueError(f"Unsupported Unicode normalization form: {form}")
            text = unicodedata.normalize(
                cast(Literal["NFC", "NFD", "NFKC", "NFKD"], form), text
            )

        if self.lowercase:
            text = text.lower()

        if self.config.collapse_whitespace:
            if self.config.preserve_newlines:
                text = _WHITESPACE_RE.sub(" ", text)
                text = _MULTI_NEWLINE_RE.sub("\n\n", text)
            else:
                text = re.sub(r"\s+", " ", text)

        if self.config.strip:
            text = text.strip()
        return text
