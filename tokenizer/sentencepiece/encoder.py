"""Text → token ID encoding via SentencePiece."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenizer.sentencepiece.normalizer import TextNormalizer

if TYPE_CHECKING:
    import sentencepiece as spm

    from tokenizer.sentencepiece.config import TokenizerConfig


class TokenizerEncoder:
    """Encode normalized text into token IDs or pieces."""

    def __init__(
        self,
        processor: spm.SentencePieceProcessor,
        config: TokenizerConfig,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self.processor = processor
        self.config = config
        self.normalizer = normalizer or TextNormalizer(config)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        """Encode text to a list of token IDs."""
        payload = self.normalizer.normalize(text) if normalize else text
        ids = list(self.processor.encode(payload, out_type=int))
        if add_bos:
            ids = [self.config.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.config.eos_id]
        return ids

    def encode_as_pieces(self, text: str, *, normalize: bool = True) -> list[str]:
        """Encode text to SentencePiece surface forms (includes ▁)."""
        payload = self.normalizer.normalize(text) if normalize else text
        return list(self.processor.encode(payload, out_type=str))

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[list[int]]:
        """Encode multiple strings."""
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, normalize=normalize)
            for text in texts
        ]
