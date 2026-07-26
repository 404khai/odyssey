"""Token ID → text decoding via SentencePiece."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sentencepiece as spm

    from tokenizer.sentencepiece.config import TokenizerConfig


class TokenizerDecoder:
    """Decode token IDs or pieces back into text."""

    def __init__(
        self,
        processor: spm.SentencePieceProcessor,
        config: TokenizerConfig,
    ) -> None:
        self.processor = processor
        self.config = config

    def decode(self, ids: list[int], *, skip_special_ids: bool = False) -> str:
        """Decode token IDs into a UTF-8 string."""
        payload = ids
        if skip_special_ids:
            special = {
                self.config.pad_id,
                self.config.bos_id,
                self.config.eos_id,
            }
            payload = [token_id for token_id in ids if token_id not in special]
        return str(self.processor.decode(payload))

    def decode_pieces(self, pieces: list[str]) -> str:
        """Decode SentencePiece pieces into text."""
        return str(self.processor.decode(pieces))

    def decode_batch(
        self, batch: list[list[int]], *, skip_special_ids: bool = False
    ) -> list[str]:
        """Decode multiple ID sequences."""
        return [self.decode(ids, skip_special_ids=skip_special_ids) for ids in batch]
