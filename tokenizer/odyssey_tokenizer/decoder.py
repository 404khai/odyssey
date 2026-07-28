"""BPE decoder: token IDs → text."""

from __future__ import annotations

from odyssey_tokenizer.vocabulary import Vocabulary


class BPEDecoder:
    """Decode token IDs back into UTF-8 text."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        special_token_bytes: set[bytes],
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        self.vocabulary = vocabulary
        self.special_token_bytes = special_token_bytes
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def decode(self, ids: list[int], *, skip_special_ids: bool = False) -> str:
        pieces: list[bytes] = []
        for token_id in ids:
            if skip_special_ids and token_id in {self.pad_id, self.bos_id, self.eos_id}:
                continue
            token = self.vocabulary.get_token(token_id)
            if token in self.special_token_bytes:
                # Control tokens are not part of the UTF-8 byte stream.
                if not skip_special_ids:
                    pieces.append(token)
                continue
            pieces.append(token)
        raw = b"".join(pieces)
        # If specials were kept, they are UTF-8 text mixed with byte tokens.
        # Prefer strict UTF-8 for pure byte payloads; fall back to replace.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")
