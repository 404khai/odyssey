"""Vocabulary data structures for Odyssey byte-level BPE."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class Vocabulary:
    """Bidirectional token ↔ ID map.

    Token values are stored as UTF-8 / latin-1 ``bytes``.
    Special tokens use their UTF-8 surface forms (e.g. ``b'<pad>'``).
    Byte tokens use single-byte values ``bytes([0])`` … ``bytes([255])``.
    Merged tokens are concatenations of their children.
    """

    token_to_id: dict[bytes, int] = field(default_factory=dict)
    id_to_token: dict[int, bytes] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def add(self, token: bytes, token_id: int | None = None) -> int:
        """Insert a token, returning its ID. Idempotent for existing tokens."""
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        assigned = len(self.token_to_id) if token_id is None else token_id
        if assigned in self.id_to_token:
            raise ValueError(f"Token ID {assigned} already assigned")
        self.token_to_id[token] = assigned
        self.id_to_token[assigned] = token
        return assigned

    def get_id(self, token: bytes, default: int | None = None) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if default is not None:
            return default
        raise KeyError(token)

    def get_token(self, token_id: int) -> bytes:
        return self.id_to_token[token_id]

    def contains(self, token: bytes) -> bool:
        return token in self.token_to_id

    def tokens(self) -> Iterable[bytes]:
        return self.token_to_id.keys()

    @classmethod
    def from_specials_and_bytes(cls, special_tokens: list[str]) -> Vocabulary:
        """Build the initial vocabulary: specials then all 256 bytes."""
        vocab = cls()
        for surface in special_tokens:
            vocab.add(surface.encode("utf-8"))
        for byte_value in range(256):
            vocab.add(bytes([byte_value]))
        return vocab

    def to_serializable(self) -> dict[str, int]:
        """JSON-friendly map using latin-1 round-trip for arbitrary bytes."""
        return {
            token.decode("latin-1"): token_id
            for token, token_id in self.token_to_id.items()
        }

    @classmethod
    def from_serializable(cls, payload: dict[str, int]) -> Vocabulary:
        vocab = cls()
        for surface, token_id in sorted(payload.items(), key=lambda item: item[1]):
            vocab.add(surface.encode("latin-1"), token_id=token_id)
        return vocab
