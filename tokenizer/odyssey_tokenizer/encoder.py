"""BPE encoder: text → token IDs."""

from __future__ import annotations

from odyssey_tokenizer.merges import MergeTable
from odyssey_tokenizer.normalizer import TextNormalizer
from odyssey_tokenizer.vocabulary import Vocabulary


def _apply_merges(symbols: list[bytes], merges: MergeTable) -> list[bytes]:
    """Greedily apply lowest-rank merges until no mergeable pair remains."""
    if len(symbols) < 2 or not merges.merges:
        return symbols

    while True:
        # Find the adjacent pair with the best (lowest) merge rank.
        best_rank: int | None = None
        best_index: int | None = None
        for index in range(len(symbols) - 1):
            rank = merges.rank_of(symbols[index], symbols[index + 1])
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_index = index
        if best_index is None or best_rank is None:
            break
        left = symbols[best_index]
        right = symbols[best_index + 1]
        symbols = [
            *symbols[:best_index],
            left + right,
            *symbols[best_index + 2 :],
        ]
    return symbols


class BPEEncoder:
    """Encode normalized UTF-8 text with a trained merge table."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        merges: MergeTable,
        normalizer: TextNormalizer,
        *,
        unk_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        self.vocabulary = vocabulary
        self.merges = merges
        self.normalizer = normalizer
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        payload = self.normalizer.normalize(text) if normalize else text
        raw = payload.encode("utf-8")
        symbols = [bytes([value]) for value in raw]
        merged = _apply_merges(symbols, self.merges)
        ids = [self.vocabulary.get_id(symbol, default=self.unk_id) for symbol in merged]
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_as_tokens(self, text: str, *, normalize: bool = True) -> list[bytes]:
        payload = self.normalizer.normalize(text) if normalize else text
        raw = payload.encode("utf-8")
        symbols = [bytes([value]) for value in raw]
        return _apply_merges(symbols, self.merges)
