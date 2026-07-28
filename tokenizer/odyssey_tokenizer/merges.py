"""Merge table for Odyssey BPE."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class Merge:
    left: bytes
    right: bytes
    merged: bytes
    rank: int
    frequency: int = 0


@dataclass(slots=True)
class MergeTable:
    """Ordered merge history + O(1) rank lookup."""

    merges: list[Merge] = field(default_factory=list)
    ranks: dict[tuple[bytes, bytes], int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.merges)

    def add(self, left: bytes, right: bytes, *, frequency: int = 0) -> Merge:
        rank = len(self.merges)
        merged = left + right
        record = Merge(
            left=left, right=right, merged=merged, rank=rank, frequency=frequency
        )
        self.merges.append(record)
        self.ranks[(left, right)] = rank
        return record

    def rank_of(self, left: bytes, right: bytes) -> int | None:
        return self.ranks.get((left, right))

    def pairs(self) -> Iterable[tuple[bytes, bytes]]:
        for merge in self.merges:
            yield merge.left, merge.right

    def to_lines(self) -> list[str]:
        """Serialize merges as hex pairs + frequency."""
        lines = ["#odyssey-bpe-merges v1 hex-freq"]
        for merge in self.merges:
            lines.append(f"{merge.left.hex()} {merge.right.hex()} {merge.frequency}")
        return lines

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> MergeTable:
        table = cls()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid merge line: {stripped!r}")
            left_s, right_s = parts[0], parts[1]
            frequency = int(parts[2]) if len(parts) >= 3 else 0
            table.add(
                bytes.fromhex(left_s),
                bytes.fromhex(right_s),
                frequency=frequency,
            )
        return table
