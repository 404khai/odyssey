"""Merge table tests."""

from __future__ import annotations

from odyssey_tokenizer.merges import MergeTable


def test_merge_table_ranks_are_insertion_order() -> None:
    table = MergeTable()
    table.add(b"e", b"r", frequency=10)
    table.add(b"er", b"s", frequency=5)
    assert table.rank_of(b"e", b"r") == 0
    assert table.rank_of(b"er", b"s") == 1
    assert table.merges[1].merged == b"ers"


def test_merge_table_roundtrip_lines() -> None:
    table = MergeTable()
    table.add(b"t", b"h")
    table.add(b"th", b"e")
    restored = MergeTable.from_lines(table.to_lines())
    assert len(restored) == 2
    assert restored.rank_of(b"t", b"h") == 0
    assert restored.merges[1].merged == b"the"
