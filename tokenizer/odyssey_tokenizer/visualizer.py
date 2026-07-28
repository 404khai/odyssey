"""Educational visualizations for BPE merges and compression."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from odyssey_tokenizer.merges import MergeTable


def render_merge_steps(merges: MergeTable, *, limit: int = 20) -> str:
    """ASCII visualization of early merge operations."""
    lines = ["Merge Visualization", "===================", ""]
    for merge in merges.merges[:limit]:
        left = merge.left.decode("latin-1", errors="replace")
        right = merge.right.decode("latin-1", errors="replace")
        merged = merge.merged.decode("latin-1", errors="replace")
        lines.extend(
            [
                f"#{merge.rank}  freq={merge.frequency}",
                f"  {left!r}",
                "  +",
                f"  {right!r}",
                "  ↓",
                f"  {merged!r}",
                "",
            ]
        )
    if len(merges) > limit:
        lines.append(f"... ({len(merges) - limit} more merges)")
    return "\n".join(lines)


def write_merge_visualization_png(
    merges: MergeTable,
    output_path: Path,
    *,
    limit: int = 30,
) -> Path:
    """Bar chart of early merge frequencies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = merges.merges[:limit]
    labels = [
        f"{m.left.decode('latin-1', errors='replace')}+"
        f"{m.right.decode('latin-1', errors='replace')}"
        for m in subset
    ]
    values = [m.frequency for m in subset]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.bar(range(len(values)), values, color="#2f5d50")
    axis.set_title("Odyssey BPE — early merge frequencies")
    axis.set_xlabel("Merge rank")
    axis.set_ylabel("Frequency at merge time")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def write_compression_graph_png(
    character_counts: Sequence[int],
    token_counts: Sequence[int],
    output_path: Path,
) -> Path:
    """Plot characters vs tokens for sample texts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(character_counts, token_counts, color="#2f5d50", alpha=0.8)
    if character_counts:
        max_x = max(character_counts)
        axis.plot([0, max_x], [0, max_x], linestyle="--", color="#999999", label="1:1")
    axis.set_xlabel("Characters")
    axis.set_ylabel("Tokens")
    axis.set_title("Odyssey BPE compression")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
