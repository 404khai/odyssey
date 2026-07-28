"""Inspection and visualization helpers for Odyssey embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.embeddings import OdysseyEmbedding
from odyssey.config import REPO_ROOT

DEFAULT_ASSET_DIR = REPO_ROOT / "assets" / "embeddings"


def nearest_neighbors(
    embedding: OdysseyEmbedding,
    token_id: int,
    *,
    k: int = 8,
) -> list[tuple[int, float]]:
    """Return ``k`` nearest token IDs by cosine similarity (excluding self)."""
    weight = embedding.weight.detach().float()
    if token_id < 0 or token_id >= weight.shape[0]:
        raise ValueError(f"token_id {token_id} out of range")
    query = weight[token_id]
    norms = torch.linalg.vector_norm(weight, dim=1).clamp_min(1e-12)
    q_norm = torch.linalg.vector_norm(query).clamp_min(1e-12)
    sims = (weight @ query) / (norms * q_norm)
    sims[token_id] = float("-inf")
    topk = torch.topk(sims, k=min(k, weight.shape[0] - 1))
    return [(int(i), float(s)) for i, s in zip(topk.indices, topk.values, strict=True)]


def plot_embedding_matrix(
    embedding: OdysseyEmbedding,
    *,
    max_rows: int = 64,
    max_cols: int = 64,
    output_path: Path | str | None = None,
    title: str | None = None,
) -> Path:
    """Save a heatmap of a truncated embedding matrix slice."""
    path = (
        Path(output_path) if output_path else DEFAULT_ASSET_DIR / "matrix_heatmap.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    weight = embedding.weight.detach().float().cpu().numpy()
    rows = min(max_rows, weight.shape[0])
    cols = min(max_cols, weight.shape[1])
    slice_ = weight[:rows, :cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(slice_, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("hidden dim")
    ax.set_ylabel("token id")
    ax.set_title(title or f"Embedding slice [{rows} × {cols}]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_row_norms(
    embedding: OdysseyEmbedding,
    *,
    output_path: Path | str | None = None,
    max_tokens: int | None = 2048,
) -> Path:
    """Histogram of L2 norms of embedding rows."""
    path = Path(output_path) if output_path else DEFAULT_ASSET_DIR / "row_norms.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    weight = embedding.weight.detach().float()
    if max_tokens is not None:
        weight = weight[:max_tokens]
    norms = torch.linalg.vector_norm(weight, dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(norms, bins=40, color="#2c5f6e", edgecolor="white")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("count")
    ax.set_title("Embedding row norms")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarize_neighbors(
    embedding: OdysseyEmbedding,
    token_ids: list[int],
    *,
    k: int = 5,
) -> str:
    """Format nearest-neighbor tables for a list of probe token IDs."""
    lines: list[str] = []
    for token_id in token_ids:
        lines.append(f"token_id={token_id}")
        for neighbor_id, score in nearest_neighbors(embedding, token_id, k=k):
            lines.append(f"  → {neighbor_id:>6d}  cos={score:+.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_embedding_preview(
    embedding: OdysseyEmbedding,
    *,
    asset_dir: Path | str | None = None,
    probe_ids: list[int] | None = None,
) -> dict[str, Path]:
    """Write standard Phase-3 visualization artifacts under ``assets/embeddings/``."""
    out = Path(asset_dir) if asset_dir else DEFAULT_ASSET_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "matrix": plot_embedding_matrix(
            embedding, output_path=out / "matrix_heatmap.png"
        ),
        "norms": plot_row_norms(embedding, output_path=out / "row_norms.png"),
    }

    probes = probe_ids if probe_ids is not None else [0, 1, 2, 7, 128]
    probes = [i for i in probes if 0 <= i < embedding.config.vocab_size]
    neighbors_path = out / "nearest_neighbors.txt"
    neighbors_path.write_text(
        summarize_neighbors(embedding, probes, k=5), encoding="utf-8"
    )
    paths["neighbors"] = neighbors_path

    # Tiny numeric snapshot for regression / docs (not a full dump).
    snapshot = embedding.weight.detach().float()[:8, :8].cpu().numpy()
    np.save(out / "weight_preview_8x8.npy", snapshot)
    paths["preview"] = out / "weight_preview_8x8.npy"
    return paths
