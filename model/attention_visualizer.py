"""Attention heatmaps and causal-mask diagrams for education / assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.attention import OdysseyAttention
from model.causal_mask import make_causal_mask
from model.config import AttentionConfig
from odyssey.config import REPO_ROOT

DEFAULT_ASSET_DIR = REPO_ROOT / "assets" / "attention"


def plot_causal_mask(
    seq_len: int = 16,
    *,
    output_path: Path | str | None = None,
) -> Path:
    path = Path(output_path) if output_path else DEFAULT_ASSET_DIR / "causal_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = make_causal_mask(seq_len).numpy()
    # Map -inf → 0 (blocked), 0 → 1 (allowed) for visualization.
    vis = np.isfinite(mask).astype(np.float32)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(vis, cmap="Blues", vmin=0, vmax=1, origin="upper")
    ax.set_xlabel("key position t")
    ax.set_ylabel("query position s")
    ax.set_title(f"Causal mask ({seq_len}×{seq_len})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_attention_heatmap(
    *,
    config: AttentionConfig | None = None,
    seq_len: int = 12,
    head: int = 0,
    seed: int = 0,
    output_path: Path | str | None = None,
) -> Path:
    """Forward a toy batch and plot one head's attention weights."""
    cfg = config or AttentionConfig(
        num_heads=4, num_kv_heads=2, head_dim=8, dropout=0.0
    )
    path = (
        Path(output_path)
        if output_path
        else DEFAULT_ASSET_DIR / "attention_heatmap.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    attn = OdysseyAttention(cfg)
    attn.eval()
    x = torch.randn(1, seq_len, cfg.hidden_size)
    with torch.no_grad():
        _, weights = attn(x, return_weights=True)
    # weights: (B, H, S, S)
    mat = weights[0, head].float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=max(mat.max(), 1e-6), origin="upper")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_title(
        f"Attention head {head} " f"(H={cfg.num_heads}, H_kv={cfg.num_kv_heads})"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_gqa_grouping(
    num_heads: int = 12,
    num_kv_heads: int = 4,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Diagram which query heads share each KV head."""
    path = Path(output_path) if output_path else DEFAULT_ASSET_DIR / "gqa_grouping.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = num_heads // num_kv_heads
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, num_kv_heads))
    for h in range(num_heads):
        kv = h // groups
        ax.barh(0, 1, left=h, color=colors[kv], edgecolor="white", height=0.6)
        ax.text(
            h + 0.5, 0, f"Q{h}", ha="center", va="center", fontsize=8, color="white"
        )
    ax.set_xlim(0, num_heads)
    ax.set_yticks([])
    ax.set_xlabel("query head index")
    ax.set_title(f"GQA grouping: {num_heads} Q heads → {num_kv_heads} KV heads")
    for kv in range(num_kv_heads):
        ax.annotate(
            f"KV{kv}",
            xy=(kv * groups + groups / 2, 0.45),
            ha="center",
            fontsize=9,
            color=colors[kv],
        )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def generate_all_assets(asset_dir: Path | str | None = None) -> list[Path]:
    out = Path(asset_dir) if asset_dir else DEFAULT_ASSET_DIR
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_causal_mask(output_path=out / "causal_mask.png"),
        plot_attention_heatmap(output_path=out / "attention_heatmap.png"),
        plot_gqa_grouping(output_path=out / "gqa_grouping.png"),
    ]


if __name__ == "__main__":
    for p in generate_all_assets():
        print(p)
