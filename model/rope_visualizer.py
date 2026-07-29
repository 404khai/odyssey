"""Educational visualizations for Odyssey RoPE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.config import RopeConfig
from model.rope import OdysseyRoPE
from model.rope_math import inverse_frequencies
from odyssey.config import REPO_ROOT

DEFAULT_ASSET_DIR = REPO_ROOT / "assets" / "rope"


def plot_inverse_frequencies(
    config: RopeConfig,
    *,
    output_path: Path | str | None = None,
) -> Path:
    path = Path(output_path) if output_path else DEFAULT_ASSET_DIR / "inv_freq.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    inv = inverse_frequencies(config.rotary_dim, config.theta).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(inv)), inv, color="#2c5f6e", marker="o", markersize=3)
    ax.set_xlabel("pair index i")
    ax.set_ylabel("inv_freq[i]")
    ax.set_title(
        f"RoPE inverse frequencies (θ={config.theta}, d_r={config.rotary_dim})"
    )
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_cos_sin_curves(
    config: RopeConfig,
    *,
    pair_index: int = 0,
    output_path: Path | str | None = None,
) -> Path:
    path = (
        Path(output_path) if output_path else DEFAULT_ASSET_DIR / "cos_sin_curves.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rope = OdysseyRoPE(config)
    cache = rope._manager.cache  # noqa: SLF001 — visualizer introspection
    pos = np.arange(min(256, cache.max_position))
    cos = cache.cos[: len(pos), pair_index].cpu().numpy()
    sin = cache.sin[: len(pos), pair_index].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos, cos, label=f"cos pair={pair_index}", color="#2c5f6e")
    ax.plot(pos, sin, label=f"sin pair={pair_index}", color="#b85c38")
    ax.set_xlabel("position m")
    ax.set_ylabel("value")
    ax.set_title("RoPE cos/sin vs position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_rotation_demo(
    config: RopeConfig,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """2D scatter of a unit pair rotating across positions."""
    path = Path(output_path) if output_path else DEFAULT_ASSET_DIR / "rotation_demo.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal head: rotary_dim dims, demo on first pair.
    demo = RopeConfig(
        theta=config.theta,
        head_dim=max(2, config.rotary_dim),
        rotary_dim=max(2, min(config.rotary_dim, 8)),
        max_position_embeddings=64,
        scaling=config.scaling,
        scaling_factor=config.scaling_factor,
    )
    rope = OdysseyRoPE(demo)
    x = torch.zeros(32, demo.head_dim)
    x[:, 0] = 1.0
    x[:, 1] = 0.0
    y = rope(x, position_offset=0)
    pts = y[:, :2].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="viridis", s=28)
    ax.axhline(0, color="#888", lw=0.5)
    ax.axvline(0, color="#888", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x0'")
    ax.set_ylabel("x1'")
    ax.set_title("Unit pair (1,0) under RoPE positions 0..31")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def export_rope_assets(config: RopeConfig | None = None) -> dict[str, Path]:
    cfg = config or RopeConfig(
        theta=10000.0,
        head_dim=128,
        rotary_dim=128,
        max_position_embeddings=4096,
    )
    DEFAULT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "inv_freq": plot_inverse_frequencies(cfg),
        "cos_sin": plot_cos_sin_curves(cfg),
        "rotation": plot_rotation_demo(cfg),
    }
