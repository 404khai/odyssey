"""Pure helpers for attention tensor geometry and scaling.

Shapes follow Odyssey Spec v1.0.0 / Phalanx ``layers::Attention``:

- Activations enter as ``(B, S, D)``.
- Heads use ``(B, H, S, d)`` for scores (query-head major).
- KV may use fewer heads ``H_kv``; GQA expands them before the score GEMM.
"""

from __future__ import annotations

import math

import torch


def attention_scale(head_dim: int) -> float:
    """``1 / √d`` — keeps dot-product variance ~1 so softmax stays sharp, not saturated."""
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    return 1.0 / math.sqrt(float(head_dim))


def reshape_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(B, S, H·d)`` → ``(B, H, S, d)``.

    Contiguous intermediate avoids strided views that break float64 GEMMs.
    """
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 (B,S,H*d), got shape {tuple(x.shape)}")
    batch, seq, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError(
            f"last dim {hidden} not divisible by num_heads {num_heads}"
        )
    head_dim = hidden // num_heads
    # (B, S, H, d) then swap heads before sequence for SDPA layout.
    return x.view(batch, seq, num_heads, head_dim).transpose(1, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, S, d)`` → ``(B, S, H·d)``."""
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 (B,H,S,d), got shape {tuple(x.shape)}")
    batch, num_heads, seq, head_dim = x.shape
    return (
        x.transpose(1, 2)
        .contiguous()
        .view(batch, seq, num_heads * head_dim)
    )


def expand_kv_heads(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Broadcast ``(B, H_kv, S, d)`` → ``(B, H, S, d)`` for GQA / MQA.

    Each KV head is repeated ``H / H_kv`` times so it serves a contiguous
    group of query heads (LLaMA-style GQA).
    """
    if kv.ndim != 4:
        raise ValueError(f"expected rank-4 KV, got shape {tuple(kv.shape)}")
    batch, num_kv, seq, head_dim = kv.shape
    if num_query_heads % num_kv != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv})"
        )
    if num_query_heads == num_kv:
        return kv
    groups = num_query_heads // num_kv
    # (B, H_kv, 1, S, d) → (B, H_kv, G, S, d) → (B, H, S, d)
    return (
        kv.unsqueeze(2)
        .expand(batch, num_kv, groups, seq, head_dim)
        .reshape(batch, num_query_heads, seq, head_dim)
        .contiguous()
    )


def heads_to_rope_layout(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, S, d)`` → ``(B, S, H, d)`` for :class:`~model.rope.OdysseyRoPE`."""
    return x.transpose(1, 2).contiguous()


def rope_layout_to_heads(x: torch.Tensor) -> torch.Tensor:
    """``(B, S, H, d)`` → ``(B, H, S, d)`` after RoPE."""
    return x.transpose(1, 2).contiguous()
