"""Grouped Query Attention kernel (primary Odyssey attention path).

Implements scaled causal attention with optional KV-head sharing:

    Attn(Q, K, V) = softmax(Q Kᵀ / √d + M) V

When ``H_kv < H``, each KV head serves ``H / H_kv`` query heads (GQA).
``H_kv == 1`` is Multi-Query Attention (MQA); ``H_kv == H`` is classic MHA.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.attention_math import (
    attention_scale,
    expand_kv_heads,
    merge_heads,
    reshape_to_heads,
)
from model.causal_mask import apply_causal_mask
from model.config import AttentionConfig
from model.softmax import stable_softmax


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core SDPA on head layouts ``(B, H, S, d)``.

    Returns ``(context, attn_weights)`` both float32 (weights match input dtype).
    GEMMs use float64 accumulators to mirror Phalanx reference kernels.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 (B, H, S, d)")
    head_dim = q.shape[-1]
    scale = attention_scale(head_dim)

    # Expand KV when GQA already applied by caller — shapes must match heads.
    if k.shape[1] != q.shape[1]:
        k = expand_kv_heads(k, q.shape[1])
        v = expand_kv_heads(v, q.shape[1])

    # scores: (B, H, S_q, S_k)
    scores = torch.matmul(q.double(), k.double().transpose(-2, -1)) * scale
    scores = scores.to(dtype=torch.float32)
    if causal:
        scores = apply_causal_mask(scores)
    weights = stable_softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
    context = torch.matmul(weights.double(), v.double()).to(dtype=torch.float32)
    return context, weights


class OdysseyGQA(nn.Module):
    """Head-layout GQA/MHA over already-projected Q, K, V tensors.

    Expects ``q: (B,S,H·d)``, ``k/v: (B,S,H_kv·d)`` and returns
    ``(B,S,H·d)`` plus optional attention weights.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        qh = reshape_to_heads(q, cfg.num_heads)
        kh = reshape_to_heads(k, cfg.num_kv_heads)
        vh = reshape_to_heads(v, cfg.num_kv_heads)
        kh = expand_kv_heads(kh, cfg.num_heads)
        vh = expand_kv_heads(vh, cfg.num_heads)
        ctx, weights = scaled_dot_product_attention(
            qh,
            kh,
            vh,
            causal=cfg.causal,
            dropout_p=cfg.dropout,
            training=self.training,
        )
        out = merge_heads(ctx)
        if return_weights:
            return out, weights
        return out

    def inspect(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "gqa_groups": cfg.gqa_groups,
            "is_gqa": cfg.is_gqa,
            "is_mqa": cfg.is_mqa,
            "causal": cfg.causal,
            "scale": attention_scale(cfg.head_dim),
        }
