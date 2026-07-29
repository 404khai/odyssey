"""Causal attention masks for decoder-only transformers.

Future positions receive ``-inf`` so Softmax assigns them zero weight.
Training and prefill use a full ``(S, S)`` lower-triangular mask; decode with
a KV cache (Phase 8+) may use a shorter key axis ``T`` with query length 1.
"""

from __future__ import annotations

import torch


def make_causal_mask(
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    key_len: int | None = None,
) -> torch.Tensor:
    """Build additive causal mask of shape ``(seq_len, key_len)``.

    ``mask[s, t] = 0`` if ``t <= s`` (allow), else ``-inf`` (block).
    When ``key_len`` differs from ``seq_len`` (cached keys), the query index
    ``s`` is treated as absolute position ``key_len - seq_len + s``.
    """
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    k_len = seq_len if key_len is None else key_len
    if k_len < 0:
        raise ValueError("key_len must be >= 0")
    if seq_len == 0 or k_len == 0:
        return torch.empty(seq_len, k_len, device=device, dtype=dtype)

    q = torch.arange(seq_len, device=device)
    k = torch.arange(k_len, device=device)
    # Absolute query positions when attending into a longer cached context.
    offset = k_len - seq_len
    allowed = k.unsqueeze(0) <= (q.unsqueeze(1) + offset)
    mask = torch.zeros(seq_len, k_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask


def apply_causal_mask(scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Add causal mask to scores ``(..., S, T)``.

    If ``mask`` is ``None``, builds a square mask from the last two dims.
    """
    if scores.ndim < 2:
        raise ValueError(f"scores need rank >= 2, got {tuple(scores.shape)}")
    seq_q, seq_k = scores.shape[-2], scores.shape[-1]
    if mask is None:
        mask = make_causal_mask(
            seq_q, key_len=seq_k, device=scores.device, dtype=torch.float32
        )
    # Broadcast mask over leading batch/head dims.
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(0)
    return scores.float() + mask.to(device=scores.device, dtype=torch.float32)
