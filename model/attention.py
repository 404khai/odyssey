"""Public attention module — Grouped Query Attention (primary) + MHA reference.

Architecture (Odyssey Spec v1.0.0 / Phalanx ``layers::Attention``)::

    x → QKV projections → reshape heads → RoPE(Q,K) → GQA SDPA → merge → W_O

RoPE is applied when an :class:`~model.rope.OdysseyRoPE` is attached (Spec
requires RoPE-rotated Q/K). Pass ``rope=None`` only for isolated unit tests of
the score/mask/softmax path.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.attention_math import (
    attention_scale,
    heads_to_rope_layout,
    merge_heads,
    reshape_to_heads,
    rope_layout_to_heads,
)
from model.config import AttentionConfig, RopeConfig
from model.gqa import OdysseyGQA, scaled_dot_product_attention
from model.parameter_counter import count_parameters, memory_bytes
from model.projection import AttentionProjections
from model.rope import OdysseyRoPE


class OdysseyAttention(nn.Module):
    """Causal self-attention with configurable GQA / MHA / MQA layouts."""

    def __init__(
        self,
        config: AttentionConfig,
        *,
        rope: OdysseyRoPE | None = None,
        build_rope: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.projections = AttentionProjections(config)
        self.gqa = OdysseyGQA(config)
        if rope is not None:
            self.rope: OdysseyRoPE | None = rope
        elif build_rope:
            self.rope = OdysseyRoPE(
                RopeConfig(
                    head_dim=config.head_dim,
                    rotary_dim=config.head_dim,
                    device=config.device,
                    dtype=config.dtype,
                )
            )
        else:
            self.rope = None

    @classmethod
    def from_config(
        cls,
        config: AttentionConfig,
        *,
        rope: OdysseyRoPE | None = None,
        build_rope: bool = False,
    ) -> OdysseyAttention:
        return cls(config, rope=rope, build_rope=build_rope)

    @property
    def num_heads(self) -> int:
        return self.config.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    def parameter_count(self) -> int:
        return count_parameters(self)

    def memory_bytes(self) -> int:
        return memory_bytes(self)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``(B, S, D)`` → ``(B, S, D)``; optionally also return attn weights."""
        self.validate_input(x)
        q, k, v = self.projections.project_qkv(x)

        qh = reshape_to_heads(q, self.config.num_heads)
        kh = reshape_to_heads(k, self.config.num_kv_heads)
        vh = reshape_to_heads(v, self.config.num_kv_heads)

        if self.rope is not None:
            # OdysseyRoPE expects (B, S, H, d); SDPA uses (B, H, S, d).
            q_rope = heads_to_rope_layout(qh)
            k_rope = heads_to_rope_layout(kh)
            q_rope, k_rope = self.rope.apply_rotary(
                q_rope, k_rope, position_offset=position_offset
            )
            qh = rope_layout_to_heads(q_rope)
            kh = rope_layout_to_heads(k_rope)

        from model.attention_math import expand_kv_heads

        kh = expand_kv_heads(kh, self.config.num_heads)
        vh = expand_kv_heads(vh, self.config.num_heads)

        ctx, weights = scaled_dot_product_attention(
            qh,
            kh,
            vh,
            causal=self.config.causal,
            dropout_p=self.config.dropout,
            training=self.training,
        )
        merged = merge_heads(ctx)
        out = self.projections.project_output(merged)
        out = out.to(dtype=x.dtype)
        if return_weights:
            return out, weights
        return out

    def validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"expected (B, S, D), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"last dim {x.shape[-1]} != hidden_size {self.config.hidden_size}"
            )

    def inspect(self) -> dict[str, Any]:
        cfg = self.config
        q_params = cfg.query_dim * cfg.hidden_size
        kv_params = cfg.kv_dim * cfg.hidden_size
        o_params = cfg.hidden_size * cfg.query_dim
        return {
            "type": cfg.attention_type,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "hidden_size": cfg.hidden_size,
            "gqa_groups": cfg.gqa_groups,
            "causal": cfg.causal,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
            "scale": attention_scale(cfg.head_dim),
            "rope": self.rope is not None,
            "parameter_count": self.parameter_count(),
            "memory_bytes": self.memory_bytes(),
            "projections": {
                "q_proj": q_params,
                "k_proj": kv_params,
                "v_proj": kv_params,
                "o_proj": o_params,
            },
            "shapes": {
                "input": f"(B, S, {cfg.hidden_size})",
                "q": f"(B, {cfg.num_heads}, S, {cfg.head_dim})",
                "k": f"(B, {cfg.num_kv_heads}, S, {cfg.head_dim})",
                "v": f"(B, {cfg.num_kv_heads}, S, {cfg.head_dim})",
                "output": f"(B, S, {cfg.hidden_size})",
            },
            "device": str(self.projections.q_proj.weight.device),
            "dtype": str(self.projections.q_proj.weight.dtype),
        }

    def format_inspect(self) -> str:
        info = self.inspect()
        return (
            f"OdysseyAttention({info['type'].upper()}, "
            f"H={info['num_heads']}, H_kv={info['num_kv_heads']}, "
            f"d={info['head_dim']}, params={info['parameter_count']:,})"
        )


class OdysseyMultiHeadAttention(OdysseyAttention):
    """Educational MHA wrapper — requires ``num_kv_heads == num_heads``."""

    def __init__(
        self,
        config: AttentionConfig,
        *,
        rope: OdysseyRoPE | None = None,
        build_rope: bool = False,
    ) -> None:
        if config.num_kv_heads != config.num_heads:
            raise ValueError(
                "OdysseyMultiHeadAttention requires num_kv_heads == num_heads "
                f"(got {config.num_kv_heads} vs {config.num_heads}); "
                "use OdysseyAttention for GQA"
            )
        super().__init__(config, rope=rope, build_rope=build_rope)
