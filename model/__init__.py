"""Model package — decoder-only transformer components.

Phase 3–6: embeddings, RoPE, RMSNorm, SwiGLU. Phase 7: GQA attention.
"""

from model.attention import OdysseyAttention, OdysseyMultiHeadAttention
from model.config import (
    AttentionConfig,
    EmbeddingConfig,
    FeedForwardConfig,
    ModelConfig,
    NormConfig,
    RopeConfig,
    load_attention_config,
    load_embedding_config,
    load_feed_forward_config,
    load_model_config,
    load_norm_config,
    load_rope_config,
)
from model.embeddings import EmbeddingInspection, OdysseyEmbedding
from model.feedforward import OdysseyFeedForward, build_feed_forward
from model.gqa import OdysseyGQA
from model.initialization import describe_strategy, initialize_embedding
from model.residual import describe_residual_flow, pre_norm_residual, residual_add
from model.rmsnorm import OdysseyRMSNorm
from model.rope import OdysseyRoPE
from model.swiglu import OdysseySwiGLU

__all__ = [
    "AttentionConfig",
    "EmbeddingConfig",
    "EmbeddingInspection",
    "FeedForwardConfig",
    "ModelConfig",
    "NormConfig",
    "OdysseyAttention",
    "OdysseyEmbedding",
    "OdysseyFeedForward",
    "OdysseyGQA",
    "OdysseyMultiHeadAttention",
    "OdysseyRMSNorm",
    "OdysseyRoPE",
    "OdysseySwiGLU",
    "RopeConfig",
    "build_feed_forward",
    "describe_residual_flow",
    "describe_strategy",
    "initialize_embedding",
    "load_attention_config",
    "load_embedding_config",
    "load_feed_forward_config",
    "load_model_config",
    "load_norm_config",
    "load_rope_config",
    "pre_norm_residual",
    "residual_add",
]
