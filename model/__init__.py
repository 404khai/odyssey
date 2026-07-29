"""Model package — decoder-only transformer components.

Phase 3–5: embeddings, RoPE, RMSNorm. Phase 6: SwiGLU feed-forward.
"""

from model.config import (
    EmbeddingConfig,
    FeedForwardConfig,
    ModelConfig,
    NormConfig,
    RopeConfig,
    load_embedding_config,
    load_feed_forward_config,
    load_model_config,
    load_norm_config,
    load_rope_config,
)
from model.embeddings import EmbeddingInspection, OdysseyEmbedding
from model.feedforward import OdysseyFeedForward, build_feed_forward
from model.initialization import describe_strategy, initialize_embedding
from model.residual import describe_residual_flow, pre_norm_residual, residual_add
from model.rmsnorm import OdysseyRMSNorm
from model.rope import OdysseyRoPE
from model.swiglu import OdysseySwiGLU

__all__ = [
    "EmbeddingConfig",
    "EmbeddingInspection",
    "FeedForwardConfig",
    "ModelConfig",
    "NormConfig",
    "OdysseyEmbedding",
    "OdysseyFeedForward",
    "OdysseyRMSNorm",
    "OdysseyRoPE",
    "OdysseySwiGLU",
    "RopeConfig",
    "build_feed_forward",
    "describe_residual_flow",
    "describe_strategy",
    "initialize_embedding",
    "load_embedding_config",
    "load_feed_forward_config",
    "load_model_config",
    "load_norm_config",
    "load_rope_config",
    "pre_norm_residual",
    "residual_add",
]
