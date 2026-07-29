"""Model package — decoder-only transformer components.

Phase 3: embeddings. Phase 4: RoPE. Phase 5: RMSNorm + residual pathway.
"""

from model.config import (
    EmbeddingConfig,
    ModelConfig,
    NormConfig,
    RopeConfig,
    load_embedding_config,
    load_model_config,
    load_norm_config,
    load_rope_config,
)
from model.embeddings import EmbeddingInspection, OdysseyEmbedding
from model.initialization import describe_strategy, initialize_embedding
from model.residual import describe_residual_flow, pre_norm_residual, residual_add
from model.rmsnorm import OdysseyRMSNorm
from model.rope import OdysseyRoPE

__all__ = [
    "EmbeddingConfig",
    "EmbeddingInspection",
    "ModelConfig",
    "NormConfig",
    "OdysseyEmbedding",
    "OdysseyRMSNorm",
    "OdysseyRoPE",
    "RopeConfig",
    "describe_residual_flow",
    "describe_strategy",
    "initialize_embedding",
    "load_embedding_config",
    "load_model_config",
    "load_norm_config",
    "load_rope_config",
    "pre_norm_residual",
    "residual_add",
]
