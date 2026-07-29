"""Model package — decoder-only transformer components.

Phase 3: embeddings. Phase 4: LLaMA-style RoPE (Phalanx-parity).
"""

from model.config import (
    EmbeddingConfig,
    ModelConfig,
    RopeConfig,
    load_embedding_config,
    load_model_config,
    load_rope_config,
)
from model.embeddings import EmbeddingInspection, OdysseyEmbedding
from model.initialization import describe_strategy, initialize_embedding
from model.rope import OdysseyRoPE

__all__ = [
    "EmbeddingConfig",
    "EmbeddingInspection",
    "ModelConfig",
    "OdysseyEmbedding",
    "OdysseyRoPE",
    "RopeConfig",
    "describe_strategy",
    "initialize_embedding",
    "load_embedding_config",
    "load_model_config",
    "load_rope_config",
]
