"""Model package — decoder-only transformer components.

Phase 3 delivers the token embedding layer. Later phases add RoPE, RMSNorm,
attention, SwiGLU, and the full decoder stack.
"""

from model.config import EmbeddingConfig, load_embedding_config
from model.embeddings import EmbeddingInspection, OdysseyEmbedding
from model.initialization import describe_strategy, initialize_embedding

__all__ = [
    "EmbeddingConfig",
    "EmbeddingInspection",
    "OdysseyEmbedding",
    "describe_strategy",
    "initialize_embedding",
    "load_embedding_config",
]
