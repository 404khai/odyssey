"""Public feed-forward network interface for Odyssey.

Today the only Spec-compliant FFN is SwiGLU. This module re-exports the
canonical class under a stable name used by docs and future decoder blocks.
"""

from __future__ import annotations

from model.config import FeedForwardConfig
from model.swiglu import OdysseySwiGLU

# Public alias — decoder blocks should depend on this name.
OdysseyFeedForward = OdysseySwiGLU


def build_feed_forward(config: FeedForwardConfig) -> OdysseySwiGLU:
    """Factory that rejects non-SwiGLU activations (Spec compliance)."""
    if config.type != "swiglu":
        raise ValueError(
            f"only feed_forward.type='swiglu' is Spec-compliant, got {config.type!r}"
        )
    return OdysseySwiGLU.from_config(config)


__all__ = [
    "OdysseyFeedForward",
    "OdysseySwiGLU",
    "build_feed_forward",
]
