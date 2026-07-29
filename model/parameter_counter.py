"""Parameter / memory accounting helpers for Odyssey modules."""

from __future__ import annotations

from typing import Any

from torch import nn


def count_parameters(module: nn.Module) -> int:
    """Total trainable parameter elements."""
    return sum(int(p.numel()) for p in module.parameters())


def memory_bytes(module: nn.Module) -> int:
    """Bytes occupied by parameters (device-agnostic size estimate)."""
    total = 0
    for p in module.parameters():
        total += int(p.numel()) * p.element_size()
    return total


def projection_breakdown(
    *,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, Any]:
    """FFN parameter breakdown for Spec weights ``w1`` / ``w3`` / ``w2``."""
    gate = intermediate_size * hidden_size
    up = intermediate_size * hidden_size
    down = hidden_size * intermediate_size
    total = gate + up + down
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "gate_proj_params": gate,  # w1
        "up_proj_params": up,  # w3
        "down_proj_params": down,  # w2
        "total_params": total,
        "memory_bytes_fp32": total * 4,
        "expansion_ratio": intermediate_size / hidden_size if hidden_size else 0.0,
    }


def format_module_params(module: nn.Module) -> str:
    n = count_parameters(module)
    m = memory_bytes(module)
    return f"params={n:,}  memory={m:,} B"
