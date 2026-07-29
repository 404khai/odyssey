"""RoPE cache tests."""

from __future__ import annotations

import torch

from model.rope_cache import RopeCacheManager, build_rope_cache


def test_build_cache_shapes() -> None:
    cache = build_rope_cache(rotary_dim=8, max_position=16, theta=10000.0)
    assert cache.cos.shape == (16, 4)
    assert cache.sin.shape == (16, 4)
    assert cache.inv_freq.shape == (4,)


def test_lazy_extension() -> None:
    mgr = RopeCacheManager(
        rotary_dim=8, theta=10000.0, initial_max_position=16, device="cpu"
    )
    assert mgr.cache.max_position == 16
    c1 = mgr.get(16)
    c2 = mgr.get(16)
    assert c1 is c2 or torch.equal(c1.cos, c2.cos)
    grown = mgr.get(40)
    assert grown.max_position >= 40


def test_dtype_device_consistency() -> None:
    cache = build_rope_cache(
        rotary_dim=4, max_position=8, dtype=torch.float32, device="cpu"
    )
    assert cache.cos.dtype == torch.float32
    assert cache.cos.device.type == "cpu"


def test_position_zero_cos_one() -> None:
    cache = build_rope_cache(rotary_dim=4, max_position=4)
    assert torch.allclose(cache.cos[0], torch.ones(2), atol=1e-6)
    assert torch.allclose(cache.sin[0], torch.zeros(2), atol=1e-6)
