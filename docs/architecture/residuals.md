# Residual Connections

## Motivation

Without skip connections, deep stacks suffer vanishing / exploding gradients. Residuals implement an **identity highway**:

\[
y = x + f(x)
\]

so \(\partial y / \partial x\) always includes a `1` term regardless of \(f\)'s local Jacobian.

## Odyssey Ordering

Spec v1 freezes **pre-norm** residuals:

```text
Input → RMSNorm → Sub-layer → Residual Add → Output
```

Never post-norm (`LayerNorm(x + f(x))`).

## Implementation

```python
from model import pre_norm_residual, residual_add

y = residual_add(x, sublayer_out)          # strict shape check
y = pre_norm_residual(x, norm=norm, sublayer=attn)
```

## Identity Mapping

When \(f \approx 0\) early in training, \(y \approx x\) — the block is a near-identity map and cannot collapse representation capacity.

## Phalanx Compatibility

Phalanx decoder blocks (Phase 13) must use the same pre-norm residual order. RMSNorm itself is validated in Phase 9 via `scripts/validate_rmsnorm.py`.
