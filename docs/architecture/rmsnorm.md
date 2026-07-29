# RMSNorm

## Motivation

Deep residual streams drift in magnitude without normalization, which destabilizes gradients. Classic Transformers used LayerNorm (mean-center + scale). Modern LLaMA-style models use **RMSNorm**, which drops mean centering and only scales by the root-mean-square — fewer ops, equal or better stability.

## Mathematical Derivation

See Spec [`spec/rmsnorm.md`](../../spec/rmsnorm.md) and pedagogy [`math/rmsnorm.md`](../../math/rmsnorm.md).

\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{D}\sum_i x_i^2+\varepsilon}
\qquad
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\mathrm{RMS}(x)}
\]

No bias. \(\varepsilon\) defaults to `1e-6` (`configs/model.yaml` → `norm.epsilon`).

Why skip the mean? Centering is not required for magnitude control; removing it cuts a reduction + broadcast and matches the Odyssey Spec / Phalanx contract.

## Implementation

| Module | Role |
| --- | --- |
| `model/normalization.py` | RMS helpers (fp32 accumulation) |
| `model/rmsnorm.py` | `OdysseyRMSNorm` public API |
| `model/residual.py` | Pre-norm residual add |
| `configs/model.yaml` | `norm.type` / `epsilon` |

```python
from model import OdysseyRMSNorm, load_norm_config, residual_add

norm = OdysseyRMSNorm(load_norm_config())
h = residual_add(x, attn(norm(x)))  # Spec pre-norm ordering
```

## Residual Ordering (frozen)

```text
x → RMSNorm → sub-layer → residual add → output
```

Post-norm is forbidden in Spec v1.

## Tensor Shapes

| Tensor | Shape |
| --- | --- |
| Input / output | `(batch, seq, hidden_size)` |
| \(\gamma\) | `(hidden_size,)` |

## Complexity

- Time: \(O(B S D)\)
- Parameters: \(O(D)\) for \(\gamma\) (cheaper than LayerNorm's \(\gamma+\beta\))

## Phalanx Compatibility

Cross-check with:

```bash
python scripts/validate_rmsnorm.py
```

Tolerance default `1e-6` (float32). Report: `experiments/ODY-0005/rmsnorm_validation.json`.

## Visualization

`assets/rmsnorm/` — residual flow inspector.
