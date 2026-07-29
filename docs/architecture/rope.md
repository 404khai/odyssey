# Rotary Positional Embeddings (RoPE)

## Motivation

Token embeddings alone do not encode order. Absolute position tables add a vector per index but extrapolate poorly. **RoPE** rotates Q/K feature pairs by an angle that grows with position so attention sees relative offsets.

## Mathematical Derivation

See Spec [`spec/rope.md`](../../spec/rope.md) and pedagogy [`math/rope.md`](../../math/rope.md).

\[
\theta_i = 10000^{-2i/d_r}
\]

\[
\begin{pmatrix} x_0' \\ x_1' \end{pmatrix}
=
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
\]

Rotation preserves pairwise (and thus subspace) L2 norms.

## Implementation

| Module | Role |
| --- | --- |
| `model/rope_math.py` | inv_freq + adjacent rotate |
| `model/rope_cache.py` | lazy cos/sin cache |
| `model/rope.py` | `OdysseyRoPE` public API |
| `configs/model.yaml` | θ, rotary_dim, scaling |

```python
from model import OdysseyRoPE, load_rope_config
rope = OdysseyRoPE(load_rope_config())
q_rot, k_rot = rope.apply_rotary(q, k, position_offset=0)
```

## Tensor Shapes

| Layout | Shape |
| --- | --- |
| Training | `(B, S, H, d)` |
| Phalanx | `(S, H, d)` or `(S, d)` |
| Output | Same as input |

## Complexity

- Time: \(O(B S H d)\) (same order as a cheap elementwise pass)
- Cache memory: \(O(S_{\max} \cdot d_r)\) for cos+sin

## Phalanx Compatibility

Cross-check with:

```bash
python scripts/validate_rope.py
```

Tolerance default `1e-6` (float32). Report: `experiments/ODY-0004/rope_validation.json`.

## Visualization

`assets/rope/` — inv-freq spectrum, cos/sin curves, 2D rotation demo.
