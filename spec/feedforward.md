# Feed-Forward (SwiGLU)

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Position-wise nonlinear MLP using a gated SwiGLU unit.

---

## Theory

Standard GeLU FFNs use one expansion. SwiGLU gates one projection with a Swish-activated projection, improving quality at similar compute (Shazeer; Llama).

---

## Mathematics

\[
\mathrm{Swish}(z) = z \cdot \sigma(z)
\]

\[
\mathrm{FFN}(x) = \bigl(\mathrm{Swish}(x W_1^\top) \odot (x W_3^\top)\bigr) W_2^\top
\]

| Weight | Role | Shape |
| --- | --- | --- |
| `w1` | Gate | `(I, D)` |
| `w3` | Up | `(I, D)` |
| `w2` | Down | `(D, I)` |

No biases. Activation key must be `swiglu`.

---

## Tensor Shapes

| Activation | Shape |
| --- | --- |
| Input / output | `(B, S, D)` |
| Gate / up mid | `(B, S, I)` |

---

## Implementation Notes

- `intermediate_size` \(I\) is independent of \(D\) (Tiny: 2048 vs 768).
- GGUF maps `w1→ffn_gate`, `w3→ffn_up`, `w2→ffn_down`.

---

## Examples

Tiny: \(D=768\), \(I=2048\).

---

## Future Extensions

- GeLU-only FFN profile (would need activation enum expansion)

---

## Compatibility Notes

Implementing ReLU/GeLU MLP while advertising `activation=swiglu` is non-compliant.
