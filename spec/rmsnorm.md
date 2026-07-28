# RMSNorm

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Stabilize residual-stream magnitude without mean centering.

---

## Theory

LayerNorm centers and scales. RMSNorm scales by the root-mean-square only, reducing compute and matching Llama-style transformers.

---

## Mathematics

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \varepsilon}
\]

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

No bias term in Spec v1. \(\varepsilon =\) `rms_norm_eps` (default `1e-6`).

Applied at:

1. Pre-attention (`attention_norm`)
2. Pre-FFN (`ffn_norm`)
3. Final (`norm`)

---

## Tensor Shapes

| Tensor | Shape |
| --- | --- |
| Input / output | `(..., D)` |
| \(\gamma\) | `(D,)` |

---

## Implementation Notes

- Prefer float32 accumulation for the sum of squares when activations are fp16/bf16.
- \(\gamma\) initialized to ones in training.

---

## Examples

Tiny: \(\gamma\) length 768, \(\varepsilon=10^{-6}\).

---

## Future Extensions

- Learnable \(\varepsilon\) (not planned)
- Post-norm stacks (breaking)

---

## Compatibility Notes

Phalanx Phase 9 must implement this exact formula. Using LayerNorm instead is non-compliant.
