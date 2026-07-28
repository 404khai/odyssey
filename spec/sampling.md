# Sampling

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes (inference)

---

## Purpose

Map LM logits to the next token id.

---

## Theory

The model defines a distribution over \(V\) tokens. Sampling strategies trade determinism for diversity without changing trained weights.

---

## Mathematics

Let \(z \in \mathbb{R}^{V}\) be logits for the active position.

**Greedy**

\[
t = \arg\max_i z_i
\]

**Temperature** \(\tau > 0\)

\[
p = \mathrm{softmax}(z / \tau)
\]

**Top-k:** keep \(k\) largest logits, mask others to \(-\infty\), then softmax.

**Top-p (nucleus):** keep smallest set whose cumulative probability \(\ge p\), then renormalize.

**Min-p:** keep tokens with \(p_i \ge p_{\min} \cdot p_{\mathrm{max}}\) (optional Spec v1 algorithm once implemented — metadata must name the variant).

---

## Tensor Shapes

| Tensor | Shape |
| --- | --- |
| Logits in | `(V,)` or `(B, V)` |
| Token out | scalar id or `(B,)` |

---

## Required Metadata (runtime)

Sampling is **not** stored in weights. Callers pass:

| Parameter | Default recommendation |
| --- | --- |
| `temperature` | `1.0` (greedy if `0`) |
| `top_k` | `0` = disabled |
| `top_p` | `1.0` = disabled |
| `seed` | required for reproducibility when stochastic |

---

## Implementation Notes

- Apply softmax numerically stably (max subtraction).
- Never sample padding/`<pad>` unless explicitly allowed by API options.

---

## Examples

Greedy decode: `temperature=0` → argmax.

---

## Future Extensions

- Mirostat, typical sampling, grammar-constrained decoding

---

## Compatibility Notes

Training ignores this document (uses teacher-forced CE). Phalanx Phase 14 implements sampling against this contract.
