# Attention

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Causal multi-head (or grouped-query) self-attention over the residual stream.

---

## Theory

Each token aggregates information from prior tokens (and itself) via content-based weights. Grouped-query attention shares KV heads across query head groups for cheaper decode.

---

## Mathematics

Projections (per token, residual \(x \in \mathbb{R}^{D}\)):

\[
Q = x W_Q^\top,\quad K = x W_K^\top,\quad V = x W_V^\top
\]

Reshape to heads; apply RoPE to Q/K; for each query head attending to its KV group:

\[
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}} + M\right) V
\]

Causal mask \(M\): \(M_{s,t} = 0\) if \(t \le s\), else \(-\infty\).

Concatenate heads and apply \(W_O\).

GQA: \(H / H_{kv}\) query heads share one KV head.

---

## Tensor Shapes

| Tensor | Shape |
| --- | --- |
| `wq` | `(D, D)` |
| `wk`, `wv` | `(D_kv, D)` with \(D_{kv}=H_{kv}d\) |
| `wo` | `(D, D)` |
| Scores (dense) | `(B, H, S, S)` or `(B, H, 1, T)` decode |

No projection biases in Spec v1.

---

## Implementation Notes

- Softmax in float32 recommended for stability.
- Flash / memory-efficient kernels allowed if numerically equivalent under Spec tolerances (to be defined in testing docs later).

---

## Examples

Tiny MHA: \(H=H_{kv}=12\), \(d=64\).

---

## Future Extensions

- Sliding window
- Multi- Latent Attention / other KV compressors (Spec 2)

---

## Compatibility Notes

Must consume RoPE-rotated Q/K. Absolute position embeddings are forbidden in Spec v1.
