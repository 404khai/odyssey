# RoPE — Rotary Positional Embeddings

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Encode token positions by rotating query/key subspaces so relative offsets appear in attention logits.

---

## Theory

Absolute position embeddings add a vector per index. RoPE instead applies a position-dependent rotation to pairs of features. Attention between positions \(m\) and \(n\) then depends on \(m-n\).

---

## Mathematics

Let head dimension \(d\) (even rotary width \(d_r \le d\)). For pair index \(i = 0..\frac{d_r}{2}-1\):

\[
\theta_i = \theta_{\mathrm{base}}^{-2i/d_r}
\]

Default \(\theta_{\mathrm{base}} =\) `rope_theta` \(= 10000\).

For position \(m\) (after optional linear scaling \(m' = m / \mathrm{factor}\)):

\[
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix}
\cos(m'\theta_i) & -\sin(m'\theta_i) \\
\sin(m'\theta_i) & \cos(m'\theta_i)
\end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
\]

Applied to **Q and K only**. Dimensions \(\ge d_r\) are unchanged (partial RoPE).

---

## Tensor Shapes

| Tensor | Shape |
| --- | --- |
| Q / K in | `(B, S, H[*], d)` or fused `(S, d)` / `(S, H, d)` |
| Q / K out | Same shape |
| Cos/Sin tables | `(S_max, d_r/2)` cached or equivalent |

---

## Config

| Key | Constraint |
| --- | --- |
| `rope_theta` | Finite, \(> 0\) |
| `rope_dim` | Even, \(1 < rope\_dim \le head\_dim\) |
| `rope_scaling.type` | `none` \| `linear` |
| `rope_scaling.factor` | \(> 0\) when linear |

YaRN/NTK are **out of scope** for Spec v1.

---

## Implementation Notes

- Training and inference must share \(\theta_{\mathrm{base}}\), \(d_r\), and scaling.
- Phalanx `layers::Rope` already matches this math (linear scaling only).

---

## Examples

Tiny: \(d=64\), \(d_r=64\), \(\theta=10000\), positions `0..2047`.

---

## Future Extensions

- NTK / YaRN scaling (Spec minor if purely additive metadata + compatible math)

---

## Compatibility Notes

Mismatch in `rope_theta` or `rope_dim` between checkpoint and runtime is a hard error.
