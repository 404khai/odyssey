# RoPE Mathematics (Architecture Companion)

Normative equations live in [`spec/rope.md`](../../spec/rope.md).

## Variables

| Symbol | Meaning |
| --- | --- |
| \(d\) | `head_dim` |
| \(d_r\) | `rotary_dim` (even, ≤ \(d\)) |
| \(\theta\) | base frequency (`theta`, default 10000) |
| \(m\) | absolute position |
| \(m'\) | \(m / \mathrm{factor}\) under linear scaling |
| \(\omega_i\) | \(\theta^{-2i/d_r}\) |

## Why Adjacent Pairing

LLaMA stores pairs as consecutive dims. The complex view \((x_{2i}+i x_{2i+1})\) maps directly onto that layout without a gather/transpose of even/odd channels across the full head.

## Why Not Rotate V

RoPE’s relative-phase argument applies to the \(QK^\top\) product. Rotating \(V\) is not part of the standard derivation and would mix content with position unnecessarily.

## Cache Layout (shared with Phalanx)

```text
cos[pos, pair], sin[pos, pair]
flat index = pos * (d_r/2) + pair
```
