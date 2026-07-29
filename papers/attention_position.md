# Attention Is All You Need — Positional Encoding

**Authors:** Vaswani et al.  
**Link:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
**Focus:** Positional Encoding section

---

## Motivation

Self-attention is permutation-invariant. Without position signals, token order is lost.

---

## Absolute Sinusoidal Encoding

\[
\begin{aligned}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d})
\end{aligned}
\]

Added to input embeddings. The \(10000\) base later reappears in RoPE’s θ schedule.

---

## Limitations

- Absolute, not relative
- Additive mixing with content embeddings
- Weaker long-context extrapolation than RoPE in practice

---

## Notes for Odyssey

Odyssey does **not** use additive sinusoids. Spec v1 mandates RoPE on Q/K (see `spec/rope.md`). This paper remains the historical root of the frequency schedule.
