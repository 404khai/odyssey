# Residual Connections (Attention Is All You Need)

**Paper:** Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762

---

## Motivation

The original Transformer wraps each sub-layer as:

\[
\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
\]

Skip connections keep gradients flowing through deep encoder/decoder stacks (**gradient highways**).

## Notes for Odyssey

Odyssey / LLaMA invert the norm placement (**pre-norm**):

\[
x + \mathrm{Sublayer}(\mathrm{RMSNorm}(x))
\]

Same residual identity idea; different norm position. Spec v1 freezes pre-norm — post-norm implementations are non-compliant.

## Identity Mapping

If the sub-layer learns near-zero, the block remains approximately \(x\), preserving representation capacity during early training.
