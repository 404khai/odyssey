# Root Mean Square Layer Normalization

**Paper:** Zhang & Sennrich, 2019 — https://arxiv.org/abs/1910.07467

---

## Motivation

LayerNorm subtracts the mean and divides by the standard deviation. The authors observed that **re-centering is not essential** for the benefits attributed to normalization in deep Transformers, while the mean reduction adds compute.

## Derivation

Given \(x \in \mathbb{R}^D\):

\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\varepsilon}
\qquad
\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\mathrm{RMS}(x)}
\]

Only a scale \(\gamma\) is learned (no bias \(\beta\) in Odyssey Spec v1).

## Comparison with LayerNorm

| Property | LayerNorm | RMSNorm |
| --- | --- | --- |
| Mean center | Yes | No |
| Scale | \(\gamma,\beta\) | \(\gamma\) only |
| Ops | higher | lower |
| Used by | original Transformer | LLaMA / Odyssey |

## Complexity

Both are \(O(D)\) per vector; RMSNorm avoids the mean pass and the bias add.

## Why Odyssey Uses RMSNorm

- Matches LLaMA / Spec v1.0.0
- Must stay bit-compatible with Phalanx `layers::RmsNorm`
- Fewer parameters per norm site (`D` vs `2D`)

Validated by `scripts/validate_rmsnorm.py`.
