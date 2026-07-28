# SwiGLU — Mathematical Note (Phase 7 outline)

> Pre-implementation outline.

## Equations (preview)

\[
\mathrm{Swish}(x) = x \cdot \sigma(x)
\]

\[
\mathrm{SwiGLU}(x) = \mathrm{Swish}(x W_1) \odot (x W_2)
\]

Followed by output projection \(W_3\). Llama-style FFN uses this gated unit.

## Complexity (preview)

Three projections: roughly \(O(B S D \cdot I)\) with intermediate size \(I\).

## Numerical stability

Sigmoid saturation is mild with Swish; watch activation scale after residual add + RMSNorm.

## PyTorch vs Phalanx

Identical affine + elementwise structure; Phalanx executes fused or sequential GEMMs at inference.
