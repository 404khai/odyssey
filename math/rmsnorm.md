# RMSNorm — Mathematical Note (Phase 5 outline)

> Pre-implementation outline.

## Equations (preview)

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \varepsilon}
\]

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

No mean centering (unlike LayerNorm).

## Complexity (preview)

Time \(O(B S D)\), memory \(O(D)\) for \(\gamma\).

## Numerical stability

Keep \(\varepsilon\) (e.g. \(10^{-6}\)); prefer float32 accumulators for the sum of squares.

## PyTorch vs Phalanx

Odyssey will implement a small `nn.Module`; Phalanx will mirror the same affine RMS formula on activations during decode/prefill.
