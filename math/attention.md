# Attention — Mathematical Note (Phase 6 outline)

> Pre-implementation outline.

## Equations (preview)

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
\]

Causal mask: set future scores to \(-\infty\) before softmax.

Multi-head: split \(D\) into \(H\) heads of size \(d = D/H\), concatenate, project.

## Complexity (preview)

- Naive attention: time \(O(B H S^2 d)\), memory \(O(B H S^2)\) for scores
- Memory-efficient kernels trade compute patterns for lower activation memory

## Numerical stability

Scale by \(\sqrt{d}\); use online softmax / flash-style kernels later if needed.

## PyTorch vs Phalanx

Training: PyTorch SDPA / manual matmul. Inference: Phalanx attention kernels over GGUF weights.
