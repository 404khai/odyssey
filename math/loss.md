# Loss — Mathematical Note (Phase 10 outline)

> Pre-implementation outline.

## Equations (preview)

Next-token cross-entropy for logits \(z \in \mathbb{R}^{V}\) and target \(y\):

\[
\mathcal{L} = -\log \mathrm{softmax}(z)_y = -z_y + \log\sum_{v=1}^{V} e^{z_v}
\]

Sequence loss averages (or sums) over non-pad positions.

## Complexity (preview)

Softmax over \(V\) per token: \(O(B S V)\) naive; can use fused CE kernels.

## Numerical stability

Log-sum-exp with max subtraction; never materialize full softmax unless needed.

## Weight tying link

If LM head weight equals \(E^\top\), logits are \(X E^\top\) — see [embeddings.md](embeddings.md).

## PyTorch vs Phalanx

Training loss stays in Odyssey (`F.cross_entropy`). Phalanx inference uses argmax / sampling on logits, not CE.
