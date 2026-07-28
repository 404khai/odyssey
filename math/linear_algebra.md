# Linear Algebra Primer for Odyssey

## Vectors and matrices

A token embedding is a vector \(x \in \mathbb{R}^{D}\).
The embedding table is a matrix \(E \in \mathbb{R}^{V \times D}\).

Row \(t\) of \(E\) is the vector for token id \(t\):

\[
x = E_{t,:} = E[t]
\]

Matrix–vector products appear later (attention projections, FFN, LM head):

\[
y = W x + b, \quad W \in \mathbb{R}^{m \times n},\; x \in \mathbb{R}^{n}
\]

## Norms

L2 (Euclidean) norm:

\[
\|x\|_2 = \sqrt{\sum_{i=1}^{D} x_i^2}
\]

Cosine similarity (nearest-neighbor inspection):

\[
\cos(a,b) = \frac{a^\top b}{\|a\|_2 \|b\|_2}
\]

## Softmax (preview)

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Numerically computed as \(\mathrm{softmax}(z - \max z)\) for stability.

## Why this matters for Odyssey

- Embeddings are **rows of a matrix**, not opaque “layers.”
- Attention is batched matmuls + softmax.
- Phalanx Runtime and PyTorch must agree on **layout** (row-major gather vs ggml axis order).

## Complexity cheat sheet

| Op | Time | Memory |
| --- | --- | --- |
| Dot product \(a^\top b\) | \(O(D)\) | \(O(1)\) extra |
| Matmul \(A_{m\times k} B_{k\times n}\) | \(O(mkn)\) | \(O(mn)\) output |
| Softmax length \(n\) | \(O(n)\) | \(O(n)\) |

## PyTorch vs Phalanx

| Concept | PyTorch | Phalanx Runtime |
| --- | --- | --- |
| Dense vector | `Tensor` | `tensor::Tensor` |
| Row-major matmul | `torch.matmul` | planned GEMM kernels |
| Gather row | `nn.Embedding` / `index_select` | `EmbeddingTable::forward` |
