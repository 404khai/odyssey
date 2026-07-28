# Embeddings — Mathematical Note (Phase 3)

## Equations

Let vocabulary size \(V\) and hidden size \(D\). Embedding matrix:

\[
E \in \mathbb{R}^{V \times D}
\]

For token id \(t \in \{0,\ldots,V-1\}\):

\[
x = E[t] \in \mathbb{R}^{D}
\]

For a batch of sequences \(T \in \mathbb{Z}^{B \times S}\):

\[
X_{b,s,:} = E[T_{b,s}]
\quad\Rightarrow\quad
X \in \mathbb{R}^{B \times S \times D}
\]

No matrix multiply occurs at the input: embedding is an **indexed gather**.

If `padding_idx = p` is set:

\[
E[p] = 0, \qquad \frac{\partial \mathcal{L}}{\partial E[p]} = 0
\]

### Weight tying (documented, not implemented yet)

GPT-style models often share input embedding \(E\) with the output projection:

\[
\mathrm{logits} = X W^\top, \quad W = E
\]

so the LM head reuses the same \(V \times D\) table. Saves \(V D\) parameters and couples input/output geometry. Odyssey will wire this in a later phase.

## Why it works

Integer token ids carry **no metric structure**. Mapping them into \(\mathbb{R}^{D}\) lets the network:

1. Place tokens in a continuous space where nearby vectors mean related usage.
2. Differentiate (backprop updates rows of \(E\)).
3. Feed dense activations into linear layers and attention.

Initially \(E\) is random. Training moves rows so that co-occurring / substitutable tokens cluster (see Word2Vec / GloVe papers).

## Time complexity

| Op | Cost |
| --- | --- |
| Gather \(B\cdot S\) rows | \(O(B S D)\) copy/read |
| Init of \(E\) | \(O(V D)\) |

Independent of matmul asymptotics; bottleneck is memory bandwidth.

## Memory complexity

| Tensor | Size |
| --- | --- |
| Weights \(E\) | \(O(V D)\) |
| Activations \(X\) | \(O(B S D)\) |
| Gradients \(\partial E\) (train) | \(O(V D)\) sparse-ish in practice (touched rows) |

## Numerical stability

- Embedding lookup itself is stable (no reductions).
- Init scale matters: too large → early attention logits explode; too small → slow learning.
- Odyssey default: **Xavier uniform**; GPT-style alternative: \(\mathcal{N}(0, 0.02^2)\).
- Prefer float32 for the table during early research; float16/bf16 save memory later.

## How PyTorch implements it

`torch.nn.Embedding` stores `weight` as `(V, D)` and implements forward as an embedding bag / index select kernel. Backward scatters gradients into rows referenced by `token_ids`. With `padding_idx`, that row stays zero and receives zero grad.

Odyssey wraps this in `OdysseyEmbedding` for config, shape checks, init, and inspection — no custom CUDA.

## How Phalanx Runtime executes it

At inference, Phalanx loads GGUF `token_embd.weight`, reinterprets to row-major `[V, D]`, and gathers via `EmbeddingTable::forward`:

```text
token ids ──► EmbeddingTable::forward ──► [seq, n_embd]
```

Same math as PyTorch: \(x = E[t]\). Training stays in Odyssey/PyTorch; serving gather lives in the Rust runtime (`runtime/src/layers/embedding.rs`).

## Odyssey mapping

| Artifact | Path |
| --- | --- |
| Module | `model/embeddings.py` |
| Config | `model/config.py`, `configs/embedding.yaml` |
| Init | `model/initialization.py` |
| Docs | `docs/architecture/embeddings.md` |
| Experiment | `ODY-0003` |
