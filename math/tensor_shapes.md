# Tensor Shapes in Odyssey

## Convention

Unless noted otherwise, training tensors use:

| Symbol | Meaning |
| --- | --- |
| \(B\) | batch size |
| \(S\) | sequence length |
| \(V\) | vocabulary size |
| \(D\) | hidden size (`hidden_size` / `n_embd`) |
| \(H\) | number of attention heads |
| \(d\) | head dimension (\(D / H\)) |

## Phase 3 shapes

```text
token_ids:  (B, S)           int64
embeddings: (B, S, D)        float
weight E:   (V, D)           float
```

Example from the Phase 3 goal:

```text
ids = [512, 1284, 7]     → treated as (1, 3)
out                      → (1, 3, D)
```

If you drop the batch axis for a single sequence of length 3:

```text
(3, D)
```

Odyssey’s `OdysseyEmbedding` **requires** the batch axis so shapes stay consistent with later Transformer blocks.

## Downstream (planned)

| Stage | Shape |
| --- | --- |
| After RoPE on Q/K | `(B, S, H, d)` or `(B, H, S, d)` |
| Attention scores | `(B, H, S, S)` |
| FFN hidden | `(B, S, intermediate)` |
| Logits | `(B, S, V)` |

## Memory

Bytes for a dense float32 tensor ≈ `numel × 4`.

Embedding table alone:

\[
\mathrm{bytes}(E) = V \cdot D \cdot \mathrm{sizeof(dtype)}
\]

For \(V=32000\), \(D=768\), float32 → **24,576,000 params ≈ 93.75 MiB**.

## Layout note (Phalanx)

GGUF stores `token_embd.weight` as ggml dims `[n_embd, n_vocab]`, but bytes are already vocab-major rows. Phalanx reinterprets to `[V, D]` before gather — same logical table as PyTorch `nn.Embedding.weight`.
