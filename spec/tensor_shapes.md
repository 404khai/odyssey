# Tensor Shapes

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Enumerate every tensor in the Odyssey model: shape, purpose, memory layout, producer, and consumer.

---

## Notation

| Symbol | Meaning |
| --- | --- |
| \(B\) | Batch |
| \(S\) | Sequence length |
| \(V\) | `vocab_size` |
| \(D\) | `hidden_size` |
| \(I\) | `intermediate_size` |
| \(L\) | `num_layers` |
| \(H\) | `num_heads` |
| \(H_{kv}\) | `num_kv_heads` |
| \(d\) | `head_dim` |
| \(D_{kv}\) | \(H_{kv} \cdot d\) |

**Layout:** All dense weights are **row-major**. Embedding rows are contiguous of length \(D\).

---

## Theory

Shapes are the compatibility surface. A runtime that loads wrong ranks or transposes silently is out of compliance even if numerics look plausible.

---

## Weight Tensors

| Name (logical) | Shape | Purpose | Layout | Produced By | Consumed By |
| --- | --- | --- | --- | --- | --- |
| `tok_embeddings.weight` | \((V, D)\) | Token embedding table | Row \(t\) = vector for id \(t\) | Training / init | Embedding gather |
| `layers.{i}.attention_norm.weight` | \((D,)\) | Pre-attn RMSNorm \(\gamma\) | Contiguous | Training | RMSNorm |
| `layers.{i}.attention.wq.weight` | \((D, D)\) | Q projection \(W_Q\) | \(y = x W\) with weight stored as \((D_{\mathrm{out}}, D_{\mathrm{in}})\) — see note | Training | Attention |
| `layers.{i}.attention.wk.weight` | \((D_{kv}, D)\) | K projection | Same convention | Training | Attention |
| `layers.{i}.attention.wv.weight` | \((D_{kv}, D)\) | V projection | Same convention | Training | Attention |
| `layers.{i}.attention.wo.weight` | \((D, D)\) | Output projection | Same convention | Training | Attention |
| `layers.{i}.ffn_norm.weight` | \((D,)\) | Pre-FFN RMSNorm \(\gamma\) | Contiguous | Training | RMSNorm |
| `layers.{i}.feed_forward.w1.weight` | \((I, D)\) | SwiGLU gate | Same convention | Training | FFN |
| `layers.{i}.feed_forward.w3.weight` | \((I, D)\) | SwiGLU up | Same convention | Training | FFN |
| `layers.{i}.feed_forward.w2.weight` | \((D, I)\) | SwiGLU down | Same convention | Training | FFN |
| `norm.weight` | \((D,)\) | Final RMSNorm \(\gamma\) | Contiguous | Training | Final norm |
| `output.weight` | \((V, D)\) | LM head | Row-major; **absent** if `weight_tying=true` (reuse `tok_embeddings.weight`) | Training | Logits |

**Linear weight convention (Spec v1):** Stored shape is `(out_features, in_features)`. Matmul is \(y = x W^\top\) when \(x\) is \((...,\mathrm{in})\) and \(W\) is \((\mathrm{out},\mathrm{in})\) — identical to common `nn.Linear` storage. GGUF Llama tensors follow the same logical out/in orientation after ggml axis interpretation (see [gguf_mapping.md](gguf_mapping.md)).

Layer index \(i \in \{0,\ldots,L-1\}\).

---

## Activation Tensors

| Tensor | Shape | Purpose | Produced By | Consumed By |
| --- | --- | --- | --- | --- |
| `token_ids` | \((B, S)\) or \((S,)\) decode | Discrete tokens | Tokenizer | Embedding |
| `hidden` | \((B, S, D)\) | Residual stream | Embedding / layer | Next layer |
| `q` | \((B, S, H, d)\) | Queries | \(W_Q\) | RoPE + attn |
| `k` | \((B, S, H_{kv}, d)\) | Keys | \(W_K\) | RoPE + attn / cache |
| `v` | \((B, S, H_{kv}, d)\) | Values | \(W_V\) | Attn / cache |
| `attn_out` | \((B, S, D)\) | Attention output | \(W_O\) | Residual |
| `ffn_out` | \((B, S, D)\) | FFN output | \(W_2\) | Residual |
| `logits` | \((B, S, V)\) or \((V,)\) | Unnormalized scores | LM head | Loss / sampler |

Prefill uses full \(S\); decode often uses \(S=1\) with KV cache length \(T\).

---

## KV Cache

| Tensor | Shape | Purpose |
| --- | --- | --- |
| `cache_k[layer]` | \((B, H_{kv}, T, d)\) | Cached keys |
| `cache_v[layer]` | \((B, H_{kv}, T, d)\) | Cached values |

\(T\) grows with generated tokens up to `context_length`.

---

## Memory Layout Notes

- Embedding gather reads row \(t\) as \(D\) contiguous floats (or dequantized equivalent).
- No padding bytes between rows in the logical dense view.
- Batch-major activations: outermost \(B\), then \(S\).

---

## Examples

Embedding lookup:

```text
token_ids (1, 3) = [512, 1284, 7]
hidden    (1, 3, 768)
```

Tiny attention projections:

```text
wq: (768, 768)
wk: (768, 768)   # H_kv = H
wv: (768, 768)
wo: (768, 768)
```

GQA (\(H=32\), \(H_{kv}=8\), \(d=128\)):

```text
wq: (4096, 4096)
wk: (1024, 4096)
wv: (1024, 4096)
```

---

## Future Extensions

- Packed QKV fused weights (export may still explode to separate names)
- Sequence-parallel / sharded layouts (must reconstruct logical shapes above)

---

## Compatibility Notes

Phalanx `EmbeddingTable` uses runtime shape `[V, D]` matching `tok_embeddings.weight`. Activation rank conventions for Q/K in RoPE accept `[S, d]` or `[S, H, d]` at the layer boundary.
