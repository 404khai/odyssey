# Weight Layout

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes — **names are frozen**

---

## Purpose

Define the permanent logical parameter namespace for Odyssey. Training checkpoints, export tools, and Phalanx Runtime all use these names (or a documented bijection to GGUF).

**Never rename after Spec v1.**

---

## Theory

A shared namespace prevents train/serve skew. The runtime loads weights by name from metadata; it does not pattern-match ad-hoc strings.

---

## Frozen Parameter Names

```text
tok_embeddings.weight

layers.{i}.attention_norm.weight
layers.{i}.attention.wq.weight
layers.{i}.attention.wk.weight
layers.{i}.attention.wv.weight
layers.{i}.attention.wo.weight

layers.{i}.ffn_norm.weight
layers.{i}.feed_forward.w1.weight
layers.{i}.feed_forward.w3.weight
layers.{i}.feed_forward.w2.weight

norm.weight
output.weight          # omitted when weight_tying = true
```

`{i}` is the decimal layer index without zero-padding (`layers.0`, `layers.10`, …).

### Semantics

| Name | Role |
| --- | --- |
| `tok_embeddings.weight` | Input embedding \(E\) |
| `attention_norm` / `ffn_norm` | RMSNorm \(\gamma\) |
| `wq` `wk` `wv` `wo` | Attention projections |
| `w1` | SwiGLU gate |
| `w3` | SwiGLU up |
| `w2` | SwiGLU down |
| `norm.weight` | Final RMSNorm |
| `output.weight` | LM head \(W_U\) |

SwiGLU: \(\mathrm{Swish}(x W_1^\top) \odot (x W_3^\top)\) then \(W_2\) — see [feedforward.md](feedforward.md).

---

## Prohibited

- Renaming `tok_embeddings` → `embed_tokens` without a Spec major bump
- Per-framework aliases inside the checkpoint itself
- Inferring layer count from filename globs when metadata is present

---

## Mathematics

Parameter count (untied):

\[
\begin{aligned}
P &= V D + L\bigl(2D + 2D^2 + 2 D D_{kv} + 3 D I\bigr) + D + V D \\
&= 2VD + L(2D + 2D^2 + 2DD_{kv} + 3DI) + D
\end{aligned}
\]

With tying, drop one \(VD\). For MHA, \(D_{kv}=D\).

---

## Tensor Shapes

See [tensor_shapes.md](tensor_shapes.md).

---

## Implementation Notes

- Odyssey PyTorch `state_dict` keys **must** match this layout once the full model lands.
- Until then, embeddings map as: module weight ↔ `tok_embeddings.weight`.
- GGUF uses different on-disk strings; [gguf_mapping.md](gguf_mapping.md) is the bijection.

---

## Examples

Layer 0 attention query:

```text
layers.0.attention.wq.weight   shape (768, 768) for Tiny
```

---

## Future Extensions

- LoRA adapters: `layers.{i}.attention.wq.lora_A.weight` (additive Spec minor if optional)
- Quantization scales: sidecar names documented in serialization Spec updates

---

## Compatibility Notes

Phalanx binds `token_embd.weight` ←→ `tok_embeddings.weight` today. Future layers must register the GGUF names from the mapping table before use.
