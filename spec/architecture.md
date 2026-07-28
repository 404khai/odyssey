# Architecture

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Define the Odyssey decoder-only Transformer as a model family: configuration keys, block structure, residual order, and non-negotiable invariants shared by training and inference.

---

## Theory

Odyssey is a **decoder-only** causal language model. Tokens become dense vectors; each layer applies pre-norm self-attention and a gated feed-forward network with residual connections; a final norm and linear head produce next-token logits.

Family members (Tiny / Base / …) share the same topology and differ only by scale hyperparameters published in model metadata.

---

## Mathematics (block)

For layer input \(x\):

\[
\begin{aligned}
h &= x + \mathrm{Attn}(\mathrm{RMSNorm}(x)) \\
y &= h + \mathrm{FFN}(\mathrm{RMSNorm}(h))
\end{aligned}
\]

RoPE is applied inside attention to Q and K. After \(L\) layers:

\[
\mathrm{logits} = \mathrm{LMHead}(\mathrm{RMSNorm}(y_L))
\]

---

## Model Configuration (frozen keys)

Every Odyssey checkpoint / GGUF **must** expose these logical keys (names are stable; storage encoding may vary):

| Key | Type | Meaning |
| --- | --- | --- |
| `spec_version` | string | Odyssey Spec semver, e.g. `1.0.0` |
| `architecture` | string | `odyssey` (GGUF may use `llama` layout compatibility — see [gguf_mapping.md](gguf_mapping.md)) |
| `vocab_size` | int | \(V\) |
| `hidden_size` | int | \(D\) |
| `intermediate_size` | int | \(I\) (SwiGLU inner) |
| `num_layers` | int | \(L\) |
| `num_heads` | int | \(H\) query heads |
| `num_kv_heads` | int | \(H_{kv}\) key/value heads (\(H \bmod H_{kv} = 0\)) |
| `head_dim` | int | \(d = D / H\) |
| `context_length` | int | Max positions \(S_{\max}\) |
| `rope_theta` | float | RoPE base \(\theta\) |
| `rope_dim` | int | Rotary dims (even, \(\le d\)); default \(d\) |
| `rope_scaling.type` | string | `none` \| `linear` |
| `rope_scaling.factor` | float | Present when type ≠ `none` |
| `rms_norm_eps` | float | \(\varepsilon\) |
| `activation` | string | `swiglu` |
| `norm_type` | string | `rmsnorm` |
| `attention_layout` | string | `pre_norm` |
| `weight_tying` | bool | Share embedding with LM head |
| `tokenizer_format` | string | `odyssey-bpe` |
| `tokenizer_version` | int | Artifact format version |

Optional training-only keys (not required at inference): init strategy, dropout, optimizer.

### Derived invariants

\[
D = H \cdot d, \quad H = k \cdot H_{kv}\ (k \in \mathbb{Z}^+), \quad
\mathrm{kv\_dim} = H_{kv} \cdot d
\]

---

## Residual Order (frozen)

**Pre-norm only.** Never post-norm in Spec v1.

```text
x
 → RMSNorm → Attention (+ RoPE on Q/K) → + residual
 → RMSNorm → SwiGLU FFN                 → + residual
```

Final RMSNorm before LM head. No LayerNorm. No bias on attention/FFN projections in Spec v1 (bias-free Llama-style).

---

## Attention Layout

- Causal mask (no future tokens)
- Multi-head or GQA via `num_kv_heads`
- Scaled dot-product with scale \(1/\sqrt{d}\)
- RoPE on Q/K only

---

## Data Types

| Phase | Allowed |
| --- | --- |
| Training master weights | float32 (bf16/fp16 compute optional) |
| Serving | float32, float16, or quantized GGUF types as published |

The runtime materializes compute tensors according to weight storage; it does not invent dtypes.

---

## Initialization (training)

| Tensor class | Default |
| --- | --- |
| Token embeddings | Xavier uniform (Normal σ=0.02 allowed) |
| Linear projections | Xavier / Kaiming as implemented in training code — must be recorded in experiment metadata |
| RMSNorm \(\gamma\) | Ones |
| Padding embedding row | Zeros if `padding_idx` set |

Initialization is **not** required for inference.

---

## Tensor Shapes

See [tensor_shapes.md](tensor_shapes.md).

---

## Implementation Notes

- Spec is language-agnostic.
- Odyssey currently implements tokenizer + embeddings; later phases must match this residual order.
- Phalanx implements embedding gather + RoPE today; remaining layers fill the same skeleton.

---

## Examples

Odyssey Tiny:

```text
V=32000 D=768 I=2048 L=12 H=12 H_kv=12 d=64 S_max=2048 θ=10000
```

GQA example (future scale):

```text
H=32 H_kv=8 d=128  →  4 queries share each KV head
```

---

## Future Extensions (require Spec ≥ 2.0 if breaking)

- YaRN / NTK RoPE scaling
- Biases on projections
- Sliding-window attention
- MoE experts
- Alternate activations (GELU-only FFN)

---

## Compatibility Notes

Phalanx Runtime supports Spec `1.0.0` as the target contract. Partial layer implementation is tracked in `runtime/docs/spec-compliance.md` — incomplete kernels do not authorize shape or name drift.
