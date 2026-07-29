# Attention (Multi-Head & Grouped Query)

## Motivation

Embeddings give each token an independent vector. Attention lets tokens
exchange information so later positions can condition on earlier context.
Odyssey uses **Grouped Query Attention (GQA)** as the primary path (LLaMA 3 /
Gemma / Mistral style), with classic Multi-Head Attention documented as the
`H_kv = H` special case.

## Mathematics

See Spec [`spec/attention.md`](../../spec/attention.md).

\[
Q = x W_Q^\top,\quad K = x W_K^\top,\quad V = x W_V^\top
\]

\[
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
\]

Causal mask \(M\): \(M_{s,t}=0\) if \(t\le s\), else \(-\infty\).

GQA: \(H/H_{kv}\) query heads share one KV head → smaller KV cache and fewer
K/V parameters than full MHA.

## Implementation

| Module | Role |
| --- | --- |
| `model/projection.py` | Q/K/V/O linears (no bias) |
| `model/causal_mask.py` | Additive causal mask |
| `model/softmax.py` | Stable Softmax |
| `model/attention_math.py` | Scale, reshape, GQA expand |
| `model/gqa.py` | SDPA kernel |
| `model/attention.py` | `OdysseyAttention` (+ MHA wrapper) |
| `configs/model.yaml` | `attention:` |

```python
from model import OdysseyAttention, load_attention_config, OdysseyRoPE, RopeConfig
cfg = load_attention_config()
rope = OdysseyRoPE(RopeConfig(head_dim=cfg.head_dim, rotary_dim=cfg.head_dim))
attn = OdysseyAttention(cfg, rope=rope)
y = attn(x)  # (B,S,D) → (B,S,D)
```

## MHA vs GQA vs MQA

| Mode | Condition | KV params |
| --- | --- | --- |
| MHA | `H_kv = H` | Full |
| GQA | `1 < H_kv < H` | Reduced by `H/H_kv` |
| MQA | `H_kv = 1` | Minimal |

Tiny Odyssey: `H=12`, `H_kv=4`, `d=64`.

## Complexity

- Time: \(O(B H S^2 d)\)
- Score memory: \(O(B H S^2)\)
- GQA reduces KV-cache memory by \(H/H_{kv}\) vs MHA at decode time.

## Phalanx Compatibility

```bash
python scripts/validate_attention.py
# or: python ../validation/test_attention.py
```

Tolerance default **`1e-3`** (GEMM accumulation; mean error typically ≪ `1e-6`).
Report: `experiments/ODY-0007/attention_validation.json`.

## Future (documented only)

FlashAttention / FlashAttention-2/3, paged attention, sliding window, Triton/CUDA kernels.
