# Paper Notes — GQA

**Paper:** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.  
**Focus:** Share KV heads across query-head groups.

## Motivation

MQA (`H_kv=1`) is fast but can hurt quality; MHA is accurate but KV-cache heavy.
GQA interpolates.

## Mathematics

Partition `H` query heads into `H_kv` groups; each group shares one K/V head.
At inference, expand KV by repeating each head `H/H_kv` times before SDPA
(or index into the shared head).

## Tradeoffs

| | Quality | KV memory | Speed |
| --- | --- | --- | --- |
| MHA | Best | Highest | Baseline |
| GQA | Near-MHA | `H_kv/H` of MHA | Faster decode |
| MQA | Lower | Minimal | Fastest |

## Memory savings

KV cache elements per layer ~ `2 · B · H_kv · T · d` (vs `H` for MHA).

## Why Odyssey uses GQA

Matches modern open models (LLaMA 3, Gemma, Mistral, Qwen) and Phalanx Runtime
`AttentionConfig` / `layers::Attention`, while remaining a strict
generalization of MHA (`H_kv=H`) and MQA (`H_kv=1`).
