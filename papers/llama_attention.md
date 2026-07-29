# Paper Notes — LLaMA Attention

**Papers:** LLaMA (Touvron et al.), Llama 2 / 3 follow-ups.  
**Focus:** Causal decoder attention + RoPE + (later) GQA.

## Motivation

Efficient open decoder-only transformers for long-context inference.

## Key points

- Pre-norm + causal self-attention + SwiGLU FFN
- RoPE on Q/K (no absolute PE)
- Llama 2 70B / Llama 3 use GQA to shrink KV cache

## Notes for Odyssey

- Odyssey Spec v1 mirrors this stack
- `OdysseyAttention` applies RoPE when attached
- Tiny config: `H=12`, `H_kv=4`, `d=64`
