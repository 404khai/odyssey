# Grouped Query Attention (GQA)

Primary Odyssey attention layout. See [attention.md](attention.md) for the full
module map and Spec links.

## Why LLaMA adopted GQA

Full multi-head attention stores a separate K/V head per query head. At long
context, the **KV cache** dominates decode memory. GQA shares each KV head
across a group of query heads, cutting cache size by `H / H_kv` with small
quality loss versus MHA (Ainslie et al., “GQA”).

Odyssey mirrors LLaMA 3 / Gemma / Mistral: configurable `num_heads` and
`num_kv_heads` with `num_heads % num_kv_heads == 0`.
