# Paper Notes — Attention Is All You Need

**Paper:** Vaswani et al., 2017.  
**Focus:** Scaled Dot-Product Attention, Multi-Head Attention.

## Motivation

Replace recurrence with content-based pairwise interactions so every position
can attend to every other in parallel.

## Key points

- Scores = `Q Kᵀ / √d_k` then Softmax → weights on `V`
- Multi-head: split channels so heads specialize
- Causal (masked) attention for autoregressive decoding

## Notes for Odyssey

- Spec formula matches the paper’s SDPA with an additive causal mask
- Softmax computed in float32 with max-subtraction (`model/softmax.py`)
- Cross-check: `scripts/validate_attention.py` vs Phalanx
