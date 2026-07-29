# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors:** Jianlin Su et al.  
**Link:** [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

---

## Motivation

Absolute position embeddings (learned or sinusoidal) inject position as an additive vector. They do not naturally encode **relative** offsets inside attention scores and extrapolate poorly beyond the training context.

---

## Mathematics

Represent a 2D feature pair as a complex number and multiply by a position-dependent phase:

\[
(x_{2i} + i x_{2i+1})\, e^{i m \theta_i}
\]

with \(\theta_i = 10000^{-2i/d}\). In real arithmetic this is the 2×2 rotation used by Odyssey / LLaMA / Phalanx.

---

## Rotation Derivation

The relative rotation between positions \(m\) and \(n\) depends on \(m-n\), so attention logits gain relative positional structure without extra bias parameters.

---

## Advantages

- No trainable position table
- Built-in relative geometry
- Efficient via cos/sin caches
- Strong length extrapolation vs absolute embeddings

---

## Weaknesses

- Pairing scheme and θ schedule are design choices
- Very long contexts still need scaling tricks (NTK, YaRN — deferred)

---

## Notes for Odyssey

Odyssey Phase 4 implements the LLaMA adjacent-pair form of RoPE and validates numerically against Phalanx Runtime (`scripts/validate_rope.py`).
