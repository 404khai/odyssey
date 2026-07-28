# Attention Is All You Need — Input Embeddings

**Authors:** Ashish Vaswani et al.  
**Link:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
**Focus section:** Input Embeddings / Embeddings and Softmax

---

## Problem

Sequence models need a continuous representation of discrete tokens before attention and feed-forward layers can operate.

---

## Motivation

The Transformer replaces recurrence with attention, but still begins with a learned embedding that maps token ids to vectors of size \(d_{\mathrm{model}}\). Those vectors are scaled by \(\sqrt{d_{\mathrm{model}}}\) in the original paper before adding positional encodings.

---

## Algorithm (embedding portion)

1. Maintain a matrix \(E \in \mathbb{R}^{V \times d_{\mathrm{model}}}\).
2. Lookup row \(E[t]\) for each token \(t\).
3. (Original Transformer) multiply by \(\sqrt{d_{\mathrm{model}}}\) and add sinusoidal position encodings.
4. Share embedding weights with the pre-softmax linear projection when weight tying is used.

Odyssey Phase 3 implements the lookup table and initialization. Positional information arrives in Phase 4 via **RoPE** (not absolute sinusoids).

---

## Advantages

- Simple, differentiable token interface
- Weight tying reduces parameters and couples input/output spaces
- Works for any discrete vocabulary (BPE, Unigram, etc.)

---

## Disadvantages

- Absolute embedding + sinusoid positions (original) extrapolate poorly vs RoPE
- Embedding table dominates parameter count in small models
- Random init has no semantic structure until trained

---

## Implementation Notes

- PyTorch: `nn.Embedding`
- Odyssey: `OdysseyEmbedding` with configurable init and `padding_idx`
- Phalanx Runtime: `EmbeddingTable` gather from GGUF `token_embd.weight`

---

## Lessons Learned

- Embeddings are a **lookup**, not a mysterious first layer.
- Shape contract `(B, S) → (B, S, D)` must be locked before stacking blocks.
- Scaling and positional scheme are architectural choices separate from the table itself.

---

## How Odyssey Will Use This Knowledge

Phase 3 owns the embedding matrix and docs weight tying for later. Phase 4 replaces absolute positions with RoPE, closer to modern decoder-only LMs than the 2017 additive sinusoids.
