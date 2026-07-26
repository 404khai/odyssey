# Papers & Reading List

Odyssey follows a **research-before-implementation** rule. Read the original work before coding a component.

---

## Phase 0 — Ecosystem Orientation

No implementation papers required for repository setup. Background reading:

| Resource | Why |
| --- | --- |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Intuition for attention and encoder/decoder stacks |
| [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) | Core tensor / module APIs |
| [SentencePiece](https://github.com/google/sentencepiece) | Subword tokenization used in many LLMs |
| [Weights & Biases Docs](https://docs.wandb.ai/) | Optional experiment tracking |
| [Hydra](https://hydra.cc/docs/intro/) | Optional hierarchical configs |

---

## Planned Reading (later phases)

| Topic | Canonical paper / resource | Phase |
| --- | --- | --- |
| Attention | *Attention Is All You Need* (Vaswani et al., 2017) | 6+ |
| RoPE | *RoFormer* (Su et al.) | 4 |
| RMSNorm | *Root Mean Square Layer Normalization* | 5 |
| SwiGLU | *GLU Variants Improve Transformer* (Shazeer) | 7 |
| GPT-style LMs | GPT / Llama technical reports | 9–10 |
| DPO | *Direct Preference Optimization* (Rafailov et al.) | 14 |

Add notes under `papers/` as each phase begins.
