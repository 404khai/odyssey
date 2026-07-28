# Papers & Reading List

Odyssey follows a **research-before-implementation** rule. Read the original work before coding a component.

---

## Phase 0 — Ecosystem Orientation

| Resource | Why |
| --- | --- |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Intuition for attention and encoder/decoder stacks |
| [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) | Core tensor / module APIs |
| [SentencePiece](https://github.com/google/sentencepiece) | Subword tokenization used in many LLMs |
| [Weights & Biases Docs](https://docs.wandb.ai/) | Optional experiment tracking |
| [Hydra](https://hydra.cc/docs/intro/) | Optional hierarchical configs |

---

## Phase 2 — Tokenization Research *(complete)*

| Paper / Study | Summary |
| --- | --- |
| SentencePiece (Kudo & Richardson) | [papers/sentencepiece.md](papers/sentencepiece.md) — revisited vs Odyssey BPE |
| BPE for rare words (Sennrich et al.) | [papers/bpe.md](papers/bpe.md) — implemented in `odyssey_tokenizer` |
| GPT-2 byte-level BPE | [papers/gpt2-tokenizer.md](papers/gpt2-tokenizer.md) |
| TikToken | [papers/tiktoken.md](papers/tiktoken.md) — speed target for Rust port |

---

## Phase 3 — Embeddings *(complete)*

| Paper / Study | Summary |
| --- | --- |
| Attention Is All You Need — Input Embeddings | [papers/transformer_embeddings.md](papers/transformer_embeddings.md) |
| Word2Vec (Mikolov et al.) | [papers/word2vec.md](papers/word2vec.md) |
| GloVe (Pennington et al.) | [papers/glove.md](papers/glove.md) |

Math companion: [math/embeddings.md](math/embeddings.md)

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
