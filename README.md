# Odyssey

### A decoder-only transformer specializing in long-horizon reasoning, software architecture, and autonomous software engineering.

*"Think before you build."*

---

**Status:** Research Project — Phase 3 complete · **Spec v1.0.0** frozen  
**Language:** Python 3.12+  
**Framework:** PyTorch  
**Target Runtime:** [Phalanx Runtime](https://github.com/404khai/phalanx)  
**Architecture:** Decoder-only Transformer

---

## Odyssey Specification (source of truth)

Architecture, tensor names, shapes, tokenizer, GGUF mapping, and the Phalanx runtime contract live in:

**[`spec/`](spec/README.md)** — Specification **`1.0.0`**

Phalanx Runtime must follow this spec; pedagogical notes in [`math/`](math/README.md) are non-normative.

---

## Repository Overview

Odyssey is a research repository for building a small, carefully engineered decoder-only language model optimized for **reasoning**, not autocomplete.

| Phase | Deliverable |
| --- | --- |
| 0 | Research repository foundation |
| 1 | SentencePiece **reference** tokenizer |
| 2 | **Owned** byte-level BPE library (`odyssey_tokenizer`) |
| 3 | **Token embedding layer** (`OdysseyEmbedding`) |

```mermaid
flowchart TD
    RawText[Raw Text] --> Tokenizer
    Tokenizer --> TokenIDs[Token IDs]
    TokenIDs --> EmbeddingLookup[Embedding Lookup]
    EmbeddingMatrix[Embedding Matrix] --> EmbeddingLookup
    EmbeddingLookup --> Vectors[Embedding Vectors]
    Vectors --> RoPE[RoPE Phase 4]
    RoPE --> TransformerBlock[Transformer Block]
```

---

## Embedding Layer (Phase 3)

```python
from model import EmbeddingConfig, OdysseyEmbedding, load_embedding_config

config = load_embedding_config()  # configs/embedding.yaml
emb = OdysseyEmbedding(config)
x = emb(token_ids)  # (batch, seq) → (batch, seq, hidden)
print(emb.inspect().format())
```

| Knob | Default |
| --- | --- |
| Vocabulary | 32,000 |
| Hidden size | 768 |
| Parameters | 24,576,000 |
| Init | Xavier uniform |
| Memory (fp32) | ~93.75 MiB |

Math notes: [`math/embeddings.md`](math/embeddings.md) · Architecture: [`docs/architecture/embeddings.md`](docs/architecture/embeddings.md)

```bash
python scripts/benchmark_embeddings.py --visualize
```

---

## Odyssey Tokenizer

The tokenizer is a **reusable library**, not model-coupled code:

```
tokenizer/
├── odyssey_tokenizer/   # import odyssey_tokenizer
├── cli/                 # odyssey-tokenizer
├── benchmarks/
├── tests/
├── docs/
└── sentencepiece/       # Phase 1 reference only
```

### Public API

```python
from odyssey_tokenizer import OdysseyTokenizer

tokenizer = OdysseyTokenizer.load("assets/tokenizer/bpe/odyssey.model")
ids = tokenizer.encode("Build authentication API")
text = tokenizer.decode(ids)
```

Training and Phalanx Runtime can share the same artifacts (`vocab.json` + `merges.txt`). A future Rust port should preserve identical behavior.

### Training

```bash
source venv/bin/activate
python scripts/prepare_tinystories_sample.py --max-stories 50000

odyssey-tokenizer train \
  --input datasets/raw/sample.txt \
  --vocab-size 2048 \
  --max-lines 1000 \
  --output assets/tokenizer/bpe/odyssey.model
```

### CLI Usage

```bash
odyssey-tokenizer encode --model assets/tokenizer/bpe/odyssey.model --text "Hello"
odyssey-tokenizer decode --model assets/tokenizer/bpe/odyssey.model --ids 12,45,90
odyssey-tokenizer inspect --model assets/tokenizer/bpe/odyssey.model --text "Build authentication API" --show-merges
odyssey-tokenizer benchmark --model assets/tokenizer/bpe/odyssey.model --input datasets/raw/sample.txt --limit 200
odyssey-tokenizer visualize --model assets/tokenizer/bpe/odyssey.model --input datasets/raw/sample.txt
```

Docs: [tokenizer/README.md](tokenizer/README.md)

---

## Math Notes

Equation-level companions for each neural component live in [`math/`](math/README.md) (embeddings, RoPE outline, attention outline, …), including PyTorch vs Phalanx Runtime execution notes.

---

## Vision

Odyssey exists to explore what an AI model looks like when it is optimized not for code completion, but for **thinking like a senior software architect.**

---

## Goals

- Deliberate reasoning before code generation
- Software architecture and systems design capability
- Reproducible research experiments
- Clear documentation of every architectural decision
- Small but excellent models (Tiny → Base → Pro → Max)

---

## Installation

```bash
cd odyssey
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Verify

```bash
pytest
black --check .
ruff check .
isort --check-only .
mypy odyssey model tokenizer training evaluation
MYPYPATH=tokenizer mypy --explicit-package-bases -p odyssey_tokenizer
```

---

## Repository Progress

| Phase | Focus | Status |
| --- | --- | --- |
| 0 | Repository setup | **Complete** |
| 1 | SentencePiece reference | **Complete** |
| 2 | Odyssey BPE library | **Complete** |
| 3 | Embedding layer | **Complete** |
| 4–20 | RoPE → Odyssey v1 | Planned |

---

## Experiment Tracking

| ID | Purpose | Result |
| --- | --- | --- |
| ODY-0000 | Repository initialization | Successful |
| ODY-0001 | SentencePiece baseline | Successful |
| ODY-0002 | Odyssey BPE tokenizer | Successful |
| ODY-0003 | Token embedding layer | Successful |

---

## License

MIT — see [LICENSE](LICENSE).
