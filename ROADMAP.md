# Odyssey Roadmap

Development proceeds **one phase at a time**. Complete, document, commit, then wait for approval before continuing.

---

## Phase 0 — Repository Setup *(complete)*

- Project structure
- Tooling (Python 3.12, PyTorch stack, lint/test)
- Documentation
- Experiment tracking
- Config system (`configs/default.yaml`)

---

## Phase 1 — Tokenizer Research *(complete)*

- SentencePiece reference pipeline
- Vocabulary design + special tokens
- Normalization strategies
- Paper summaries + ODY-0001 baseline

---

## Phase 2 — Custom Tokenizer *(complete)*

- Byte-level BPE from first principles
- Reusable `odyssey_tokenizer` library + CLI
- Serialization, inspector, benchmarks, visualizations
- ODY-0002 baseline

---

## Phase 3 — Embedding Layer *(complete)*

- Token embeddings (`OdysseyEmbedding`)
- Configurable weight initialization
- Shape validation, inspector, benchmarks
- Math notes under `math/`
- ODY-0003 baseline

---

## Phase 4 — Rotary Positional Embeddings

- RoPE implementation
- Visualization
- Tests

---

## Phase 5 — RMSNorm & Residuals

- RMSNorm
- Residual connections

---

## Phase 6 — Multi-Head Attention

- Causal masking
- Scaled dot-product attention

---

## Phase 7 — SwiGLU Feed Forward

- Activation research
- FFN implementation

---

## Phase 8 — Transformer Block

- Residual wiring
- Decoder block assembly

---

## Phase 9 — Full Decoder

- End-to-end forward pass
- Basic inference

---

## Phase 10 — Training Pipeline

- Cross-entropy loss
- Optimizer & scheduler
- Mixed precision

---

## Phase 11 — Checkpointing

- Save / resume training
- Logging

---

## Phase 12 — Evaluation Pipeline

- Perplexity
- Reasoning benchmarks

---

## Phase 13 — Instruction Tuning

- Conversation formatting
- Assistant behavior

---

## Phase 14 — Preference Optimization

- DPO
- Alignment evaluation

---

## Phase 15 — Software Engineering Dataset

- Planning data
- Architecture data

---

## Phase 16 — Reasoning Evaluation

- Internal benchmarks
- Failure analysis

---

## Phase 17 — Model Optimization

- Inference improvements
- Memory optimizations

---

## Phase 18 — Odyssey v1 Candidate

- Full evaluation
- Model card
- Release notes

---

## Phase 19 — Community Feedback

- Bug fixes
- Research improvements

---

## Phase 20 — Odyssey v1 Release

- Documentation
- Benchmarks
- Weights
- Final report

---

## Future Model Family

| Variant | Approx. size | Purpose |
| --- | --- | --- |
| Odyssey Tiny | ~100M | Pipeline validation, education |
| Odyssey Base | ~350M | Primary research target |
| Odyssey Pro | TBD | Stronger reasoning |
| Odyssey Max | TBD | Flagship (later) |
