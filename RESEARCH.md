# Research Notes

Living document for Odyssey research questions, hypotheses, and findings.

---

## Guiding Principles

1. **Reason first, code second** — optimize for the thinking before generation.
2. **Understanding over memorization** — systems comprehension, not repo regurgitation.
3. **Small but excellent** — prefer a well-trained 100–350M model over a poorly trained giant.
4. **Research before implementation** — read the paper first.
5. **Reproducibility** — another engineer should be able to rerun experiments from this repo.
6. **Documentation is engineering** — decisions belong in writing.

---

## Active Research Questions

### RQ1 — How do modern decoder-only transformers actually work?

Investigate embeddings, positional encoding, attention, normalization, feed-forward layers, and decoding without opaque abstractions.

**Status:** In progress (Phase 3 embeddings complete; Phases 4–9 remain)

### RQ2 — How do reasoning models differ from code-completion models?

Investigate chain-of-thought, planning, decomposition, reasoning length, and context usage.

**Status:** Not started (Phases 13–16)

### RQ3 — Can software engineering become a specialized capability?

Research architecture docs, RFCs, issues, PRs, and design documents as training signal — not only source code.

**Status:** Not started (Phase 15)

### RQ4 — Can reasoning be measured beyond loss?

Build evaluations that capture planning quality, architectural judgment, and failure modes.

**Status:** Not started (Phases 12, 16)

---

## Phase 0 Findings

- Repository layout, config loading, lint/test tooling, and experiment IDs are in place.
- No empirical ML results yet — by design.

## Phase 1 Findings

- SentencePiece Unigram is a solid reference for LLaMA-like pipelines.
- On 50k TinyStories stories, achievable vocab saturated near **17k** despite a 32k request — corpus diversity dominates `vocab_size`.
- Chat/role special tokens work cleanly as SentencePiece `user_defined_symbols` with stable core IDs (pad/bos/eos/unk = 0/1/2/3).
- Encode → decode round-trips are reliable after explicit NFKC + whitespace normalization.

## Phase 2 Findings

- Owning BPE as `odyssey_tokenizer` (library boundary) is the right shape for Phalanx Runtime reuse.
- Byte-level alphabet eliminates UTF-8 unknowns; compression on TinyStories is competitive (~4.2 chars/token at 2048 vocab).
- Pure-Python greedy encoding is correct but far slower than SentencePiece — Rust port is the performance path, not a redesign.
- Hex-encoded `merges.txt` avoids delimiter bugs with whitespace bytes.

## Phase 3 Findings

- Token embedding is a row gather from \(E \in \mathbb{R}^{V \times D}\); shape contract `(B,S)→(B,S,D)` is now enforced in code.
- Xavier uniform is a sensible default; Normal(σ=0.02) remains available for GPT-style ablations.
- At 32k×768 fp32 the table alone is ~94 MiB / 24.6M params — memory planning matters before stacking layers.
- Adding `math/` alongside `model/` keeps equations, complexity, and PyTorch↔Phalanx notes next to each phase.
- Weight tying is documented only; implement with the LM head later.

---

## Open Decisions

| Decision | Options | Status |
| --- | --- | --- |
| Experiment tracker | TensorBoard default; W&B optional | Tentative |
| Config system | Plain YAML now; Hydra optional later | Tentative |
| First model size | Odyssey Tiny (~100M) for pipeline validation | Planned |
| Final tokenizer alphabet | Bytes (current) vs GPT-2 unicode map | Bytes for now |
| Production vocab size | 32k target vs corpus-driven size | Open |
| Serving tokenizer | Python lib vs Rust port | Rust planned |
| Embedding init | Xavier uniform (default) vs Normal(0.02) | Xavier for now |
| Weight tying | Tie input E with LM head | Planned (later phase) |
