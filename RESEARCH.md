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

**Status:** Not started (Phases 3–9)

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

---

## Open Decisions

| Decision | Options | Status |
| --- | --- | --- |
| Experiment tracker | TensorBoard default; W&B optional | Tentative |
| Config system | Plain YAML now; Hydra optional later | Tentative |
| First model size | Odyssey Tiny (~100M) for pipeline validation | Planned |
