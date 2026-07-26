# Odyssey Model Card

> Placeholders — fill in as training and evaluation progress.

---

## Model Details

| Field | Value |
| --- | --- |
| Model name | Odyssey |
| Version | v0.0.0 (repository only) |
| Architecture | Decoder-only Transformer *(planned)* |
| Parameters | TBD |
| Context length | TBD (config default: 2048) |
| Vocabulary size | TBD (config default: 32000) |
| Tokenizer | TBD (SentencePiece planned) |
| License | MIT |
| Developers | Odyssey Contributors |
| Languages | English *(planned)* |

---

## Intended Use

### Primary intended uses

- Long-horizon reasoning research
- Software architecture and systems design assistance
- Autonomous software engineering research
- Educational implementation of modern transformers

### Out-of-scope uses

- Real-time code autocomplete as a primary product
- General-purpose chat without domain focus
- Safety-critical autonomous deployment without human oversight
- Generating or assisting with illegal activity

---

## Training Data

| Field | Value |
| --- | --- |
| Pretraining corpus | TBD |
| Instruction data | TBD |
| Preference data | TBD |
| Software engineering data | TBD |
| Data cutoff | TBD |

---

## Evaluation

| Benchmark | Metric | Score | Notes |
| --- | --- | --- | --- |
| Perplexity | — | TBD | Phase 12 |
| Reasoning suite | — | TBD | Phase 16 |
| SE planning tasks | — | TBD | Phase 15–16 |

---

## Limitations

- No trained weights in Phase 0
- Capabilities, biases, and failure modes are unknown until evaluation phases
- Small research models will not match frontier systems on broad knowledge

---

## Ethical Considerations

- Document dataset provenance before training
- Measure and report failure modes on reasoning / SE tasks
- Prefer transparent research artifacts over opaque releases

---

## Citation

```bibtex
@software{odyssey2026,
  title  = {Odyssey: A Reasoning-Oriented Decoder-Only Transformer},
  year   = {2026},
  note   = {Research project — Phase 0 repository initialization},
  url    = {https://github.com/PLACEHOLDER/odyssey}
}
```
