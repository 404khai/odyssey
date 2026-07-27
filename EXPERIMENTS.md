# Experiment Log

Canonical index of Odyssey experiments. Detailed notes live under `experiments/ODY-XXXX/`.

Naming and logging conventions: [experiments/README.md](experiments/README.md).

---

## ODY-0000 — Repository initialization

| Field | Value |
| --- | --- |
| ID | ODY-0000 |
| Date | 2026-07-26 |
| Phase | 0 |
| Purpose | Establish research repository infrastructure |
| Config | `configs/default.yaml` |
| Result | **Successful** |
| Metrics | N/A (no training) |
| Lessons | Research infrastructure is now established. Maintainability starts before the first model layer. |

Details: [experiments/ODY-0000/README.md](experiments/ODY-0000/README.md)

---

## ODY-0001 — Baseline SentencePiece tokenizer

| Field | Value |
| --- | --- |
| ID | ODY-0001 |
| Date | 2026-07-27 |
| Phase | 1 |
| Purpose | Baseline SentencePiece tokenizer |
| Config | `configs/tokenizer.yaml` (snapshot in experiment folder) |
| Dataset | TinyStories sample (50,000 stories) |
| Model | Unigram |
| Requested vocab | 32000 |
| Actual vocab | 17047 |
| Result | **Successful** |
| Metrics | Training ~155s; compression ~4.29 chars/token; unk rate ~0.0022 |
| Lessons | Corpus diversity bounds achievable vocabulary size; soft vocab limits are required for homogeneous research samples. |

Details: [experiments/ODY-0001/README.md](experiments/ODY-0001/README.md)
