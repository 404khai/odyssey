# Experiments

This directory stores Odyssey experiment artifacts and write-ups.

---

## Experiment IDs

Format:

```
ODY-XXXX
```

| Example | Meaning |
| --- | --- |
| `ODY-0000` | Repository initialization |
| `ODY-0001` | First research experiment |
| `ODY-0012` | Twelfth logged experiment |

IDs are monotonic. Never reuse an ID.

---

## Naming Conventions

Directory per experiment:

```
experiments/ODY-XXXX/
├── README.md          # purpose, setup, results, lessons
├── config.yaml        # snapshot of config used (optional)
├── metrics.json       # scalar metrics (optional)
└── notes.md           # free-form notes (optional)
```

Experiment title style:

```
ODY-XXXX — short-kebab-or-plain description
```

---

## Logging

Every experiment should record:

1. **Purpose** — what question does this answer?
2. **Config** — path or snapshot of hyperparameters
3. **Seed** — for reproducibility
4. **Environment** — Python version, key package versions, hardware notes
5. **Commands** — exact launch commands
6. **Result** — success / failure / inconclusive
7. **Metrics** — loss, perplexity, custom scores
8. **Lessons** — what to change next

Also add a summary row to [`EXPERIMENTS.md`](../EXPERIMENTS.md) at the repo root.

---

## Metrics

Preferred early metrics:

| Metric | When |
| --- | --- |
| Train / val loss | Training phases |
| Perplexity | Evaluation phases |
| Token accuracy | Diagnostic |
| Custom reasoning scores | Later benchmarks |

Log scalars to TensorBoard by default. Weights & Biases is optional (`pip install -e ".[tracking]"`).

---

## Future Reports

As the project matures, experiment folders may include:

- training curves
- evaluation tables
- qualitative reasoning samples
- failure analyses
- comparison against prior `ODY-XXXX` runs

Keep reports short and decision-oriented. The goal is cumulative research memory, not vanity dashboards.
