# ODY-0002 — Odyssey BPE tokenizer (from first principles)

| Field | Value |
| --- | --- |
| Phase | 2 |
| Date | 2026-07-28 |
| Purpose | Implement owned byte-level BPE library |
| Result | **Successful** |

## Configuration

| Knob | Value |
| --- | --- |
| Algorithm | Byte-level BPE |
| Requested / config default | 32000 |
| Experiment vocab size | **2048** |
| Merges learned | 1782 |
| Dataset | TinyStories sample (`max_lines=1000`) |
| Min frequency | 2 |
| Library API | `odyssey_tokenizer.OdysseyTokenizer` |

The YAML default remains `vocab_size: 32000`. This experiment used a smaller
vocab + line cap so the pure-Python trainer finishes in minutes while still
exercising the full pipeline (train → serialize → load → encode/decode →
benchmark → visualize). Larger runs:

```bash
odyssey-tokenizer train --input datasets/raw/sample.txt --vocab-size 32000 --max-lines 0
```

## Metrics

See `metrics.json`.

| Metric | Approx. value |
| --- | --- |
| Training time | ~157 s (1000 lines → 2048 vocab) |
| Vocabulary size | 2048 |
| Compression ratio | ~4.16 chars/token (200-line bench) |
| Compression reduction | ~76% |
| Unknown token rate | 0.0 (byte-level) |
| Encode speed | ~900–4000 tok/s (Python greedy merges) |
| Decode speed | ~6e6 tok/s |
| Peak memory (bench) | ~0.5 MB traced |

## Library layout

```
tokenizer/
├── odyssey_tokenizer/   # reusable package
├── cli/                 # odyssey-tokenizer
├── benchmarks/
├── tests/
└── docs/
```

```python
from odyssey_tokenizer import OdysseyTokenizer
tok = OdysseyTokenizer.load("assets/tokenizer/bpe/odyssey.model")
ids = tok.encode("Build authentication API")
text = tok.decode(ids)
```

## Strengths

- Fully owned implementation (no SentencePiece required for Odyssey training)
- Deterministic merges + stable serialization (`vocab.json` / `merges.txt`)
- Clean library boundary for Phalanx Runtime / future Rust port
- Byte alphabet → no UTF-8 unknowns

## Weaknesses

- Pure-Python encode is slower than SentencePiece / tiktoken
- Greedy BPE is locally optimal only
- No GPT-2-style regex pre-tokenizer yet (code tokenization will improve later)

## Ideas for optimization

- Rust port with identical merge semantics
- Incremental heap-based pair selection
- Regex pre-tokenization for code
- Parallel corpus scanning

## Artifacts

- Model: `assets/tokenizer/bpe/odyssey.model/`
- Plots: `assets/tokenizer/bpe/merge_visualization.png`, `compression_graph.png`
- Config snapshot: `config.yaml`
- Bench: `metrics.json`
