# ODY-0001 — Baseline SentencePiece tokenizer

| Field | Value |
| --- | --- |
| Phase | 1 |
| Date | 2026-07-27 |
| Purpose | Baseline SentencePiece tokenizer |
| Result | **Successful** |

## Configuration

| Knob | Value |
| --- | --- |
| Requested vocab size | 32000 |
| Actual vocab size | 17047 |
| Model type | Unigram |
| Character coverage | 0.9995 |
| Dataset | TinyStories sample (50,000 stories) |
| Config snapshot | `config.yaml` |

TinyStories is lexically homogeneous (children’s prose). SentencePiece therefore
could not materialize a hard 32k vocabulary; with `hard_vocab_limit: false` the
trainer settled at **17047** pieces. The requested 32k remains the target for
richer future corpora.

## Metrics

See `metrics.json`.

| Metric | Value |
| --- | --- |
| Training time | ~155 s |
| Vocabulary size | 17047 |
| Avg token length | ~4.30 chars |
| Compression ratio | ~4.29 chars/token |
| Unknown token frequency | ~0.0022 |
| Model file size | ~532 KB |
| Encoding speed | ~7.7e5 tokens/s (sample) |
| Decoding speed | ~5.7e6 tokens/s (sample) |

## Reproduce

```bash
source venv/bin/activate
python scripts/prepare_tinystories_sample.py --max-stories 50000
python scripts/train.py --input datasets/raw/sample.txt --vocab-size 32000
python scripts/inspect_tokenizer.py --text "Build authentication API"
```

## Lessons Learned

- Tokenizer quality is bounded by corpus diversity, not only `vocab_size`.
- Reserved chat tokens (`<system>`, `<user>`, `<assistant>`, `<mask>`) integrate cleanly as SentencePiece `user_defined_symbols`.
- Explicit Odyssey normalization + SentencePiece yields stable encode/decode round-trips.
- Soft vocab limits are required for research corpora that cannot support the configured size.

## Future Improvements

- Mix software-engineering / reasoning text into the tokenizer corpus (Phase 15 data will help).
- Compare Unigram vs BPE under identical corpora.
- Replace SentencePiece with Odyssey BPE in Phase 2 while keeping this baseline for metrics.
