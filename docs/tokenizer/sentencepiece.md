# SentencePiece Integration Notes

## Why SentencePiece first?

LLaMA-style models popularized SentencePiece. Using it as Odyssey’s Phase 1 reference lets us:

- study Unigram vs BPE under one toolkit
- ship a working encode/decode path immediately
- collect compression / unknown / speed baselines for Phase 2

## Odyssey-specific choices

| Choice | Value | Rationale |
| --- | --- | --- |
| Model type | `unigram` (default) | Strong baseline; BPE available via config |
| Vocab size | `32000` | Matches early Odyssey config defaults |
| Character coverage | `0.9995` | Standard multilingual-friendly coverage |
| Control IDs | pad=0, bos=1, eos=2, unk=3 | Stable, explicit layout |
| Normalization | NFKC + whitespace collapse | Deterministic preprocessing outside SP |

## Files

| Artifact | Path |
| --- | --- |
| Config | `configs/tokenizer.yaml` |
| Model | `assets/tokenizer/odyssey.model` |
| Vocab | `assets/tokenizer/odyssey.vocab` |
| Metadata | `assets/tokenizer/metadata.json` |

## Round-trip expectation

For ordinary prose without unpaired unpaired control characters:

```
decode(encode(text)) ≈ normalize(text)
```

Exact whitespace fidelity depends on normalization settings. Tests assert semantic round-trips on normalized text.

## Related reading

- [`papers/sentencepiece.md`](../../papers/sentencepiece.md)
- [`papers/bpe.md`](../../papers/bpe.md)
- [`docs/tokenizer/architecture.md`](architecture.md)
