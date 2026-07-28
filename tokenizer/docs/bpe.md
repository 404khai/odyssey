# Odyssey Byte-Level BPE

## Algorithm

1. Normalize text (NFKC, whitespace policy).
2. Encode to UTF-8 bytes.
3. Seed vocabulary with reserved special tokens + all 256 bytes.
4. Count adjacent symbol-pair frequencies across the corpus (weighted by line frequency).
5. Greedily merge the highest-frequency pair (deterministic tie-break).
6. Repeat until `vocab_size` or `min_frequency` stops progress.
7. Export `vocab.json` + `merges.txt`.

Encoding applies merges in rank order (lowest rank first), matching GPT-2-style BPE application.

## Complexity

| Phase | Time | Notes |
| --- | --- | --- |
| Pair counting | O(N) per merge step | N = total symbols across unique lines |
| Merge application | O(N) per merge step | Rebuilds symbol sequences |
| Encoding | O(T · M′) | T tokens, M′ candidate merges checked greedily |

The Python trainer prioritizes clarity. Future Rust / SIMD ports keep the same artifacts.

## Tradeoffs

| Strength | Weakness |
| --- | --- |
| No `<unk>` for arbitrary UTF-8 (byte alphabet) | Pure-Python training is slower than SentencePiece C++ |
| Deterministic, inspectable merges | Greedy merges are locally optimal only |
| Library boundary ready for Phalanx Runtime | Regex pre-tokenization (GPT-2 style) not yet added |

## Implementation map

| Module | Role |
| --- | --- |
| `vocabulary.py` | Token ↔ ID map |
| `merges.py` | Ordered merge table |
| `trainer.py` | Frequency counting + greedy merges |
| `encoder.py` / `decoder.py` | Text ↔ IDs |
| `serialization.py` | Model directory I/O |
| `tokenizer.py` | Public `OdysseyTokenizer` API |

## Future improvements

- Regex pre-tokenizer for code-heavy corpora
- Incremental pair-count updates / parallel training
- Rust port with identical `merges.txt` semantics for Phalanx Runtime
