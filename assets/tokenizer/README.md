# Tokenizer Artifacts

## Odyssey BPE (Phase 2 — primary)

```bash
odyssey-tokenizer train --input datasets/raw/sample.txt --vocab-size 2048 --max-lines 1000
```

Produces:

- `bpe/odyssey.model/{vocab.json,merges.txt,config.json,metadata.json}`
- `bpe/merge_visualization.png`
- `bpe/compression_graph.png`

## SentencePiece reference (Phase 1)

```bash
python scripts/train.py --input datasets/raw/sample.txt --vocab-size 32000
```

Produces `odyssey.model` / `odyssey.vocab` (gitignored binaries; regenerate locally).
