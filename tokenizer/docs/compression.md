# Compression

Compression ratio is reported as **characters per token** on normalized text:

```
ratio = characters / tokens
reduction% = (1 - tokens/characters) * 100
```

Example:

```
Characters 238
Tokens     129
Ratio      1.845 chars/token
Reduction  45.8%
```

Byte-level BPE usually improves ratio as `vocab_size` grows, until the corpus cannot support additional productive merges (`min_frequency` stop).

Compare against SentencePiece (`ODY-0001`) on the same sample texts when evaluating tokenizer quality for Odyssey.
