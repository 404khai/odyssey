# Tokenizer

Phase 1 ships a **SentencePiece reference tokenizer**.  
Phase 2 will replace the internals with Odyssey’s own BPE implementation.

---

## Layout

```
tokenizer/
├── __init__.py
├── README.md
└── sentencepiece/
    ├── config.py           # YAML → TokenizerConfig
    ├── special_tokens.py   # Reserved token documentation
    ├── normalizer.py       # Unicode / whitespace cleanup
    ├── trainer.py          # SentencePiece training
    ├── encoder.py          # Text → IDs
    ├── decoder.py          # IDs → Text
    ├── tokenizer.py        # Public API
    └── utils.py            # Shared helpers
```

---

## Special Tokens

| Name | Surface | ID | Purpose |
| --- | --- | --- | --- |
| pad | `<pad>` | 0 | Batch padding |
| bos | `<bos>` | 1 | Begin sequence |
| eos | `<eos>` | 2 | End sequence / stop |
| unk | `<unk>` | 3 | Unknown fallback |
| mask | `<mask>` | dynamic | Masking experiments |
| system | `<system>` | dynamic | Chat system turn |
| user | `<user>` | dynamic | Chat user turn |
| assistant | `<assistant>` | dynamic | Chat assistant turn |

IDs for user-defined symbols are assigned by SentencePiece after the core controls.

---

## Train

```bash
# Optional: fetch TinyStories sample used by ODY-0001
python scripts/prepare_tinystories_sample.py --max-stories 5000

python scripts/train.py \
  --input datasets/raw/sample.txt \
  --vocab-size 32000
```

Artifacts land under `assets/tokenizer/`:

- `odyssey.model`
- `odyssey.vocab`
- `metadata.json`

---

## Inspect

```bash
python scripts/inspect_tokenizer.py \
  --model assets/tokenizer/odyssey.model \
  --text "Build authentication API" \
  --show-specials
```

---

## Python API

```python
from tokenizer import OdysseySentencePieceTokenizer, load_tokenizer_config

config = load_tokenizer_config()
tok = OdysseySentencePieceTokenizer(config)
tok.train("datasets/raw/sample.txt", vocab_size=8000)

ids = tok.encode("Build authentication API")
text = tok.decode(ids)
print(tok.inspect("Build authentication API").render())
```

---

## Configuration

All knobs live in [`configs/tokenizer.yaml`](../configs/tokenizer.yaml).

Nothing tokenizer-related should be hardcoded at call sites.
