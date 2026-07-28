# Odyssey Tokenizer Library

Reusable **byte-level BPE** tokenizer owned by Odyssey.

```
tokenizer/
├── odyssey_tokenizer/   # importable library
├── cli/                 # `odyssey-tokenizer` entrypoint
├── benchmarks/          # encode/decode / memory suite
├── tests/               # library unit + integration tests
├── docs/                # algorithm & architecture notes
├── sentencepiece/       # Phase 1 reference backend
└── README.md
```

## Why a separate library?

Training and inference must share **identical** tokenization.

Phalanx Runtime can consume this Python library today and later a Rust port with
the same `vocab.json` + `merges.txt` artifacts — without depending on the model
package layout.

## Public API

```python
from odyssey_tokenizer import OdysseyTokenizer

tokenizer = OdysseyTokenizer.load("assets/tokenizer/bpe/odyssey.model")
ids = tokenizer.encode("Build authentication API")
text = tokenizer.decode(ids)
print(tokenizer.inspect("Build authentication API").render())
```

Train:

```python
from odyssey_tokenizer import OdysseyTokenizer, load_bpe_config

config = load_bpe_config()
config.vocab_size = 4096
config.training.max_lines = 3000
tokenizer, result = OdysseyTokenizer.train(
    "datasets/raw/sample.txt",
    config=config,
    save_path="assets/tokenizer/bpe/odyssey.model",
)
```

## CLI

```bash
odyssey-tokenizer train --input datasets/raw/sample.txt --vocab-size 4096 --max-lines 3000
odyssey-tokenizer encode --model assets/tokenizer/bpe/odyssey.model --text "Hello"
odyssey-tokenizer decode --model assets/tokenizer/bpe/odyssey.model --ids 12,45,90
odyssey-tokenizer inspect --model assets/tokenizer/bpe/odyssey.model --text "Build authentication API" --show-merges
odyssey-tokenizer benchmark --model assets/tokenizer/bpe/odyssey.model --input datasets/raw/sample.txt --limit 200
odyssey-tokenizer visualize --model assets/tokenizer/bpe/odyssey.model --input datasets/raw/sample.txt
```

## Special tokens

| ID | Surface | Purpose |
| --- | --- | --- |
| 0 | `<pad>` | Padding |
| 1 | `<bos>` | Begin sequence |
| 2 | `<eos>` | End sequence |
| 3 | `<unk>` | Unknown (rare with byte-level BPE) |
| 4+ | `<mask>`, `<system>`, `<user>`, `<assistant>`, `<tool>`, `<think>` | Chat / tooling / reasoning |

Then IDs continue with the 256-byte alphabet and learned merges.

## Artifacts

A model directory contains:

- `vocab.json` — token surface → id
- `merges.txt` — ordered merge rules
- `config.json` — frozen hyperparameters
- `metadata.json` — training metrics

## SentencePiece reference

Phase 1 lives in `tokenizer/sentencepiece/` and
`configs/tokenizer_sentencepiece.yaml`. It remains available for comparison
(`ODY-0001`) but is **not** required to train Odyssey.
