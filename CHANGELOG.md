# Changelog

All notable changes to Odyssey are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-07-28

### Added

- Owned byte-level BPE library (`tokenizer/odyssey_tokenizer`)
- Clean API: `OdysseyTokenizer.load/encode/decode`
- CLI entrypoint `odyssey-tokenizer` (train/encode/decode/inspect/benchmark/visualize)
- Merge visualizer + compression plots
- Benchmark suite with memory tracing
- Library docs under `tokenizer/docs/`
- Experiment `ODY-0002`

### Changed

- Tokenizer packaged as a reusable library for Phalanx Runtime consistency
- SentencePiece config moved to `configs/tokenizer_sentencepiece.yaml` (reference)
- Primary config `configs/tokenizer.yaml` now targets BPE

### Notes

- ODY-0002 trained a **2048**-piece model on 1000 TinyStories lines for a practical Python baseline
- Config default `vocab_size: 32000` remains available via CLI for larger runs

---

## [0.1.0] — 2026-07-27

### Added

- SentencePiece reference tokenizer package (`tokenizer/sentencepiece/`)
- Tokenizer config (`configs/tokenizer.yaml`) with reserved special tokens
- Training CLI (`scripts/train.py`) and inspector (`scripts/inspect_tokenizer.py`)
- TinyStories corpus preparer (`scripts/prepare_tinystories_sample.py`)
- Encode / decode / save / load / stats / inspect APIs
- Unit tests for training, encoding, decoding, and special tokens
- Paper summaries: SentencePiece, BPE, GPT-2 tokenizer, tiktoken
- Tokenizer architecture docs and experiment `ODY-0001`

### Notes

- Actual Unigram vocab on 50k TinyStories stories settled at **17047** (corpus-limited vs requested 32000)
- Custom Odyssey BPE begins in Phase 2

---

## [0.0.0] — 2026-07-26

### Added

- Research repository structure (`configs/`, `datasets/`, `docs/`, `experiments/`, packages)
- Python 3.12 project packaging via `pyproject.toml` and `requirements.txt`
- Default experiment config at `configs/default.yaml`
- Dev tooling: Black, Ruff, isort, mypy, pytest
- Documentation: README, ROADMAP, MODEL_CARD, PAPERS, EXPERIMENTS, RESEARCH
- Experiment tracking conventions and log entry `ODY-0000`
- Phase 0 smoke tests (imports, config load, structure checks)

### Notes

- No machine learning implementation in this release
- Next: Phase 1 — Tokenizer Research
