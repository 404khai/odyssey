# Changelog

All notable changes to Odyssey are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- Odyssey Specification **v1.0.0** under `spec/` (architecture, shapes, weights, GGUF mapping, tokenizer, runtime contract)
- Cross-links from README / math / docs to the normative Spec

---

## [0.5.0] — 2026-07-29

### Added

- LLaMA-style RMSNorm (`model.OdysseyRMSNorm`) with float32 RMS accumulation
- Pre-norm residual helpers (`model.residual`)
- Shared normalization math (`model.normalization`)
- `NormConfig` + `norm:` section in `configs/model.yaml` / `configs/default.yaml`
- Cross-implementation validator `scripts/validate_rmsnorm.py` (vs Phalanx)
- Benchmarks `scripts/benchmark_rmsnorm.py` + residual flow assets
- Papers: RMSNorm, residual connections, LLaMA normalization
- Math notes: `math/rmsnorm.md`, `math/residuals.md`
- Experiment `ODY-0005`
- Shared monorepo suite entry: `../validation/test_rmsnorm.py`

---

## [0.4.0] — 2026-07-29

### Added

- LLaMA-style RoPE (`model.OdysseyRoPE`, cache, math, visualizer)
- `configs/model.yaml` with rope hyperparameters
- Cross-implementation validator `scripts/validate_rope.py` (vs Phalanx)
- Benchmarks + assets under `assets/rope/`
- Papers: RoFormer, attention positional encodings, LLaMA RoPE
- Experiment `ODY-0004` (validation PASS, max abs error ~4.8e-7)

---

## [0.3.0] — 2026-07-28

### Added

- Configurable token embedding layer (`model.OdysseyEmbedding`)
- Embedding config (`configs/embedding.yaml`, `model.config.EmbeddingConfig`)
- Weight initialization: Normal, Xavier, Kaiming (`model.initialization`)
- Embedding inspector + visualizer (`model.embedding_visualizer`)
- Shape / init / embedding unit tests + tokenizer→embedding integration
- Benchmark script `scripts/benchmark_embeddings.py`
- Architecture doc `docs/architecture/embeddings.md`
- Math notes directory (`math/`) with embeddings + foundation notes and outlines for later phases
- Paper summaries: Transformer embeddings, Word2Vec, GloVe
- Experiment `ODY-0003`

### Notes

- Weight tying documented but not implemented (future phase)
- Ready for Phase 4 RoPE — do not start without approval

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
