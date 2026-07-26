# Changelog

All notable changes to Odyssey are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/).

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
