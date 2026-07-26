# SentencePiece: A simple and language independent subword tokenizer

**Authors:** Taku Kudo, John Richardson  
**Link:** [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)

---

## Problem

Neural NLP systems need a stable mapping from raw Unicode text to a fixed vocabulary of integer IDs. Traditional tokenizers often assume whitespace word boundaries, which fails for languages without spaces (Japanese, Chinese, Thai) and forces brittle language-specific pre-tokenization.

---

## Motivation

Prior subword methods (BPE, WordPiece) usually run on pre-tokenized words. That couples the subword model to a language-specific word segmenter. SentencePiece aims for an end-to-end, language-independent pipeline that trains and applies subword segmentation directly on raw sentences.

---

## Algorithm

SentencePiece treats the input as a raw Unicode sequence and:

1. Optionally normalizes Unicode (NFKC by default in many setups).
2. Replaces whitespace with a visible meta-symbol (`▁`, U+2581) so spaces become ordinary characters in the alphabet.
3. Trains a subword model with either:
   - **BPE** — iterative merge of frequent adjacent pairs, or
   - **Unigram LM** — start from a large candidate vocabulary and prune by likelihood loss.
4. Encodes new text by finding a segmentation consistent with the trained model (greedy / Viterbi for unigram).

Because whitespace is an explicit symbol, detokenization is largely reversible: join pieces and map `▁` back to spaces.

---

## Advantages

- Language-independent; no external word tokenizer required
- Strong multilingual and CJK support
- Clean train → serialize → load workflow (`.model` + `.vocab`)
- Used in production stacks (T5, LLaMA-family SentencePiece tokenizers)
- Supports both BPE and Unigram under one API

---

## Disadvantages

- Vocabulary quality depends heavily on corpus composition
- Unigram training is heavier than pure BPE merges
- Special-token ID layouts must be configured carefully (pad/bos/eos/unk)
- Not identical to GPT-style byte-level BPE; transfer between ecosystems needs care

---

## Implementation Notes

- Odyssey uses the official `sentencepiece` Python package as a **reference** tokenizer in Phase 1.
- Reserved control tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`) are assigned fixed IDs via trainer flags.
- Chat/role tokens (`<system>`, `<user>`, `<assistant>`, `<mask>`) are registered as `user_defined_symbols`.
- Normalization is applied in Odyssey before calling SentencePiece so behavior is explicit and testable.

---

## Lessons Learned

- Tokenization is part of the model contract: changing the tokenizer later invalidates embeddings.
- Treating whitespace as a first-class symbol simplifies round-trip encode → decode.
- Configuration must own vocab size, coverage, model type, and special IDs — never hardcode them in call sites.

---

## How Odyssey Will Use This Knowledge

Phase 1 adopts SentencePiece as the reference pipeline for training, encoding, decoding, inspection, and metrics. Phase 2 will re-implement BPE from first principles, using SentencePiece outputs as a behavioral and quality baseline.
