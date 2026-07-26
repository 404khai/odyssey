# GPT-2 Byte Pair Encoding Tokenizer

**Study target:** OpenAI GPT-2 tokenizer (byte-level BPE)  
**Primary references:** GPT-2 paper / public `gpt2` tokenizer artifacts in Hugging Face `transformers`

---

## Problem

Unicode text includes millions of code points. A character-level vocab is huge; a naive word vocab leaves many unknowns. GPT-2 needs a tokenizer that never fails on arbitrary UTF-8 while keeping sequences short enough for Transformer training.

---

## Motivation

Earlier BPE often operated on Unicode characters after whitespace splitting and still risked unknown tokens. GPT-2 instead tokenizes **UTF-8 bytes**, so any string is representable. Subword merges then recover efficient multi-byte / multi-character tokens for common patterns.

---

## Algorithm

High-level GPT-2 tokenization:

1. Encode text to UTF-8 bytes.
2. Map bytes into an internal unicode-ish alphabet used by the BPE machinery.
3. Apply a fixed list of BPE merges learned on WebText.
4. Emit integer IDs from a ~50k vocabulary (50,257 including special tokens in common releases).
5. Decoding inverts merges and byte mapping back to UTF-8 text.

Notable properties:

- No traditional `<unk>` for odd Unicode — bytes always cover the input.
- Leading spaces are part of tokens (similar in spirit to SentencePiece `▁`).
- Special tokens are minimal compared with chat-oriented models (`<|endoftext|>`).

---

## Advantages

- True open vocabulary over bytes
- Strong performance on English web text and code-ish snippets
- Widely available reference implementations (`tiktoken`, `transformers`)
- Stable public vocabulary enables interoperable tooling

---

## Disadvantages

- English / WebText-centric merges; weaker for some multilingual scripts
- Vocabulary is large (~50k), increasing embedding table cost
- Byte fallback can produce longer sequences for under-represented languages
- Chat/role scaffolding is not native to the original GPT-2 vocab

---

## Implementation Notes

- Odyssey does **not** ship a GPT-2 tokenizer; we study it as a design reference.
- Compare: GPT-2 (byte-level BPE) vs LLaMA (SentencePiece) vs Odyssey Phase 1 (SentencePiece Unigram reference).
- Inspectors should always show: raw text → pieces → IDs → decoded text.

---

## Lessons Learned

- “No unknowns” is achievable if the base alphabet is bytes (or a complete unicode strategy).
- Space-aware tokens matter for fidelity and for how models learn word boundaries.
- Tokenizer choice is an architectural decision with lasting effects on context efficiency.

---

## How Odyssey Will Use This Knowledge

When implementing Odyssey BPE in Phase 2, we will decide explicitly whether the alphabet is characters, bytes, or a hybrid. GPT-2’s byte-level approach is the default industry answer for robust open-vocabulary encoding; SentencePiece remains our Phase 1 reference for LLaMA-like pipelines.
