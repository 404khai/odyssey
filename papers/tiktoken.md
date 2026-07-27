# TikToken

**Study target:** OpenAI `tiktoken` library  
**Link:** [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)

---

## Problem

Large-scale inference and training need tokenization that is not only correct but extremely fast. Pure-Python BPE over long prompts becomes a bottleneck when serving chat and batch workloads.

---

## Motivation

OpenAI open-sourced `tiktoken` as a fast BPE implementation (Rust core, Python bindings) that matches production encodings used by models such as `cl100k_base` (ChatGPT / GPT-4 era) and earlier GPT-2/GPT-3 encodings.

---

## Algorithm

Conceptually still BPE:

1. Load a named encoding (merge ranks + special token map).
2. Split / scan text according to the encoding’s regex pre-tokenizer.
3. Apply byte-level BPE using merge ranks (lower rank = earlier merge).
4. Return token IDs; decode reverses the process.

Engineering focus:

- Efficient core implementation
- Stable public encoding names
- First-class special token APIs for chat formats

---

## Advantages

- Very fast relative to naive Python tokenizers
- Exact compatibility with OpenAI public encodings
- Simple `encode` / `decode` surface for applications
- Useful reference for production tokenizer performance targets

---

## Disadvantages

- Not a training framework — vocabularies are precomputed artifacts
- Regex pre-tokenization is encoding-specific and opaque at first glance
- Tied to OpenAI encoding families; not a drop-in for SentencePiece models
- Less educational for learning how merges are *learned* (only how they are *applied*)

---

## Implementation Notes

- Odyssey Phase 1 trains with SentencePiece; we do not depend on `tiktoken` at runtime.
- Benchmarks in this phase record encode/decode speed so Phase 2 BPE can set performance goals informed by `tiktoken`-class tooling.
- Special-token handling in Odyssey chat templates should eventually be as explicit as `tiktoken`’s special token maps.

---

## Lessons Learned

- Training a tokenizer and serving a tokenizer are different products.
- Performance belongs in the tokenizer contract for real systems.
- Named, versioned encodings prevent silent train/serve vocabulary drift.

---

## How Odyssey Will Use This Knowledge

Use `tiktoken` as a speed and API-design reference. Odyssey’s custom Phase 2 BPE should prioritize clarity first, then approach production throughput. Phalanx Runtime may later host a Rust tokenizer path inspired by this split between research Python and serving-optimized cores.
