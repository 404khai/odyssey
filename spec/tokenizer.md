# Tokenizer Contract

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Define a language-agnostic tokenizer interface and artifact rules so Python training and Rust inference encode/decode identically.

---

## Theory

Byte-level BPE maps UTF-8 text to a finite vocabulary without unknown Unicode holes. Special tokens reserve stable IDs for pad/bos/eos/roles.

---

## Public API (identical across implementations)

| Method | Behavior |
| --- | --- |
| `encode(text, add_bos=false, add_eos=false, normalize=true) → list[int]` | Text → token ids |
| `decode(ids, skip_special_ids=false) → string` | Ids → text |
| `save(path) → path` | Persist artifacts |
| `load(path) → tokenizer` | Restore artifacts |
| `inspect(text) → structured report` | Debug segmentation |

Optional but recommended: `encode_as_tokens`, vocabulary introspection, stats.

---

## Vocabulary & Special Tokens

Stable core IDs (Spec v1):

| ID | Token |
| --- | --- |
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<eos>` |
| 3 | `<unk>` |
| 4 | `<mask>` |
| 5 | `<system>` |
| 6 | `<user>` |
| 7 | `<assistant>` |
| 8 | `<tool>` |
| 9 | `<think>` |

Additional specials may append with IDs \(\ge 10\) but must be listed in metadata. Byte tokens and merges follow.

Default target `vocab_size = 32000` (actual trained size recorded in artifacts).

---

## Normalization (before encode when `normalize=true`)

| Rule | Default |
| --- | --- |
| Unicode | NFKC |
| Lowercase | Off |
| Strip ends | On |
| Collapse whitespace | On |
| Preserve newlines | On (collapse runs of newlines sensibly; non-newline WS → space) |

Normalization must be bit-identical across implementations for the same flags.

---

## Encoding Rules

1. Normalize (optional flag).
2. UTF-8 encode to bytes.
3. Map bytes / merges via learned BPE merge table (greedy by merge priority).
4. Optionally prepend `bos_id` / append `eos_id`.

Decoding reverses merges/bytes to UTF-8; invalid sequences must fail loudly or use a documented replacement policy — Spec v1 prefers **failure on corrupt id streams** in library mode.

---

## Artifact Format (`odyssey-bpe`, version `1`)

Directory:

| File | Content |
| --- | --- |
| `vocab.json` | token string → id |
| `merges.txt` | ordered merges (`#odyssey-bpe-merges v1 …` header; hex-encoded pieces) |
| `config.json` | tokenizer config |
| `metadata.json` | `format`, `version`, `vocab_size`, `merge_count`, … |

`MODEL_FORMAT_VERSION = 1`. Breaking artifact changes bump tokenizer version **and** require Spec compatibility notes.

---

## GGUF Embedding

Export must populate `tokenizer.ggml.tokens`, `.merges`, `.model`, special id fields so Phalanx can `Tokenizer::from_gguf`. Round-trip fidelity with `odyssey-bpe` artifacts is required before declaring serve parity.

---

## Mathematics

Compression ratio (diagnostic):

\[
\mathrm{chars\_per\_token} = \frac{\#\mathrm{chars}}{\#\mathrm{tokens}}
\]

Not a correctness criterion.

---

## Tensor Shapes

Tokenizer outputs rank-1 id sequences; batching is the caller’s responsibility → `(B, S)` after pad/pack.

---

## Implementation Notes

- Odyssey: `odyssey_tokenizer.OdysseyTokenizer`
- Phalanx today: GGUF tokenizer path; **native `odyssey-bpe` directory loader is a required compliance item** (see runtime compliance matrix)

---

## Examples

```text
encode("Hello") → […ids…]
decode(ids) → "Hello"
```

---

## Future Extensions

- Regex pretokenizer for code
- Rust port reading the same `merges.txt` / `vocab.json`

---

## Compatibility Notes

Train/serve token id mismatch is a **P0** contract break. Any change to merge application order requires a tokenizer format version bump.
