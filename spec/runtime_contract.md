# Runtime Contract

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

State mutual guarantees between Odyssey (producer) and Phalanx Runtime (consumer).

---

## Theory

The runtime is the reference inference engine for Odyssey — not an independent model zoo. Compatibility is contractual, versioned, and metadata-driven.

---

## What Odyssey Guarantees

1. Publishes Spec version with every released model.
2. Uses frozen weight names ([weight_layout.md](weight_layout.md)).
3. Uses frozen residual order and component math.
4. Ships tokenizer artifacts that honor [tokenizer.md](tokenizer.md).
5. Exports GGUF per [gguf_mapping.md](gguf_mapping.md) for serve builds.
6. Does not require the runtime to guess `num_layers`, head counts, or RoPE parameters.

---

## What Phalanx Guarantees

1. Declares supported Spec versions explicitly.
2. Loads architecture **only** from metadata / GGUF keys.
3. Binds tensors using the GGUF mapping table.
4. Implements layer math matching Spec (within documented numeric tolerance).
5. Surfaces typed errors for missing weights, shape mismatches, OOV token ids, RoPE OOB positions.
6. Updates `docs/spec-compliance.md` when layer coverage changes.

---

## Compatibility Rules

| Rule | Enforcement |
| --- | --- |
| Spec major mismatch | Hard refuse, or explicit compat mode named in release notes |
| Unknown required tensor | `MissingWeight` |
| Shape ≠ config | `ConfigMismatch` / `InvalidWeightShape` |
| Tokenizer vocab ≠ embd rows | Reject at load |
| Partial implementation | Allowed during bring-up; must not claim full Spec compliance |

---

## Versioning

```text
Odyssey publishes:  odyssey.spec.version
Phalanx declares:   supported_spec_versions = ["1.0.0"]
```

See [compatibility.md](compatibility.md).

---

## Required Metadata at Load

Minimum to construct the forward graph:

- `block_count` / `num_layers`
- `context_length`
- `embedding_length` / `hidden_size`
- `feed_forward_length` / `intermediate_size`
- `attention.head_count`, `head_count_kv`, head sizes
- `rope.*`, `rms_norm_eps`
- Embedding table (+ output if untied)
- Tokenizer model

---

## Error Handling

| Condition | Behavior |
| --- | --- |
| Missing Spec KV on Odyssey export | Warn; still load if Llama keys complete |
| Missing required Llama key | Error |
| Token id ≥ vocab | Error |
| Position ≥ context | Error |
| Unsupported RoPE scaling | Error |

No silent clamps that change model semantics.

---

## Future Extensions

- Capability negotiation API (`supports("kv_cache")`)
- Multi-arch beyond Llama-layout GGUF

---

## Compatibility Notes

CUDA/Metal backends are execution details. They must obey this same contract; acceleration is not a license to alter math.
