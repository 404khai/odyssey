# Compatibility Policy

**Spec:** Odyssey Specification `1.0.0`  
**Normative:** Yes

---

## Purpose

Define how the Odyssey Specification evolves and how Phalanx tracks it.

---

## Specification Semver

| Bump | When |
| --- | --- |
| **Major** | Breaking: rename weights, change residual order, change default activation/norm, alter tokenizer merge semantics incompatibly |
| **Minor** | Additive: optional metadata, new optional scaling mode with explicit opt-in, new optional sampler |
| **Patch** | Clarifications, diagram fixes, non-semantic wording |

Current: **`1.0.0`**.

---

## Dual Evolution Rule

Every architectural decision in Odyssey requires:

1. Spec edit + version bump as appropriate
2. Phalanx compliance matrix update
3. Shared review that train and serve still agree

Every runtime optimization must preserve Odyssey numerical intent (bitwise identity not required; semantic equivalence is).

---

## Phalanx Support Declaration

Phalanx Runtime **v0.x** targets:

```text
supported_odyssey_spec = ["1.0.0"]
```

Layer completeness is orthogonal — see `runtime/docs/spec-compliance.md`.

---

## Deprecation

- Deprecated behaviors remain at least one minor release after announcement.
- Removed behaviors require major Spec bump.

---

## Testing Expectations (future harness)

- Embedding gather parity tests (Odyssey vs Phalanx) on fixed weights
- RoPE rotate parity on fixed Q/K
- Tokenizer encode/decode golden files
- Full logits parity once decoder lands

---

## Implementation Notes

Pedagogical notes in `math/` are **non-normative**. If `math/` and `spec/` disagree, **`spec/` wins**.

---

## Examples

Adding YaRN as optional `rope_scaling.type=yarn` → Spec `1.1.0` if old models remain valid.  
Switching default residual to post-norm → Spec `2.0.0`.

---

## Compatibility Notes

Third-party runtimes may implement Spec v1; Phalanx remains the reference.
