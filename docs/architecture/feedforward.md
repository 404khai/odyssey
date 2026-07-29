# Feed-Forward Network

Odyssey's position-wise MLP is **SwiGLU only** (Spec v1). See [swiglu.md](swiglu.md) for the full derivation.

```text
x → gate_proj → SiLU ─┐
                       ⊙ → down_proj → y
x → up_proj ──────────┘
```

Complexity: \(O(B S D I)\) time; peak activation memory \(O(B S I)\).
