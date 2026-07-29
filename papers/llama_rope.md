# LLaMA RoPE Notes

**References:** LLaMA / Llama 2 technical reports; llama.cpp / GGUF RoPE metadata

---

## Formulation Used by Odyssey & Phalanx

- Adjacent-pair rotation (interleaved even/odd dims)
- \(\theta_i = \texttt{rope.freq\_base}^{-2i/d_r}\) (default base 10000)
- Partial rotary dims: rotate first `rope.dimension_count` features
- Optional **linear** position scaling: \(m' = m / \mathrm{factor}\)
- Apply to **Q and K only**

---

## Differences from RoFormer Presentation

| Topic | RoFormer paper | LLaMA / Odyssey |
| --- | --- | --- |
| Complex vs real | Emphasizes complex view | Real 2×2 rotate (equivalent) |
| Pairing | Complex channel pairs | Adjacent dims `(0,1), (2,3), …` |
| Partial RoPE | Not always stressed | First `rotary_dim` only |
| Scaling | Later literature | Linear in Spec v1; YaRN/NTK later |

---

## Phalanx Parity Contract

Must match:

- θ / inverse-frequency generation
- Adjacent-pair rotate
- Cos/sin cache indexing `pos * n_pairs + pair`
- Linear scaling
- Tensor layouts `[seq, heads, head_dim]` (Odyssey also accepts batched rank-4)

Validated by `scripts/validate_rope.py` + `runtime` bin `validate_rope`.
