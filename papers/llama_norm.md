# LLaMA Normalization Notes

**References:** LLaMA / Llama 2 technical reports; Odyssey Spec `rmsnorm.md`

---

## Pre-Norm Ordering

Each decoder block:

1. `attention_norm` (RMSNorm) → attention (+ RoPE on Q/K) → residual
2. `ffn_norm` (RMSNorm) → SwiGLU FFN → residual
3. Final `norm` (RMSNorm) before the LM head

## Odyssey / Phalanx Contract

| Knob | Value |
| --- | --- |
| Formula | \(\gamma \odot x / \mathrm{RMS}(x)\) |
| ε | `rms_norm_eps` / `norm.epsilon` (default `1e-6`) |
| Bias | none |
| γ init (train) | ones |
| GGUF γ names | `blk.{i}.attn_norm.weight`, `blk.{i}.ffn_norm.weight`, `output_norm.weight` |

Validated by `scripts/validate_rmsnorm.py` + Phalanx `validate_rmsnorm`.
