# GLU Variants Improve Transformer

**Paper:** Noam Shazeer, 2020 — https://arxiv.org/abs/2002.05202

---

## Motivation

Gated Linear Units (GLU) multiply one linear projection by a nonlinear transform of another, improving Transformer quality vs plain ReLU/GeLU FFNs.

## Variants

| Name | Gate nonlinearity |
| --- | --- |
| GLU | sigmoid |
| ReGLU | ReLU |
| GEGLU | GELU |
| **SwiGLU** | SiLU / Swish |

## Why SwiGLU

Empirically strongest among the GLU family in Shazeer's ablations; adopted by LLaMA / PaLM-style models. Odyssey Spec freezes `activation=swiglu`.

## Notes for Odyssey

Implement `SiLU(x)=x·σ(x)` explicitly and keep `w1/w3/w2` shapes identical to Phalanx `layers::SwiGlu`. Validated by `scripts/validate_swiglu.py`.
