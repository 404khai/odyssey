# SwiGLU Feed-Forward Network

## Motivation

Attention mixes tokens; the FFN lets each position apply a nonlinear transform independently, adding most of the model's capacity. LLaMA-style models replace GeLU MLPs with **SwiGLU**.

## Mathematics

See Spec [`spec/feedforward.md`](../../spec/feedforward.md).

\[
\mathrm{SiLU}(z)=z\cdot\sigma(z)
\qquad
\mathrm{FFN}(x)=\bigl(\mathrm{SiLU}(x W_1^\top)\odot(x W_3^\top)\bigr)W_2^\top
\]

| Weight | Role | Shape |
| --- | --- | --- |
| \(W_1\) (`gate_proj`) | Gate | `(I, D)` |
| \(W_3\) (`up_proj`) | Up | `(I, D)` |
| \(W_2\) (`down_proj`) | Down | `(D, I)` |

No biases. Intermediate size \(I\) is independent of \(D\) (Tiny: 2048 vs 768) — larger than the classic \(4D\) GeLU FFN after the SwiGLU \(2/3\) parameter adjustment.

## Implementation

| Module | Role |
| --- | --- |
| `model/activations.py` | Manual SiLU |
| `model/swiglu.py` | `OdysseySwiGLU` |
| `model/feedforward.py` | Public FFN factory |
| `configs/model.yaml` | `feed_forward:` |

```python
from model import OdysseySwiGLU, load_feed_forward_config
ffn = OdysseySwiGLU(load_feed_forward_config())
y = ffn(x)  # (B,S,D) → (B,S,D)
```

## Phalanx Compatibility

```bash
python scripts/validate_swiglu.py
# or: python ../validation/test_swiglu.py
```

Tolerance default **`1e-3`** (GEMM accumulation; mean error typically ≪ `1e-6`).
Report: `experiments/ODY-0006/swiglu_validation.json`.
