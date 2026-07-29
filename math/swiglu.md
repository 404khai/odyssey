# SwiGLU — Mathematical Note

**Status:** Written + cross-validated (ODY-0006 / Phalanx Phase 10)

## Equations

\[
\mathrm{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}
\]

\[
\mathrm{FFN}(x)=\bigl(\mathrm{SiLU}(x W_1^\top)\odot(x W_3^\top)\bigr)W_2^\top
\]

| Weight | Role | Shape |
| --- | --- | --- |
| \(W_1\) | Gate | `(I, D)` |
| \(W_3\) | Up | `(I, D)` |
| \(W_2\) | Down | `(D, I)` |

(Spec naming — not the outline's older \(W_2\)/`W_3` swap.)

## Complexity

Time \(O(B S D I)\); params \(3DI\) (no biases).

## PyTorch vs Phalanx

| Side | Module |
| --- | --- |
| Odyssey | `model.swiglu.OdysseySwiGLU` |
| Phalanx | `phalanx::layers::SwiGlu` |

Parity: `scripts/validate_swiglu.py` (default abs tol `1e-3` — GEMM float
accumulation order differs between Phalanx's reference ijk kernel and PyTorch;
mean error is typically ≪ `1e-6`).
