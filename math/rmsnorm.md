# RMSNorm — Mathematical Note

**Status:** Written + cross-validated (ODY-0005 / Phalanx Phase 9)

## Equations

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \varepsilon}
\]

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

No mean centering (unlike LayerNorm). No bias term in Spec v1.

## Complexity

| Resource | Cost |
| --- | --- |
| Time | \(O(B S D)\) |
| Parameters | \(O(D)\) for \(\gamma\) |
| vs LayerNorm | one fewer reduction + no \(\beta\) |

## Numerical Stability

- Keep \(\varepsilon > 0\) (default \(10^{-6}\))
- Accumulate \(\sum x_i^2\) in **float64**, then cast RMS back to float32 — keeps
  Odyssey and Phalanx bit-aligned on large `hidden_size` reductions (Principle 8)
- Activations may still be fp16/bf16; the RMS reduction promotes as needed

## PyTorch vs Phalanx

| Side | Module |
| --- | --- |
| Odyssey | `model.rmsnorm.OdysseyRMSNorm` |
| Phalanx | `phalanx::layers::RmsNorm` |

Parity gate: `scripts/validate_rmsnorm.py` (tol `1e-6` float32) — **PASS**, max error `0`.
