# RoPE — Mathematical Note (Phase 4)

**Status:** Implemented in Odyssey + Phalanx (cross-validated)

## Equations

\[
\omega_i = \theta^{-2i/d_r}, \quad
\begin{pmatrix} x_0' \\ x_1' \end{pmatrix}
=
R(m'\omega_i)
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
\]

Linear scaling: \(m' = m / \mathrm{factor}\).

## Complexity

| Resource | Cost |
| --- | --- |
| Apply | \(O(BSH d)\) |
| Cache | \(O(S_{\max} d_r)\) |

## Numerical Stability

- Build angles in float32
- Default validation tolerance vs Phalanx: `1e-6`

## PyTorch

`OdysseyRoPE` / `rope_cache.RopeCacheManager`

## Phalanx Runtime

`layers::Rope` — identical adjacent-pair kernel; validated via `scripts/validate_rope.py`.
