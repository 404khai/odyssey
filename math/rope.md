# RoPE — Mathematical Note (Phase 4 outline)

> Implementation lands in Phase 4. Equations recorded here so Phase 3 readers see what comes next.

## Equations (preview)

For head dimension \(d\) (even), position \(m\), and pair index \(i\):

\[
\theta_i = 10000^{-2i/d}
\]

\[
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix}
=
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\]

Applied to query and key vectors; values typically untouched.

## Why it works

Relative position is encoded in the **relative rotation** between positions \(m\) and \(n\), improving length extrapolation vs absolute learned position embeddings.

## Complexity (preview)

- Time: \(O(B S D)\) to apply rotations
- Memory: \(O(S \cdot d/2)\) for cached \(\cos/\sin\) tables (or on-the-fly)

## Numerical stability

Use float32 for angle tables when possible; watch bf16 precision at long contexts.

## PyTorch vs Phalanx

| Path | Status |
| --- | --- |
| Odyssey `model/` RoPE module | Phase 4 |
| Phalanx `layers::rope` | Already prototyping in runtime |

## References

- Su et al., *RoFormer* (RoPE)
