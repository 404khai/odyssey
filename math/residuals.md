# Residual Connections — Mathematical Note

**Status:** Written (Phase 5)

## Identity Skip

\[
y = x + f(x)
\qquad
\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}
\]

The identity term prevents gradient vanishing when \(\partial f/\partial x\) is small.

## Pre-Norm (Odyssey Spec v1)

\[
y = x + f(\mathrm{RMSNorm}(x))
\]

Frozen. Post-norm \(y = \mathrm{Norm}(x + f(x))\) is out of scope.

## Shape Invariant

Residuals require identical shapes for \(x\) and \(f(\cdot)\). Odyssey `residual_add` rejects broadcasting to catch wiring bugs early.
