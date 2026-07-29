# Causal Masking

Decoder-only LMs must not attend to future tokens.

Additive mask \(M\) with shape `(S, S)` (prefill) or `(1, T)` (decode):

- `M[s,t] = 0` if `t ≤ s` (allowed)
- `M[s,t] = -∞` otherwise (blocked; Softmax → 0)

Implemented in `model/causal_mask.py`. Visualization:
`assets/attention/causal_mask.png`.
