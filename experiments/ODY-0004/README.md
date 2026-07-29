# ODY-0004 — Rotary Positional Embeddings

| Field | Value |
| --- | --- |
| Phase | 4 |
| Date | 2026-07-29 |
| Purpose | Implement LLaMA-style RoPE + cross-validate vs Phalanx |
| Result | **Successful** |

## Configuration

| Knob | Value |
| --- | --- |
| theta | 10000 |
| rotary_dim | 128 (experiment) / 64 (Tiny default) |
| head_dim | 128 (experiment) / 64 (Tiny) |
| max_position_embeddings | 4096 (experiment) |
| scaling | none (linear supported) |

## Cross-Implementation Validation

```bash
python scripts/validate_rope.py --head-dim 128 --rotary-dim 128 --max-position 4096
```

See `rope_validation.json` for max/mean absolute error vs Phalanx `layers::Rope`.

| Metric | Value |
| --- | --- |
| Max abs error | ≈ 4.77e-7 |
| Mean abs error | ≈ 5.62e-9 |
| Tolerance | 1e-6 |
| Status | **PASS** |

## Artifacts

- Metrics: `metrics.json`
- Validation: `rope_validation.json`
- Plots: `assets/rope/`
- Config snapshot: `config.yaml`
