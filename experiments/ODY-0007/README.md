# ODY-0007 — Grouped Query Attention

| Field | Value |
| --- | --- |
| Phase | 7 |
| Date | 2026-07-29 |
| Purpose | Implement causal GQA + cross-validate vs Phalanx |
| Result | **Successful** |

## Configuration

| Knob | Value |
| --- | --- |
| num_heads | 12 (Tiny) / 8 (validation default) |
| num_kv_heads | 4 (Tiny) / 2 (validation default) |
| head_dim | 64 (Tiny) / 8 (validation default) |
| causal | true |
| bias | false |
| RoPE | applied on Q/K (Spec) |

## Cross-Implementation Validation

```bash
python scripts/validate_attention.py
# or: python ../validation/test_attention.py
```

| Metric | Value |
| --- | --- |
| Max abs error | ≈ 1.30e-04 (with RoPE) |
| Mean abs error | ≈ 2.90e-06 |
| Tolerance | 1e-3 (GEMM accum; documented) |
| Status | **PASS** |

## Artifacts

- Metrics: `metrics.json`
- Validation: `attention_validation.json`
- Config snapshot: `config.yaml`
- Heatmaps: `../../assets/attention/`
