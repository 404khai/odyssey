# Experiment Log

Canonical index of Odyssey experiments. Detailed notes live under `experiments/ODY-XXXX/`.

Naming and logging conventions: [experiments/README.md](experiments/README.md).

---

## ODY-0000 — Repository initialization

| Field | Value |
| --- | --- |
| ID | ODY-0000 |
| Date | 2026-07-26 |
| Phase | 0 |
| Purpose | Establish research repository infrastructure |
| Config | `configs/default.yaml` |
| Result | **Successful** |
| Metrics | N/A (no training) |
| Lessons | Research infrastructure is now established. Maintainability starts before the first model layer. |

Details: [experiments/ODY-0000/README.md](experiments/ODY-0000/README.md)

---

## ODY-0001 — Baseline SentencePiece tokenizer

| Field | Value |
| --- | --- |
| ID | ODY-0001 |
| Date | 2026-07-27 |
| Phase | 1 |
| Purpose | Baseline SentencePiece tokenizer |
| Config | `configs/tokenizer.yaml` (snapshot in experiment folder) |
| Dataset | TinyStories sample (50,000 stories) |
| Model | Unigram |
| Requested vocab | 32000 |
| Actual vocab | 17047 |
| Result | **Successful** |
| Metrics | Training ~155s; compression ~4.29 chars/token; unk rate ~0.0022 |
| Lessons | Corpus diversity bounds achievable vocabulary size; soft vocab limits are required for homogeneous research samples. |

Details: [experiments/ODY-0001/README.md](experiments/ODY-0001/README.md)

---

## ODY-0002 — Odyssey BPE tokenizer

| Field | Value |
| --- | --- |
| ID | ODY-0002 |
| Date | 2026-07-28 |
| Phase | 2 |
| Purpose | Implement Odyssey byte-level BPE from first principles |
| Config | `configs/tokenizer.yaml` (experiment used vocab=2048, max_lines=1000) |
| Dataset | TinyStories sample |
| Algorithm | Byte-level BPE |
| Actual vocab | 2048 |
| Result | **Successful** |
| Metrics | ~4.16 chars/token; unk=0; train ~157s; library API stable |
| Lessons | Owning merges + vocab artifacts is the right boundary for Phalanx; pure-Python encode is the next optimization target (Rust). |

Details: [experiments/ODY-0002/README.md](experiments/ODY-0002/README.md)

---

## ODY-0003 — Token embedding layer

| Field | Value |
| --- | --- |
| ID | ODY-0003 |
| Date | 2026-07-28 |
| Phase | 3 |
| Purpose | Implement configurable token embedding layer |
| Config | `configs/embedding.yaml` (snapshot in experiment folder) |
| Vocabulary | 32000 |
| Hidden size | 768 |
| Initialization | Xavier uniform |
| Result | **Successful** |
| Metrics | See experiment `metrics.json` (lookup speed, memory, param count) |
| Lessons | Embedding is a gather not a matmul; `math/` notes should ship with each neural phase; weight tying deferred |

Details: [experiments/ODY-0003/README.md](experiments/ODY-0003/README.md)

---

## ODY-0004 — Rotary Positional Embeddings

| Field | Value |
| --- | --- |
| ID | ODY-0004 |
| Date | 2026-07-29 |
| Phase | 4 |
| Purpose | LLaMA-style RoPE + Phalanx numerical parity |
| Config | `configs/model.yaml` / experiment `config.yaml` |
| theta / rotary_dim | 10000 / 128 |
| Result | **Successful** |
| Validation | Max abs error ≈ 4.8e-7 vs Phalanx (`PASS` @ 1e-6) |
| Lessons | Keep inv_freq + adjacent-pair rotate bit-identical to Runtime; validate every layer |

Details: [experiments/ODY-0004/README.md](experiments/ODY-0004/README.md)

---

## ODY-0005 — RMSNorm & Residual Pathways

| Field | Value |
| --- | --- |
| ID | ODY-0005 |
| Date | 2026-07-29 |
| Phase | 5 |
| Purpose | LLaMA-style RMSNorm + pre-norm residuals + Phalanx numerical parity |
| Config | `configs/model.yaml` / experiment `config.yaml` |
| epsilon | 1e-6 |
| Result | **Successful** |
| Validation | See `rmsnorm_validation.json` (`PASS` @ 1e-6) |
| Lessons | Float32 sum-of-squares + identical ε/γ keep train/serve aligned; shared suite under `../validation/` |

Details: [experiments/ODY-0005/README.md](experiments/ODY-0005/README.md)


---

## ODY-0006 — SwiGLU Feed-Forward

| Field | Value |
| --- | --- |
| ID | ODY-0006 |
| Date | 2026-07-29 |
| Phase | 6 |
| Purpose | LLaMA-style SwiGLU + Phalanx numerical parity |
| Config | `configs/model.yaml` / experiment `config.yaml` |
| Result | **Successful** |
| Validation | See `swiglu_validation.json` (PASS @ 1e-3 GEMM tol) |
| Lessons | Float64 GEMM accum + documented SwiGLU abs tol keep train/serve aligned |

Details: [experiments/ODY-0006/README.md](experiments/ODY-0006/README.md)


---

## ODY-0007 — Grouped Query Attention

| Field | Value |
| --- | --- |
| ID | ODY-0007 |
| Date | 2026-07-29 |
| Phase | 7 |
| Purpose | Causal GQA + Phalanx parity |
| Config | `configs/model.yaml` `attention:` (H=12, H_kv=4, d=64) |
| Result | **Successful** |
| Metrics | Max abs error ≈ 1.30e-04 (tol 1e-3); Tiny params 1,572,864 |
| Lessons | GQA broadcast + float64 GEMM parity with Phalanx; RoPE-on-Q/K must share layout conventions |

Details: [experiments/ODY-0007/README.md](experiments/ODY-0007/README.md)
