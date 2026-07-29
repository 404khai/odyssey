# Odyssey Math Notes

Mathematical companions to each model implementation phase.

Each note covers:

1. Equations
2. Why the algorithm works
3. Time complexity
4. Memory complexity
5. Numerical stability
6. How PyTorch implements it (training path)
7. How Phalanx Runtime executes the same computation at inference

| Note | Phase | Status |
| --- | --- | --- |
| [linear_algebra.md](linear_algebra.md) | foundation | Written |
| [tensor_shapes.md](tensor_shapes.md) | foundation | Written |
| [embeddings.md](embeddings.md) | 3 | Written |
| [rope.md](rope.md) | 4 | Written + cross-validated |
| [rmsnorm.md](rmsnorm.md) | 5 | Outline (pre-implementation) |
| [attention.md](attention.md) | 6 | Outline (pre-implementation) |
| [swiglu.md](swiglu.md) | 7 | Outline (pre-implementation) |
| [loss.md](loss.md) | 10 | Outline (pre-implementation) |

Code lives under `model/`. Architecture prose lives under `docs/architecture/`.

**Normative contract:** [`../spec/`](../spec/README.md) (Odyssey Specification v1).  
If `math/` and `spec/` disagree, **`spec/` wins**. These notes are pedagogical companions.
