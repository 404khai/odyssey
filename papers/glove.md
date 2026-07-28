# GloVe: Global Vectors for Word Representation

**Authors:** Jeffrey Pennington, Richard Socher, Christopher Manning  
**Link:** [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)

---

## Problem

Count-based co-occurrence methods and local-context predictive methods (Word2Vec) were developed separately; neither fully used global corpus statistics in a clean bilinear model.

---

## Motivation

Factorize a global word–word co-occurrence matrix with a weighted least-squares objective so that dot products encode log co-occurrence ratios.

---

## Algorithm (intuition)

Learn vectors \(w_i, \tilde{w}_j\) such that:

\[
w_i^\top \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}
\]

with a weighting function that down-weights very rare and very frequent pairs.

---

## Advantages

- Leverages **global** statistics, not only local windows
- Strong static baselines on analogy / similarity suites of its era
- Clarifies that embedding spaces encode co-occurrence structure

---

## Disadvantages

- Still non-contextual
- Requires building large co-occurrence matrices
- Not the training path for Odyssey (we train end-to-end LM loss)

---

## Implementation Notes

No GloVe trainer in-repo. Useful for explaining *why* nearby embedding rows should correlate with distributional similarity after LM training.

---

## Lessons Learned

- Embedding quality can be viewed as matrix factorization of statistics.
- Modern LMs implicitly learn a richer, context-dependent version of the same idea.

---

## How Odyssey Will Use This Knowledge

When documenting embedding spaces and future probing tools (PCA in a later iteration), GloVe provides the co-occurrence / geometry vocabulary.
