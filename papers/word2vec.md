# Word2Vec

**Authors:** Tomas Mikolov et al.  
**Links:** [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781), [Distributed Representations…](https://arxiv.org/abs/1310.3781)

---

## Problem

One-hot / sparse word features do not capture similarity. Rare words and analogies are hard for classical NLP features.

---

## Motivation

Learn dense vectors such that linear relationships approximate linguistic regularities (e.g. king − man + woman ≈ queen).

---

## Algorithm

Two training styles:

- **CBOW** — predict a word from its context bag
- **Skip-gram** — predict context words from a center word

Negative sampling approximates the softmax over the full vocabulary.

The result is an embedding matrix conceptually identical to modern LM input embeddings, trained with a shallow objective.

---

## Advantages

- Cheap to train relative to full LMs
- Produces interpretable nearest neighbors
- Established the “embedding space” intuition used everywhere since

---

## Disadvantages

- Static embeddings (one vector per word type, no context)
- Weak on polysemy and sentence-level meaning
- Superseded as end models by contextual Transformers — but still pedagogically essential

---

## Implementation Notes

Odyssey does not train Word2Vec; Transformer LM training will move embedding rows via backprop. The nearest-neighbor inspector in `embedding_visualizer.py` is the same geometric probe Word2Vec popularized.

---

## Lessons Learned

- Geometry of rows in \(E\) is meaningful after learning.
- Initialization + objective jointly determine early geometry.
- Contextual models still *start* from the same lookup-table idea.

---

## How Odyssey Will Use This Knowledge

Use Word2Vec as conceptual grounding: embeddings are learned coordinates. Inspect norms and neighbors during Phase 3 even before language-model training begins.
