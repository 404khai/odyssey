# Neural Machine Translation of Rare Words with Subword Units (BPE)

**Authors:** Rico Sennrich, Barry Haddow, Alexandra Birch  
**Link:** [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

---

## Problem

Neural machine translation vocabularies cannot cover every word form. Rare and unseen words collapse to a single `<unk>`, destroying translation quality for names, compounds, and morphologically rich languages.

---

## Motivation

Character-level models generalize but produce very long sequences. Word-level models are compact but brittle on the long tail. Subword units sit in between: frequent words stay atomic; rare words decompose into reusable pieces.

---

## Algorithm

Byte Pair Encoding adapted from compression:

1. Initialize the vocabulary with characters (plus an end-of-word marker in classic BPE).
2. Count adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the target vocabulary size is reached.
5. At inference, apply the learned merge table greedily to new text.

Example intuition: `low` + `er` → `lower` once that pair is frequent enough.

---

## Advantages

- Open-vocabulary behavior without a huge word list
- Improves rare-word handling in NMT and LMs
- Deterministic merges once the table is fixed
- Conceptually simple; easy to reimplement for learning

---

## Disadvantages

- Greedy merges are not globally optimal segmentations
- Classic BPE often assumes pre-tokenization / word boundaries
- Merge order can be sensitive to corpus domain (code vs prose)
- Does not directly model a probabilistic segmentation (unlike Unigram LM)

---

## Implementation Notes

- GPT-2 later popularized **byte-level** BPE (alphabet = 256 bytes), avoiding `<unk>` for Unicode.
- SentencePiece can train a BPE model without whitespace pre-tokenization.
- Odyssey Phase 2 will implement custom BPE; Phase 1 uses SentencePiece Unigram/BPE as reference.

---

## Lessons Learned

- Vocabulary size is a compression ↔ generalization tradeoff, not a magic constant.
- Domain mismatch between tokenizer corpus and model corpus wastes context window.
- Merge rules are an artifact worth inspecting — they reveal what the model “thinks” is atomic.

---

## How Odyssey Will Use This Knowledge

BPE is the conceptual foundation for Odyssey’s Phase 2 custom tokenizer. We will own merge training, vocabulary export, and visualization, comparing compression and unknown rates against the Phase 1 SentencePiece baseline.
