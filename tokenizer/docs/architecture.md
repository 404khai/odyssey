# Tokenizer Architecture (Library)

```mermaid
flowchart TD
    RawCorpus --> Normalize
    Normalize --> SplitIntoBytes
    SplitIntoBytes --> BuildInitialVocabulary
    BuildInitialVocabulary --> CountPairFrequencies
    CountPairFrequencies --> SelectMostFrequentPair
    SelectMostFrequentPair --> MergePair
    MergePair --> UpdateVocabulary
    UpdateVocabulary -->|repeat| CountPairFrequencies
    UpdateVocabulary --> ExportVocabulary
    ExportVocabulary --> Encoder
    Encoder --> TokenIDs
```

## Package boundary

```
odyssey_tokenizer.OdysseyTokenizer
        ↑
 training / evaluation / future Phalanx Runtime
```

SentencePiece remains under `tokenizer/sentencepiece/` as a reference implementation only.
