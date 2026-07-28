# Merge Algorithm

## Greedy selection

At each step the trainer computes:

```
count(a, b) = how often symbol a is immediately followed by symbol b
```

Then selects:

```
argmax count(a, b)
```

Ties break on lexicographic `(a, b)` so runs are reproducible given the same corpus + seed/shuffle settings.

## Vocabulary evolution

```
specials (10)
+ bytes (256)
+ merge #0
+ merge #1
…
→ vocab_size
```

Example:

```
e
+
r
↓
er

er
+
s
↓
ers
```

## Frequency counting

Lines are normalized, UTF-8 encoded, and counted as unique symbol sequences with multiplicity. Identical lines share one entry — this is the classic BPE “word frequency dictionary” optimization.

## Encoding with merges

Given merges ranked `0..M-1`, encoding repeatedly replaces the adjacent pair whose merge rank is smallest until no mergeable pair remains.
