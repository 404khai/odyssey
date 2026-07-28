# Vocabulary

## Layout

| Range | Content |
| --- | --- |
| `0 .. S-1` | Special tokens (`<pad>`, `<bos>`, …) |
| `S .. S+255` | Raw bytes `0x00`–`0xFF` |
| `S+256 ..` | Learned merge tokens |

Surfaces are stored as latin-1 round-trippable strings in `vocab.json` so arbitrary byte sequences survive JSON.

## Serialization

```
odyssey.model/
  vocab.json
  merges.txt
  config.json
  metadata.json
```

## Determinism

Given the same:

- corpus bytes
- shuffle seed / max_lines
- `vocab_size` / `min_frequency`
- special token list

vocabulary IDs and merge ranks are stable across save → load → encode.
