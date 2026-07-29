# LLaMA Feed-Forward Notes

**References:** LLaMA / Llama 2 reports; Odyssey Spec `feedforward.md`

---

## Structure

Each decoder block FFN:

1. `ffn_norm` (RMSNorm)
2. SwiGLU: gate (`w1`) × up (`w3`) → down (`w2`)
3. Residual add

## Dimensions

- Hidden \(D\) = `hidden_size` / GGUF `embedding_length`
- Intermediate \(I\) = `intermediate_size` / GGUF `feed_forward_length`
- Tiny defaults: \(D=768\), \(I=2048\)

## Naming

| Odyssey | GGUF |
| --- | --- |
| `layers.{i}.feed_forward.w1.weight` | `blk.{i}.ffn_gate.weight` |
| `...w3.weight` | `blk.{i}.ffn_up.weight` |
| `...w2.weight` | `blk.{i}.ffn_down.weight` |

No biases.
