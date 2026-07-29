# RoPE Visualization Guide

Artifacts are written to `assets/rope/` by `model.rope_visualizer.export_rope_assets`.

| Figure | Meaning |
| --- | --- |
| `inv_freq.png` | Wavelength spectrum across pair indices |
| `cos_sin_curves.png` | Cos/sin vs position for one pair |
| `rotation_demo.png` | Unit vector `(1,0)` tracing a circle across positions |

Regenerate:

```bash
python -c "from model.rope_visualizer import export_rope_assets; export_rope_assets()"
```

Intuition: each pair is a 2D vector spun by an angle \(m\omega_i\); magnitude stays constant, only orientation encodes position.
