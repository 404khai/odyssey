"""Tiny residual-flow diagram writer for assets/rmsnorm/."""

from __future__ import annotations

from pathlib import Path

from model.residual import describe_residual_flow


def export_rmsnorm_assets(output_dir: Path, *, hidden_size: int = 768) -> Path:
    """Write a text residual inspector artifact (no heavy plotting dependency)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flow = describe_residual_flow(hidden_size=hidden_size, batch=1, seq_len=16)
    lines = [
        "Odyssey Pre-Norm Residual Flow",
        "==============================",
        f"ordering: {flow['ordering']}",
        "",
    ]
    for step in flow["steps"]:
        stage = step["stage"]
        shape = step.get("shape", "")
        op = step.get("op", "")
        extra = f"  ({op})" if op else ""
        lines.append(f"{stage}: {shape}{extra}")
    lines.append("")
    lines.append(str(flow["notes"]))
    path = output_dir / "residual_flow.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
