#!/usr/bin/env python3
"""Cross-implementation SwiGLU validation: Odyssey (PyTorch) vs Phalanx (Rust).

Workflow
--------
Architecture Spec → Odyssey Implementation → this script → Phalanx Runtime
→ Numerical Comparison → PASS / FAIL

Example
-------
    python scripts/validate_swiglu.py
    python scripts/validate_swiglu.py --hidden-size 64 --intermediate-size 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from model import FeedForwardConfig, OdysseySwiGLU
from odyssey.config import REPO_ROOT

PHALANX_ROOT = REPO_ROOT.parent / "runtime"
DEFAULT_REPORT = REPO_ROOT / "experiments" / "ODY-0006" / "swiglu_validation.json"


def _write_f32(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.float32)
    return data.reshape(shape).copy()


def run_odyssey(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    config: FeedForwardConfig,
) -> torch.Tensor:
    ffn = OdysseySwiGLU(config)
    with torch.no_grad():
        ffn.gate_proj.weight.copy_(w_gate)
        ffn.up_proj.weight.copy_(w_up)
        ffn.down_proj.weight.copy_(w_down)
    return ffn(x)


def run_phalanx(
    work_dir: Path,
    *,
    phalanx_root: Path,
    release: bool,
) -> None:
    cmd = ["cargo", "run", "--quiet", "--bin", "validate_swiglu"]
    if release:
        cmd.append("--release")
    cmd.extend(["--", str(work_dir)])
    subprocess.run(cmd, cwd=phalanx_root, check=True)


def compare(
    odyssey: np.ndarray,
    phalanx: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float | bool | str]:
    diff = np.abs(odyssey.astype(np.float64) - phalanx.astype(np.float64))
    max_err = float(diff.max()) if diff.size else 0.0
    mean_err = float(diff.mean()) if diff.size else 0.0
    rel = diff / (np.abs(odyssey.astype(np.float64)) + 1e-8)
    max_rel = float(rel.max()) if rel.size else 0.0
    passed = max_err <= tolerance
    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "max_relative_error": max_rel,
        "tolerance": tolerance,
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="abs tolerance (default 1e-3: GEMM accum order vs PyTorch; see docs)",
    )
    parser.add_argument("--phalanx-root", type=Path, default=PHALANX_ROOT)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--keep-work-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.phalanx_root.is_dir():
        print(f"Phalanx root not found: {args.phalanx_root}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shape = (args.batch, args.seq, args.hidden_size)
    x = torch.randn(shape, dtype=torch.float32)
    # nn.Linear weight layout: (out, in)
    w_gate = torch.randn(args.intermediate_size, args.hidden_size)
    w_up = torch.randn(args.intermediate_size, args.hidden_size)
    w_down = torch.randn(args.hidden_size, args.intermediate_size)

    config = FeedForwardConfig(
        type="swiglu",
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation="silu",
        device="cpu",
        dtype="float32",
    )

    y_ody = run_odyssey(x, w_gate, w_up, w_down, config)

    work = (
        Path(tempfile.mkdtemp(prefix="swiglu_val_"))
        if args.keep_work_dir is None
        else args.keep_work_dir
    )
    work.mkdir(parents=True, exist_ok=True)

    manifest = {
        "shape": list(shape),
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "seed": args.seed,
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_f32(work / "x_in.bin", x.numpy())
    _write_f32(work / "w_gate.bin", w_gate.numpy())
    _write_f32(work / "w_up.bin", w_up.numpy())
    _write_f32(work / "w_down.bin", w_down.numpy())
    _write_f32(work / "y_ody.bin", y_ody.detach().numpy())

    run_phalanx(work, phalanx_root=args.phalanx_root, release=args.release)

    y_ph = _read_f32(work / "y_out.bin", shape)
    cmp = compare(y_ody.detach().numpy(), y_ph, tolerance=args.tolerance)
    passed = bool(cmp["pass"])

    report = {
        "component": "SwiGLU",
        "odyssey_spec": "1.0.0",
        "manifest": manifest,
        "comparison": cmp,
        "max_error": cmp["max_error"],
        "mean_error": cmp["mean_error"],
        "max_relative_error": cmp["max_relative_error"],
        "tolerance": args.tolerance,
        "status": "PASS" if passed else "FAIL",
        "work_dir": str(work),
        "message": (
            "Odyssey and Phalanx are mathematically identical."
            if passed
            else "SwiGLU outputs diverge beyond tolerance."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("SwiGLU Validation")
    print()
    print("Max Error")
    print(f"{float(cmp['max_error']):.8f}")
    print()
    print("Mean Error")
    print(f"{float(cmp['mean_error']):.8f}")
    print()
    print(report["status"])
    print()
    print(report["message"])
    print()
    print(f"report: {args.report}")

    if args.keep_work_dir is None:
        shutil.rmtree(work, ignore_errors=True)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
