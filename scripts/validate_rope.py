#!/usr/bin/env python3
"""Cross-implementation RoPE validation: Odyssey (PyTorch) vs Phalanx (Rust).

Workflow
--------
Architecture Spec → Odyssey Implementation → this script → Phalanx Runtime
→ Numerical Comparison → PASS / FAIL

Procedure
1. Generate identical random Q, K (float32).
2. Apply Odyssey ``OdysseyRoPE``.
3. Serialize inputs + manifest for the Phalanx ``validate_rope`` binary.
4. Run Phalanx RoPE on the same tensors.
5. Compare max / mean absolute error against a tolerance (default 1e-6).

Example
-------
    python scripts/validate_rope.py
    python scripts/validate_rope.py --tolerance 1e-6 --seed 0
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

from model import OdysseyRoPE, RopeConfig
from odyssey.config import REPO_ROOT

PHALANX_ROOT = REPO_ROOT.parent / "runtime"
DEFAULT_REPORT = REPO_ROOT / "experiments" / "ODY-0004" / "rope_validation.json"


def _write_f32(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.float32)
    return data.reshape(shape).copy()


def run_odyssey(
    q: torch.Tensor,
    k: torch.Tensor,
    config: RopeConfig,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = OdysseyRoPE(config)
    return rope.apply_rotary(q, k, position_offset=position_offset)


def run_phalanx(
    work_dir: Path,
    *,
    phalanx_root: Path,
    release: bool,
) -> None:
    """Invoke Phalanx via ``cargo run`` so ``CARGO_TARGET_DIR`` overrides work."""
    cmd = ["cargo", "run", "--quiet", "--bin", "validate_rope"]
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
    passed = max_err <= tolerance
    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "tolerance": tolerance,
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--max-position", type=int, default=4096)
    parser.add_argument("--position-offset", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0, help="linear scale factor")
    parser.add_argument("--tolerance", type=float, default=1e-6)
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

    # Phalanx layout: [seq, heads, head_dim]
    shape = (args.seq, args.heads, args.head_dim)
    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(shape, dtype=torch.float32)

    scaling = "linear" if args.scale != 1.0 else "none"
    config = RopeConfig(
        theta=args.theta,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        max_position_embeddings=args.max_position,
        scaling=scaling,  # type: ignore[arg-type]
        scaling_factor=args.scale,
        device="cpu",
        dtype="float32",
    )

    q_ody, k_ody = run_odyssey(q, k, config, args.position_offset)

    work = (
        Path(tempfile.mkdtemp(prefix="rope_val_"))
        if args.keep_work_dir is None
        else args.keep_work_dir
    )
    work.mkdir(parents=True, exist_ok=True)

    manifest = {
        "shape": list(shape),
        "head_dim": args.head_dim,
        "rotary_dim": args.rotary_dim,
        "theta": args.theta,
        "scale": args.scale,
        "max_position": args.max_position,
        "position_offset": args.position_offset,
        "seed": args.seed,
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_f32(work / "q_in.bin", q.numpy())
    _write_f32(work / "k_in.bin", k.numpy())
    _write_f32(work / "q_ody.bin", q_ody.detach().numpy())
    _write_f32(work / "k_ody.bin", k_ody.detach().numpy())

    run_phalanx(work, phalanx_root=args.phalanx_root, release=args.release)

    q_ph = _read_f32(work / "q_out.bin", shape)
    k_ph = _read_f32(work / "k_out.bin", shape)

    q_cmp = compare(q_ody.detach().numpy(), q_ph, tolerance=args.tolerance)
    k_cmp = compare(k_ody.detach().numpy(), k_ph, tolerance=args.tolerance)
    max_err = max(float(q_cmp["max_error"]), float(k_cmp["max_error"]))
    mean_err = (float(q_cmp["mean_error"]) + float(k_cmp["mean_error"])) / 2.0
    passed = max_err <= args.tolerance

    report = {
        "component": "RoPE",
        "odyssey_spec": "1.0.0",
        "manifest": manifest,
        "q": q_cmp,
        "k": k_cmp,
        "max_error": max_err,
        "mean_error": mean_err,
        "tolerance": args.tolerance,
        "status": "PASS" if passed else "FAIL",
        "work_dir": str(work),
        "message": (
            "Odyssey and Phalanx are mathematically identical."
            if passed
            else "RoPE outputs diverge beyond tolerance."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("RoPE Validation")
    print()
    print("Max Error")
    print(f"{max_err:.8f}")
    print()
    print("Mean Error")
    print(f"{mean_err:.8f}")
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
