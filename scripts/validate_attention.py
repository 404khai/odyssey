#!/usr/bin/env python3
"""Cross-implementation Attention validation: Odyssey (PyTorch) vs Phalanx (Rust).

Workflow
--------
Architecture Spec → Odyssey Implementation → this script → Phalanx Runtime
→ Numerical Comparison → PASS / FAIL

Example
-------
    python scripts/validate_attention.py
    python scripts/validate_attention.py --num-heads 8 --num-kv-heads 2 --apply-rope
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

from model import AttentionConfig, OdysseyAttention, OdysseyRoPE, RopeConfig
from odyssey.config import REPO_ROOT

PHALANX_ROOT = REPO_ROOT.parent / "runtime"
DEFAULT_REPORT = REPO_ROOT / "experiments" / "ODY-0007" / "attention_validation.json"


def _write_f32(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.float32)
    return data.reshape(shape).copy()


def run_odyssey(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    config: AttentionConfig,
    *,
    apply_rope: bool,
    rope_theta: float,
    position_offset: int,
) -> torch.Tensor:
    rope = None
    if apply_rope:
        rope = OdysseyRoPE(
            RopeConfig(
                theta=rope_theta,
                head_dim=config.head_dim,
                rotary_dim=config.head_dim,
                max_position_embeddings=max(position_offset + x.shape[1], 64),
                device="cpu",
                dtype="float32",
            )
        )
    attn = OdysseyAttention(config, rope=rope)
    with torch.no_grad():
        attn.projections.q_proj.weight.copy_(w_q)
        attn.projections.k_proj.weight.copy_(w_k)
        attn.projections.v_proj.weight.copy_(w_v)
        attn.projections.o_proj.weight.copy_(w_o)
    return attn(x, position_offset=position_offset)


def run_phalanx(
    work_dir: Path,
    *,
    phalanx_root: Path,
    release: bool,
) -> None:
    cmd = ["cargo", "run", "--quiet", "--bin", "validate_attention"]
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
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--apply-rope", action="store_true", default=True)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="abs tolerance (default 1e-3: GEMM accum order; see docs)",
    )
    parser.add_argument("--phalanx-root", type=Path, default=PHALANX_ROOT)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--keep-work-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    apply_rope = args.apply_rope and not args.no_rope

    if not args.phalanx_root.is_dir():
        print(f"Phalanx root not found: {args.phalanx_root}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hidden = args.num_heads * args.head_dim
    shape = (args.batch, args.seq, hidden)
    x = torch.randn(shape, dtype=torch.float32)
    w_q = torch.randn(hidden, hidden)
    w_k = torch.randn(args.num_kv_heads * args.head_dim, hidden)
    w_v = torch.randn(args.num_kv_heads * args.head_dim, hidden)
    w_o = torch.randn(hidden, hidden)

    config = AttentionConfig(
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=hidden,
        dropout=0.0,
        causal=True,
        bias=False,
        device="cpu",
        dtype="float32",
    )

    y_ody = run_odyssey(
        x,
        w_q,
        w_k,
        w_v,
        w_o,
        config,
        apply_rope=apply_rope,
        rope_theta=args.rope_theta,
        position_offset=args.position_offset,
    )

    work = (
        Path(tempfile.mkdtemp(prefix="attn_val_"))
        if args.keep_work_dir is None
        else args.keep_work_dir
    )
    work.mkdir(parents=True, exist_ok=True)

    manifest = {
        "shape": list(shape),
        "hidden_size": hidden,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "apply_rope": apply_rope,
        "rope_theta": args.rope_theta,
        "position_offset": args.position_offset,
        "seed": args.seed,
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_f32(work / "x_in.bin", x.numpy())
    _write_f32(work / "w_q.bin", w_q.numpy())
    _write_f32(work / "w_k.bin", w_k.numpy())
    _write_f32(work / "w_v.bin", w_v.numpy())
    _write_f32(work / "w_o.bin", w_o.numpy())
    _write_f32(work / "y_ody.bin", y_ody.detach().numpy())

    run_phalanx(work, phalanx_root=args.phalanx_root, release=args.release)

    y_ph = _read_f32(work / "y_out.bin", shape)
    cmp = compare(y_ody.detach().numpy(), y_ph, tolerance=args.tolerance)
    passed = bool(cmp["pass"])

    report = {
        "component": "Attention",
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
            else "Attention outputs diverge beyond tolerance."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Attention Validation")
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
