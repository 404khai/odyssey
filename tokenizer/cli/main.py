"""CLI entrypoint for ``odyssey-tokenizer``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from odyssey.config import REPO_ROOT
from odyssey_tokenizer import OdysseyTokenizer, load_bpe_config
from odyssey_tokenizer.visualizer import (
    render_merge_steps,
    write_compression_graph_png,
    write_merge_visualization_png,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odyssey-tokenizer",
        description="Odyssey byte-level BPE tokenizer (owned library).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a BPE tokenizer from a corpus")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "tokenizer.yaml"
    )
    train.add_argument("--vocab-size", type=int, default=None)
    train.add_argument("--max-lines", type=int, default=None)
    train.add_argument("--output", type=Path, default=None)
    train.add_argument("--min-frequency", type=int, default=None)

    encode = sub.add_parser("encode", help="Encode text to token IDs")
    encode.add_argument("--model", type=Path, required=True)
    encode.add_argument("--text", type=str, required=True)
    encode.add_argument("--add-bos", action="store_true")
    encode.add_argument("--add-eos", action="store_true")

    decode = sub.add_parser("decode", help="Decode token IDs to text")
    decode.add_argument("--model", type=Path, required=True)
    decode.add_argument(
        "--ids",
        type=str,
        required=True,
        help="Comma-separated token IDs, e.g. 12,45,90",
    )

    inspect = sub.add_parser("inspect", help="Inspect tokenization of a string")
    inspect.add_argument("--model", type=Path, required=True)
    inspect.add_argument("--text", type=str, required=True)
    inspect.add_argument("--show-specials", action="store_true")
    inspect.add_argument("--show-merges", action="store_true")

    bench = sub.add_parser("benchmark", help="Benchmark encode/decode performance")
    bench.add_argument("--model", type=Path, required=True)
    bench.add_argument("--input", type=Path, required=True)
    bench.add_argument("--limit", type=int, default=200)

    viz = sub.add_parser("visualize", help="Write merge/compression plots")
    viz.add_argument("--model", type=Path, required=True)
    viz.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "tokenizer" / "bpe",
    )
    viz.add_argument("--input", type=Path, default=None)
    viz.add_argument("--limit", type=int, default=50)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_bpe_config(args.config)
        if args.vocab_size is not None:
            config.vocab_size = args.vocab_size
        if args.max_lines is not None:
            config.training.max_lines = args.max_lines
        if args.min_frequency is not None:
            config.min_frequency = args.min_frequency
        output = args.output or config.model_dir_path()
        tokenizer, result = OdysseyTokenizer.train(
            args.input, config=config, save_path=output
        )
        print("Training complete")
        print(f"  model:        {output}")
        print(f"  vocab_size:   {result.vocab_size}")
        print(f"  merges:       {result.merge_count}")
        print(f"  seconds:      {result.training_seconds:.3f}")
        print(f"  corpus_lines: {result.corpus_lines}")
        print(f"  corpus_bytes: {result.corpus_bytes}")
        sample = tokenizer.inspect("Build authentication API")
        print()
        print(sample.render())
        return 0

    if args.command == "encode":
        tokenizer = OdysseyTokenizer.load(args.model)
        ids = tokenizer.encode(args.text, add_bos=args.add_bos, add_eos=args.add_eos)
        print(ids)
        return 0

    if args.command == "decode":
        tokenizer = OdysseyTokenizer.load(args.model)
        ids = [int(part.strip()) for part in args.ids.split(",") if part.strip()]
        print(tokenizer.decode(ids))
        return 0

    if args.command == "inspect":
        tokenizer = OdysseyTokenizer.load(args.model)
        print(tokenizer.inspect(args.text).render())
        if args.show_specials:
            print()
            print(tokenizer.special_tokens_table())
        if args.show_merges:
            print()
            print(render_merge_steps(tokenizer.merges, limit=15))
        return 0

    if args.command == "benchmark":
        from tokenizer.benchmarks.suite import run_benchmark

        report = run_benchmark(args.model, args.input, limit=args.limit)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "visualize":
        tokenizer = OdysseyTokenizer.load(args.model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        merge_png = write_merge_visualization_png(
            tokenizer.merges, args.output_dir / "merge_visualization.png"
        )
        print(f"Wrote {merge_png}")
        print(render_merge_steps(tokenizer.merges, limit=10))
        if args.input and args.input.is_file():
            lines = [
                line.strip()
                for line in args.input.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][: args.limit]
            chars = [len(tokenizer.normalizer.normalize(line)) for line in lines]
            toks = [len(tokenizer.encode(line)) for line in lines]
            graph = write_compression_graph_png(
                chars, toks, args.output_dir / "compression_graph.png"
            )
            print(f"Wrote {graph}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
