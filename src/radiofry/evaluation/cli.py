"""Command-line entry point for the Synthetic V1 evaluation harness."""

import argparse
from typing import Sequence

from .harness import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m radiofry.evaluation",
        description="Score the unmodified RadioFry pipeline against a Synthetic V1 dataset.",
    )
    parser.add_argument("--dataset", required=True, help="V1 dataset directory containing manifest.csv")
    parser.add_argument("--output", required=True, help="directory for results.csv / results.json / summary.md")
    parser.add_argument("--formats", nargs="+", default=["iq", "wav"], choices=["iq", "wav"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N captures")
    parser.add_argument("--model", default=None, help="CNN checkpoint to evaluate (default: the pipeline default)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_evaluation(args.dataset, args.output, formats=tuple(args.formats), limit=args.limit,
                             model_path=args.model)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0
