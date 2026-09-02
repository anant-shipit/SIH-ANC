#!/usr/bin/env python3
"""
eval_baseline.py — Run evaluation harness over a manifest.
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sih26052.eval.harness import run_eval_harness


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline or model on manifest")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()

    print(f"Running evaluation on manifest: {args.manifest}")
    results = run_eval_harness(args.manifest, enhance_fn=None, sr=args.sr, max_pairs=args.max_pairs)
    print("\nHeadline Results Table:")
    print(results.format_table())


if __name__ == "__main__":
    main()
