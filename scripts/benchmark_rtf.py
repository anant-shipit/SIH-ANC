#!/usr/bin/env python3
"""
benchmark_rtf.py — Measure Real-Time Factor (RTF) of the streaming ONNX model.
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sih26052.export.benchmark import benchmark_rtf


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model RTF")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulated audio duration in seconds")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--threads", type=int, default=1, help="Number of intra-op threads (default: 1)")
    args = parser.parse_args()

    print(f"Benchmarking RTF for {args.onnx} ({args.duration}s audio, {args.runs} runs, {args.threads} thread(s))...")
    res = benchmark_rtf(args.onnx, duration_s=args.duration, n_runs=args.runs, num_threads=args.threads)
    print(f"\nResult: Median RTF = {res['rtf_median']:.4f} ({res['inference_ms_per_frame']:.3f} ms/frame)")
    if res['rtf_median'] < 0.5:
        print("PASS: RTF < 0.5 meets real-time budget with headroom.")
    elif res['rtf_median'] < 1.0:
        print("WARNING: Real-time capable but headroom < 2x.")
    else:
        print("FAIL: RTF >= 1.0 cannot sustain real-time processing.")


if __name__ == "__main__":
    main()
