#!/usr/bin/env python3
"""
verify_export.py — Verify exported ONNX model against PyTorch or check numeric validity.
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sih26052.export.verify import verify_onnx_vs_pytorch, verify_quantized_quality


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX export correctness")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint path")
    parser.add_argument("--int8-onnx", type=Path, default=None, help="Quantized ONNX model path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to test")
    args = parser.parse_args()

    if args.int8_onnx:
        print(f"Verifying quantized model {args.int8_onnx} against fp32 model {args.onnx}...")
        res = verify_quantized_quality(args.onnx, args.int8_onnx, n_frames=args.frames)
        print(f"Quantization verification result: {res}")
        sys.exit(0 if res.get("passed", False) else 1)
    else:
        print(f"Verifying streaming ONNX model {args.onnx}...")
        res = verify_onnx_vs_pytorch(args.checkpoint or "", args.onnx, n_frames=args.frames)
        print(f"ONNX verification result: {res}")
        sys.exit(0 if res.get("passed", False) else 1)


if __name__ == "__main__":
    main()
