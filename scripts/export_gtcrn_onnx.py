#!/usr/bin/env python3
"""
export_gtcrn_onnx.py — Convert fine-tuned GTCRN to streaming ONNX and quantize to int8.
"""
import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "models" / "gtcrn"))
sys.path.insert(0, str(repo_root / "models" / "gtcrn" / "stream"))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import torch
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    default_ckpt = repo_root / "models" / "checkpoints" / "checkpoint_best.pth"
    if not default_ckpt.exists():
        default_ckpt = repo_root / "models" / "checkpoints" / "checkpoint_epoch_004.pth"

    parser = argparse.ArgumentParser(description="Export fine-tuned GTCRN to streaming ONNX")
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt)
    parser.add_argument("--out-onnx", type=Path, default=repo_root / "models" / "gtcrn_finetuned_stream.onnx")
    parser.add_argument("--out-int8", type=Path, default=repo_root / "models" / "gtcrn_finetuned_stream_int8.onnx")
    args = parser.parse_args()

    device = torch.device("cpu")

    # 1. Load offline GTCRN
    model = GTCRN().to(device).eval()
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    print("Loaded fine-tuned weights from", args.checkpoint)

    # 2. Convert to streaming GTCRN
    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)
    print("Successfully mapped weights to StreamGTCRN!")

    # 3. Export to ONNX
    conv_cache = torch.zeros(2, 1, 16, 16, 33, device=device)
    tra_cache = torch.zeros(2, 3, 1, 1, 16, device=device)
    inter_cache = torch.zeros(2, 1, 33, 16, device=device)
    dummy_input = torch.randn(1, 257, 1, 2, device=device)

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        stream_model,
        (dummy_input, conv_cache, tra_cache, inter_cache),
        str(args.out_onnx),
        input_names=["mix", "conv_cache", "tra_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
        opset_version=18,
        verbose=False,

    )
    print(f"Exported streaming ONNX model to {args.out_onnx} ({args.out_onnx.stat().st_size / 1024:.1f} KB)")

    # 4. Quantize to INT8
    quantize_dynamic(
        model_input=str(args.out_onnx),
        model_output=str(args.out_int8),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized INT8 ONNX model to {args.out_int8} ({args.out_int8.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
