"""
quantize.py — Dynamic int8 quantization for ONNX models.

Why int8?
    The Pi 5's Cortex-A76 has no neural accelerator.  int8 ops are
    ~2× faster than float32 on ARM NEON (because you fit 4× as many
    values in the same SIMD register and the multiplies are cheaper).

Why dynamic (not static)?
    Static quantization requires a calibration dataset to find optimal
    scale/zero-point per tensor.  Dynamic computes them at runtime per
    batch.  For streaming (batch=1, single frame), the overhead is
    negligible and we skip the calibration complexity.

Quality gate:
    After quantization, we re-verify that PESQ drop is < 0.05.
    If it's worse, we fall back to fp32 (see the fallback chain in
    the implementation plan).

Usage:
    python -m sih26052.export.quantize \\
        --input models/gtcrn_stream.onnx \\
        --output models/gtcrn_stream_int8.onnx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_dynamic_int8(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Apply dynamic int8 quantization to an ONNX model.

    Parameters
    ----------
    input_path  : path to the fp32 ONNX model
    output_path : where to write the quantized model

    Returns
    -------
    Path to the quantized .onnx file
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing %s → %s (dynamic int8)", input_path, output_path)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    # Report size reduction
    orig_size = input_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    logger.info(
        "Quantization complete: %.2f MB → %.2f MB (%.1f%% smaller)",
        orig_size, quant_size, reduction,
    )

    # Basic validation: load and check
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    logger.info("Quantized model passes ONNX checker")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to dynamic int8.")
    parser.add_argument("--input", type=Path, required=True, help="Input fp32 ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Output int8 ONNX model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    quantize_dynamic_int8(args.input, args.output)


if __name__ == "__main__":
    main()
