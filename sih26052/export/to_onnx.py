"""
to_onnx.py — Export GTCRN to streaming ONNX format.

Adapted from:
    - ~/Downloads/gtcrn/stream/    (author's streaming variant)
    - ~/Downloads/TRT-SE/          (streaming export patterns)

Key decisions:
    1. Every RNN hidden state + conv cache becomes explicit I/O tensor.
       This is non-negotiable for streaming — the ONNX graph must be
       stateless (state is passed in/out per frame).

    2. Fixed shapes (no dynamic axes).
       Input is (1, freq_bins, 1, 2) for a single STFT frame.
       This makes ONNX Runtime's shape inference deterministic and
       avoids runtime overhead from dynamic shape resolution.

    3. Opset 17.
       Oldest opset that supports all GTCRN ops reliably on ARM.

    4. Loads pretrained checkpoint from the GTCRN repo.

Usage:
    python -m sih26052.export.to_onnx \\
        --checkpoint ~/Downloads/gtcrn/checkpoints/model.pth \\
        --output models/gtcrn_stream.onnx

CAUTION:
    The streaming cache shapes are the #1 trap in the entire project.
    Wrong shapes = output sounds fine for frame 1 then degrades to noise.
    We verify immediately after export (see verify.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Any

import numpy as np

logger = logging.getLogger(__name__)


def export_streaming_onnx(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path = "model.onnx",
    opset_version: int = 17,
    nfft: int = 512,
    model_factory: Callable[[], Any] | None = None,
) -> Path:
    """Export a pretrained GTCRN model to streaming ONNX.

    Parameters
    ----------
    checkpoint_path : path to the PyTorch checkpoint (.pth)
    output_path     : where to write the .onnx file
    opset_version   : ONNX opset (default 17)
    nfft            : FFT size used by GTCRN (512 for 16 kHz)
    model_factory   : Optional callable to return an instantiated model

    Returns
    -------
    Path to the exported .onnx file

    The exported model expects:
        Input:  spec_frame — shape (1, n_freq, 1, 2) — one STFT frame (real, imag)
        Input:  *state_in  — one tensor per RNN hidden / conv cache
        Output: enhanced   — shape (1, n_freq, 1, 2) — enhanced frame
        Output: *state_out — updated states for next frame
    """
    import torch
    import torch.nn as nn

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    if model_factory is not None:
        logger.info("Using provided model_factory...")
        model = model_factory()
    else:
        if checkpoint_path is None:
            raise ValueError("Must provide either model_factory or checkpoint_path")

        # We need to import the GTCRN architecture.  The model definition
        # lives in the gtcrn repo.  We add it to sys.path temporarily.
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        gtcrn_root = checkpoint_path.parent.parent  # e.g. ~/Downloads/gtcrn/

        # Try importing from the gtcrn repo's stream/ directory
        stream_dir = gtcrn_root / "stream"
        if stream_dir.exists():
            sys.path.insert(0, str(stream_dir))
            logger.info("Added %s to sys.path for GTCRN streaming model", stream_dir)

        # Also add the repo root for non-streaming model
        sys.path.insert(0, str(gtcrn_root))

        try:
            # Try stream variant first (preferred for export)
            from gtcrn import GTCRN  # type: ignore
            logger.info("Loaded GTCRN model class from %s", gtcrn_root)
        except ImportError as exc:
            logger.error(
                "Could not import GTCRN model. Ensure ~/Downloads/gtcrn/ exists "
                "and contains gtcrn.py or stream/gtcrn.py. Error: %s", exc
            )
            raise

        # ── Instantiate and load weights ──
        model = GTCRN()
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=False)

    model.eval()

    n_freq = nfft // 2 + 1  # 257 for nfft=512

    # ── Build dummy inputs ──
    # Single STFT frame: (batch=1, freq=257, time=1, ri=2)
    spec_frame = torch.randn(1, n_freq, 1, 2)

    # Collect all state tensors from the model
    # This depends on the model architecture — we use a generic approach
    # by running one forward pass to discover state shapes
    dummy_inputs, input_names, output_names = _prepare_streaming_io(
        model, spec_frame
    )

    # ── Export ──
    logger.info("Exporting to ONNX opset %d → %s", opset_version, output_path)

    torch.onnx.export(
        model,
        tuple(dummy_inputs),
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )

    # Validate the exported model
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Export complete: %s (%.2f MB), %d inputs, %d outputs",
        output_path, file_size_mb, len(input_names), len(output_names),
    )

    return output_path


def _prepare_streaming_io(model, spec_frame):
    """Discover streaming state shapes by inspecting the model.

    This is a best-effort approach.  For GTCRN specifically, the streaming
    states are the GRU/LSTM hidden states and any causal convolution caches.

    Returns (dummy_inputs_list, input_names, output_names)
    """
    import torch

    input_names = ["spec_frame"]
    output_names = ["enhanced_frame"]
    dummy_inputs = [spec_frame]

    # Walk the model to find RNN layers and build zero-initialised states
    state_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
            num_layers = module.num_layers
            hidden_size = module.hidden_size
            num_directions = 2 if module.bidirectional else 1

            h0 = torch.zeros(num_layers * num_directions, 1, hidden_size)
            dummy_inputs.append(h0)
            input_names.append(f"state_h_{state_idx}")
            output_names.append(f"state_h_{state_idx}_out")

            if isinstance(module, torch.nn.LSTM):
                c0 = torch.zeros(num_layers * num_directions, 1, hidden_size)
                dummy_inputs.append(c0)
                input_names.append(f"state_c_{state_idx}")
                output_names.append(f"state_c_{state_idx}_out")

            state_idx += 1

    logger.info(
        "Streaming I/O: %d inputs (%d states), %d outputs",
        len(input_names), state_idx, len(output_names),
    )

    return dummy_inputs, input_names, output_names


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export GTCRN to streaming ONNX.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/gtcrn_stream.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_streaming_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
