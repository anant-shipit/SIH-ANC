"""
verify.py — Numerical verification: PyTorch vs ONNX Runtime output.

This is the safety net for the #1 risk: wrong streaming cache shapes.

We run the exact same input through both backends and assert that the
max absolute difference is < 1e-4.  If it's larger, the export is broken
and everything downstream (quantization, real-time loop) will produce
garbage.

Usage:
    python -m sih26052.export.verify \\
        --checkpoint ~/Downloads/gtcrn/checkpoints/model.pth \\
        --onnx models/gtcrn_stream.onnx
"""
from __future__ import annotations

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Callable, Any

import numpy as np

logger = logging.getLogger(__name__)


def verify_onnx_vs_pytorch(
    checkpoint_path: str | Path | None = None,
    onnx_path: str | Path = "model.onnx",
    n_frames: int = 100,
    tolerance: float = 1e-4,
    nfft: int = 512,
    hop: int = 256,
    audio_path: str | Path | None = None,
    model_factory: Callable[[], Any] | None = None,
) -> dict:
    """Compare PyTorch and ONNX Runtime outputs frame by frame.

    Parameters
    ----------
    checkpoint_path : path to PyTorch checkpoint
    onnx_path       : path to exported ONNX model
    n_frames        : number of frames to compare
    nfft            : FFT size
    tolerance       : max acceptable absolute difference

    Returns
    -------
    dict with keys:
        max_abs_diff : float — worst case difference
        mean_abs_diff : float — average difference
        passed : bool — True if max_abs_diff < tolerance
        n_frames : int
    """
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    n_freq = nfft // 2 + 1  # 257

    # ── Set up ONNX Runtime session ──
    sess = ort.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    logger.info("ONNX inputs: %s", input_names)
    logger.info("ONNX outputs: %s", output_names)

    # ── Set up PyTorch model ──
    import torch
    import sys
    
    if model_factory is not None:
        logger.info("Using provided model_factory...")
        model = model_factory()
    else:
        if checkpoint_path is None:
            raise ValueError("Must provide either model_factory or checkpoint_path")
            
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        gtcrn_root = checkpoint_path.parent.parent
        
        stream_dir = gtcrn_root / "stream"
        if stream_dir.exists():
            sys.path.insert(0, str(stream_dir))
        sys.path.insert(0, str(gtcrn_root))
        
        from gtcrn import GTCRN  # type: ignore
        model = GTCRN()
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()

    # ── Generate test sequence ──
    if audio_path:
        import soundfile as sf
        from sih26052.runtime.ola import OverlapAdd
        
        ola = OverlapAdd(nfft=nfft, hop=hop)
        logger.info("Loading %s for verification...", audio_path)
        frames = []
        with sf.SoundFile(str(audio_path)) as sf_in:
            for block in sf_in.blocks(blocksize=hop, dtype='float32', fill_value=0.0):
                if block.ndim > 1:
                    block = block[:, 0]
                spec = ola.analyze(block)
                frames.append(spec)
                if len(frames) >= n_frames:
                    break
        
        while len(frames) < n_frames:
            frames.append(ola.analyze(np.zeros(hop, dtype=np.float32)))
            
        spec_seq = np.stack(frames, axis=1) # shape: (n_freq, n_frames, 2)
        spec_seq = spec_seq[np.newaxis, ...] # shape: (1, n_freq, n_frames, 2)
    else:
        rng = np.random.default_rng(42)
        # Shape: (batch=1, freq=257, time=n_frames, complex=2)
        spec_seq = rng.standard_normal((1, n_freq, n_frames, 2)).astype(np.float32)

    # ── PyTorch full sequence ──
    with torch.no_grad():
        pt_out = model(torch.from_numpy(spec_seq)).numpy()

    # ── ONNX frame by frame ──
    onnx_states = {}
    for inp in sess.get_inputs():
        if inp.name != "spec_frame":
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            onnx_states[inp.name] = np.zeros(shape, dtype=np.float32)

    onnx_out_frames = []
    for t in range(n_frames):
        spec_frame = spec_seq[:, :, t:t+1, :]
        feed = {"spec_frame": spec_frame}
        feed.update(onnx_states)
        onnx_outputs = sess.run(output_names, feed)
        
        onnx_out_frames.append(onnx_outputs[0])
        
        for i, name in enumerate(output_names):
            if name != "enhanced_frame":
                state_in_name = name.replace("_out", "")
                if state_in_name in onnx_states:
                    onnx_states[state_in_name] = onnx_outputs[i]

    onnx_out = np.concatenate(onnx_out_frames, axis=2)

    # ── Compare ──
    diff = np.abs(pt_out - onnx_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    if max_diff > tolerance:
        logger.error(
            "Verification FAILED: max diff %.2e > tolerance %.2e",
            max_diff, tolerance,
        )
        passed = False
    else:
        logger.info(
            "Verification PASSED: max diff %.2e < tolerance %.2e",
            max_diff, tolerance,
        )
        passed = True

    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "passed": passed,
        "n_frames": n_frames,
    }


def verify_quantized_quality(
    fp32_onnx_path: str | Path,
    int8_onnx_path: str | Path,
    n_frames: int = 200,
    nfft: int = 512,
    tolerance: float = 0.01,
) -> dict:
    """Compare fp32 and int8 ONNX outputs to measure quantization error.

    Returns
    -------
    dict with max_abs_diff, mean_abs_diff, passed
    """
    import onnxruntime as ort

    n_freq = nfft // 2 + 1
    rng = np.random.default_rng(42)

    sess_fp32 = ort.InferenceSession(str(fp32_onnx_path))
    sess_int8 = ort.InferenceSession(str(int8_onnx_path))

    # Initialize states
    def init_states(sess):
        states = {}
        for inp in sess.get_inputs():
            if inp.name != "spec_frame":
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                states[inp.name] = np.zeros(shape, dtype=np.float32)
        return states

    states_fp32 = init_states(sess_fp32)
    states_int8 = init_states(sess_int8)

    output_names_fp32 = [o.name for o in sess_fp32.get_outputs()]
    output_names_int8 = [o.name for o in sess_int8.get_outputs()]

    max_diffs = []

    for frame_idx in range(n_frames):
        spec = rng.standard_normal((1, n_freq, 1, 2)).astype(np.float32)

        feed_fp32 = {"spec_frame": spec, **states_fp32}
        feed_int8 = {"spec_frame": spec, **states_int8}

        out_fp32 = sess_fp32.run(output_names_fp32, feed_fp32)
        out_int8 = sess_int8.run(output_names_int8, feed_int8)

        diff = np.max(np.abs(out_fp32[0] - out_int8[0]))
        max_diffs.append(diff)

        # Update states
        for i, name in enumerate(output_names_fp32):
            in_name = name.replace("_out", "")
            if in_name in states_fp32:
                states_fp32[in_name] = out_fp32[i]
        for i, name in enumerate(output_names_int8):
            in_name = name.replace("_out", "")
            if in_name in states_int8:
                states_int8[in_name] = out_int8[i]

    max_abs = float(np.max(max_diffs))
    mean_abs = float(np.mean(max_diffs))

    logger.info(
        "Quantization verification: max_diff=%.6f, mean_diff=%.6f, tolerance=%.4f",
        max_abs, mean_abs, tolerance,
    )

    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "passed": max_abs < tolerance,
        "n_frames": n_frames,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify ONNX export.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--audio", type=Path, default=None, help="Optional real WAV file to verify against")
    parser.add_argument("--frames", type=int, default=100, help="Frames to test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = verify_onnx_vs_pytorch(args.checkpoint, args.onnx, n_frames=args.frames, audio_path=args.audio)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
