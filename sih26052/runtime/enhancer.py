"""
enhancer.py — ONNX Runtime session wrapper with streaming state management.

This module wraps the ONNX model and manages the streaming state tensors
(RNN hidden states, conv caches) that must be passed in/out every frame.

The key invariant: state_out from frame N becomes state_in for frame N+1.
Get this wrong and the model output degrades to noise after frame 1.

NO torch imports — only numpy + onnxruntime.

Usage:
    from sih26052.runtime.enhancer import StreamingEnhancer

    enhancer = StreamingEnhancer("models/gtcrn_stream_int8.onnx")

    # In the audio callback:
    enhanced_spec = enhancer.process_frame(spec)  # (n_freq, 2)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class StreamingEnhancer:
    """ONNX-based frame-by-frame speech enhancer.

    Manages streaming state tensors automatically.  Thread-safe if
    only called from a single thread (the audio callback).
    """

    def __init__(self, onnx_path: str | Path, n_freq: int = 257):
        """
        Parameters
        ----------
        onnx_path : path to the streaming ONNX model
        n_freq    : number of frequency bins (257 for nfft=512)
        """
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.n_freq = n_freq

        # ── Create session ──
        # Use only CPU provider — no CUDA on Pi 5
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1  # single thread — predictable latency
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # ── Discover I/O ──
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        # ── Initialize states to zero ──
        self._states: dict[str, np.ndarray] = {}
        for inp in self._session.get_inputs():
            if inp.name != "spec_frame":
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                self._states[inp.name] = np.zeros(shape, dtype=np.float32)

        logger.info(
            "StreamingEnhancer loaded: %s (%d inputs, %d outputs, %d state tensors)",
            onnx_path, len(self._input_names), len(self._output_names),
            len(self._states),
        )

        # ── Warm-up run (before audio stream opens) ──
        dummy_spec = np.zeros((self.n_freq, 2), dtype=np.float32)
        self.process_frame(dummy_spec)
        self.reset()
        logger.info("StreamingEnhancer warmed up successfully")

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        """Enhance a single STFT frame.

        Parameters
        ----------
        spec : float32 array of shape (n_freq, 2) — [real, imag] per bin

        Returns
        -------
        enhanced : float32 array of shape (n_freq, 2) — enhanced frame

        The internal state tensors are updated automatically.
        """
        # Reshape to model's expected input: (1, n_freq, 1, 2)
        spec_input = spec[np.newaxis, :, np.newaxis, :]

        # Build feed dict
        feed = {"spec_frame": spec_input}
        feed.update(self._states)

        # Run inference
        outputs = self._session.run(self._output_names, feed)

        # Extract enhanced frame
        enhanced = outputs[0]  # shape: (1, n_freq, 1, 2)
        enhanced = enhanced[0, :, 0, :]  # → (n_freq, 2)

        # Update states: output states become input states for next frame
        for i, name in enumerate(self._output_names):
            if name == "enhanced_frame":
                continue
            # Map output name back to input name
            in_name = name.replace("_out", "")
            if in_name in self._states:
                self._states[in_name] = outputs[i]

        return enhanced

    def reset(self) -> None:
        """Reset all streaming states to zero.

        Call this when switching audio sources or after a gap in the
        input stream.  Without reset, stale states from the previous
        audio will leak into the new stream for several frames.
        """
        for name in self._states:
            self._states[name][:] = 0.0
        logger.debug("Enhancer states reset")

    @property
    def state_count(self) -> int:
        """Number of streaming state tensors."""
        return len(self._states)
