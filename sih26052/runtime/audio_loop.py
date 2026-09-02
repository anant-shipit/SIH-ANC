"""
audio_loop.py — Real-time audio processing loop using sounddevice.

This is the main entry point for live speech enhancement.  It:
    1. Opens a sounddevice stream with the configured audio device.
    2. In the callback: reads mic input → STFT → enhance → ISTFT → output.
    3. Pushes metrics to a queue for the dashboard (non-blocking).
    4. Counts xruns via the sounddevice status flags.

Architecture:
    ┌─────────┐     ┌─────┐     ┌──────────┐     ┌─────┐     ┌──────────┐
    │ mic in  │────>│ OLA │────>│ enhancer │────>│ OLA │────>│ speaker  │
    │         │     │ .analyze  │ .process │     │ .synth    │ out      │
    └─────────┘     └─────┘     └──────────┘     └─────┘     └──────────┘
                                      │
                                      ├── ab_switch
                                      ├── impulse_gate (Phase 5)
                                      └── queue → dashboard

Rules for the callback:
    - NO memory allocation (everything preallocated)
    - NO print / logging / file I/O
    - NO locks (except the A/B switch's minimal lock)
    - NO blocking calls

NO torch imports — only numpy, sounddevice, onnxruntime.

Usage:
    python -m sih26052.runtime.audio_loop \\
        --onnx models/gtcrn_stream_int8.onnx \\
        --device 0
"""
from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioLoop:
    """Real-time speech enhancement loop.

    Parameters
    ----------
    onnx_path    : path to the streaming ONNX model
    device       : sounddevice device index (None = default)
    sr           : sample rate (must match model)
    nfft         : FFT size
    hop          : hop size (= callback block size)
    queue_size   : max items in the dashboard metrics queue
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: int | None = None,
        sr: int = 16000,
        nfft: int = 512,
        hop: int = 256,
        queue_size: int = 100,
    ):
        from sih26052.runtime.ola import OverlapAdd
        from sih26052.runtime.enhancer import StreamingEnhancer
        from sih26052.runtime.ab_switch import ABSwitch

        self.sr = sr
        self.hop = hop
        self.device = device

        # ── Processing chain ──
        self.ola = OverlapAdd(nfft=nfft, hop=hop)
        self.enhancer = StreamingEnhancer(onnx_path, n_freq=nfft // 2 + 1)
        self.ab_switch = ABSwitch(sr=sr)

        # ── Impulse gate placeholder (populated in Phase 5) ──
        self.impulse_gate = None

        # ── Metrics queue for dashboard (non-blocking push) ──
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # ── Xrun tracking ──
        self.xrun_count = 0
        self.frame_count = 0
        self.start_time = 0.0

        # ── Preallocated buffers ──
        self._raw_buffer = np.zeros(hop, dtype=np.float32)
        self._enhanced_buffer = np.zeros(hop, dtype=np.float32)

    def _callback(self, indata, outdata, frames, time_info, status):
        """Sounddevice stream callback.

        This runs in a separate high-priority thread.  It MUST NOT
        allocate memory, print, or call any blocking function.
        """
        # ── Track xruns ──
        if status:
            self.xrun_count += 1

        self.frame_count += 1

        # ── Get mono input ──
        # indata shape: (frames, channels) — take first channel
        mono_in = indata[:, 0].astype(np.float32)

        # ── STFT analysis ──
        spec = self.ola.analyze(mono_in)

        # ── Neural enhancement ──
        enhanced_spec = self.enhancer.process_frame(spec)

        # ── ISTFT synthesis ──
        self._raw_buffer[:] = mono_in[:self.hop]
        self._enhanced_buffer[:] = self.ola.synthesize(enhanced_spec)

        # ── Impulse gate (Phase 5 — no-op if not set) ──
        if self.impulse_gate is not None:
            self._enhanced_buffer = self.impulse_gate.process(self._enhanced_buffer)

        # ── A/B crossfade ──
        output = self.ab_switch.apply_vectorized(
            self._raw_buffer, self._enhanced_buffer
        )

        # ── Write to output ──
        outdata[:, 0] = output
        if outdata.shape[1] > 1:
            outdata[:, 1] = output  # duplicate to stereo if needed

        # ── Push metrics to dashboard queue (non-blocking) ──
        try:
            metrics = {
                "frame": self.frame_count,
                "spec_in": spec[:, 0].copy(),   # real part for spectrogram
                "spec_out": enhanced_spec[:, 0].copy(),
                "xruns": self.xrun_count,
                "enhanced": self.ab_switch.is_enhanced,
            }
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass  # Dashboard is behind — drop this frame's metrics

    def run(self, duration: float | None = None) -> None:
        """Start the real-time loop.

        Parameters
        ----------
        duration : run for this many seconds, then stop.
                   None = run until Ctrl+C.
        """
        import sounddevice as sd

        logger.info(
            "Starting audio loop: device=%s, sr=%d, hop=%d, model=%s",
            self.device, self.sr, self.hop, self.enhancer.onnx_path,
        )

        # ── Open stream ──
        stream = sd.Stream(
            device=self.device,
            samplerate=self.sr,
            blocksize=self.hop,
            channels=1,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )

        self.start_time = time.monotonic()

        # ── Graceful shutdown on Ctrl+C ──
        stop_event = False

        def handle_sigint(sig, frame):
            nonlocal stop_event
            stop_event = True

        signal.signal(signal.SIGINT, handle_sigint)

        with stream:
            logger.info("Audio stream active.  Press Ctrl+C to stop.")
            logger.info("Press SPACE to toggle A/B switch (if keyboard handler is active)")

            try:
                while not stop_event:
                    time.sleep(0.1)

                    # Duration limit
                    if duration and (time.monotonic() - self.start_time) >= duration:
                        break

                    # Periodic status
                    if self.frame_count > 0 and self.frame_count % 1000 == 0:
                        elapsed = time.monotonic() - self.start_time
                        logger.info(
                            "Frames: %d | Xruns: %d | Elapsed: %.1fs",
                            self.frame_count, self.xrun_count, elapsed,
                        )
            except KeyboardInterrupt:
                pass

        elapsed = time.monotonic() - self.start_time
        logger.info(
            "Audio loop stopped.  %d frames, %d xruns, %.1f seconds",
            self.frame_count, self.xrun_count, elapsed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time speech enhancement.")
    parser.add_argument("--onnx", type=Path, required=True, help="Streaming ONNX model")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--hop", type=int, default=256, help="Hop size")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    loop = AudioLoop(
        onnx_path=args.onnx,
        device=args.device,
        sr=args.sr,
        hop=args.hop,
    )
    loop.run(duration=args.duration)


if __name__ == "__main__":
    main()
