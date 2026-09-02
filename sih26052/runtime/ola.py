"""
ola.py — Overlap-Add engine with sqrt-Hann windowing.

This is the signal-level building block of the real-time loop.  It
handles the STFT analysis window on input and OLA synthesis window on
output, frame by frame.

Design:
    - 512-sample window, 256-sample hop (50% overlap)
    - sqrt-Hann on analysis × sqrt-Hann on synthesis = Hann
    - Hann at 50% overlap sums to a flat 1.0 (COLA condition)
    - No amplitude wobble — verified by test_ola.py

Why sqrt-Hann instead of Hann + Hann?
    Hann × Hann = Hann² which does NOT sum flat at 50% overlap.
    sqrt(Hann) × sqrt(Hann) = Hann which DOES sum flat.
    This is a common gotcha in real-time OLA systems.

Why not use librosa/scipy STFT?
    Those are batch-mode (process entire file).  We need frame-by-frame
    streaming with explicit state management.

NO torch imports — this runs on the Pi with only numpy.
"""
from __future__ import annotations

import numpy as np


class OverlapAdd:
    """Frame-by-frame STFT + OLA reconstruction engine.

    Usage:
        ola = OverlapAdd(nfft=512, hop=256)

        # In the audio callback:
        spec = ola.analyze(new_samples)       # (nfft//2+1, 2) complex→ real,imag
        enhanced_spec = model(spec)            # neural net processes spec
        output = ola.synthesize(enhanced_spec) # (hop,) output samples
    """

    def __init__(self, nfft: int = 512, hop: int = 256):
        """
        Parameters
        ----------
        nfft : FFT size (window length)
        hop  : hop size (= callback block size)
        """
        assert nfft % hop == 0, f"nfft ({nfft}) must be divisible by hop ({hop})"
        assert nfft // hop == 2, "Only 50% overlap (nfft = 2 × hop) is supported"

        self.nfft = nfft
        self.hop = hop
        self.n_freq = nfft // 2 + 1  # 257 for nfft=512

        # ── Windows ──
        # sqrt-Hann: analysis × synthesis = Hann = COLA-compliant at 50% overlap
        hann = np.hanning(nfft).astype(np.float32)
        self.analysis_window = np.sqrt(hann)
        self.synthesis_window = np.sqrt(hann)

        # ── Buffers ──
        # Input ring buffer: accumulates incoming samples until we have a full frame
        self._input_buffer = np.zeros(nfft, dtype=np.float32)

        # Output overlap buffer: holds the previous frame's tail for OLA
        self._output_overlap = np.zeros(hop, dtype=np.float32)

    def analyze(self, samples: np.ndarray) -> np.ndarray:
        """Accept *hop* new samples, return the STFT of the current frame.

        Parameters
        ----------
        samples : float32 array of shape (hop,)

        Returns
        -------
        spec : float32 array of shape (n_freq, 2) — [real, imag] per bin

        The input buffer slides: old samples shift left, new ones go right.
        """
        assert len(samples) == self.hop, f"Expected {self.hop} samples, got {len(samples)}"

        # Shift buffer left by hop, append new samples
        self._input_buffer[:self.hop] = self._input_buffer[self.hop:]
        self._input_buffer[self.hop:] = samples

        # Apply analysis window
        windowed = self._input_buffer * self.analysis_window

        # FFT → complex → split to (real, imag)
        spec_complex = np.fft.rfft(windowed)
        spec = np.stack([spec_complex.real, spec_complex.imag], axis=-1)

        return spec.astype(np.float32)

    def synthesize(self, spec: np.ndarray) -> np.ndarray:
        """Reconstruct *hop* samples from a (possibly modified) spectrum.

        Parameters
        ----------
        spec : float32 array of shape (n_freq, 2) — [real, imag]

        Returns
        -------
        output : float32 array of shape (hop,) — output samples ready for playback
        """
        # Reconstruct complex spectrum
        spec_complex = spec[:, 0] + 1j * spec[:, 1]

        # IFFT
        frame = np.fft.irfft(spec_complex, n=self.nfft).astype(np.float32)

        # Apply synthesis window
        frame *= self.synthesis_window

        # Overlap-add: add first half to previous overlap, store second half
        output = frame[:self.hop] + self._output_overlap
        self._output_overlap = frame[self.hop:].copy()

        return output

    def reset(self) -> None:
        """Clear all internal buffers (e.g. when switching audio sources)."""
        self._input_buffer[:] = 0.0
        self._output_overlap[:] = 0.0

    @property
    def latency_samples(self) -> int:
        """Algorithmic latency in samples (excludes ALSA buffers)."""
        return self.nfft  # one full window must fill before first output

    @property
    def latency_ms(self, sr: int = 16000) -> float:
        """Algorithmic latency in milliseconds."""
        return self.latency_samples / sr * 1000
