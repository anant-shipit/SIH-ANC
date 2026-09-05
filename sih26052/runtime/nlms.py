"""
nlms.py — Normalised Least Mean Squares adaptive filter for dual-mic noise suppression.

This is the dual-mic post-processing stage.  It uses a reference
microphone (facing away from the speaker, toward the noise source)
to predict and subtract residual noise from the primary mic signal
after neural enhancement.

How it works:
    1. Reference mic captures mostly noise (pointed away from speaker).
    2. NLMS filter adapts coefficients to predict what that noise looks
       like at the primary mic position.
    3. Subtract the prediction from the enhanced signal.
    4. Adapt coefficients per-sample using the error signal.

Why NLMS instead of RLS or Kalman?
    - NLMS is simpler, lighter, and converges within a few hundred ms
      for broadband noise — good enough for our use case.
    - RLS is faster to converge but heavier (matrix inversions).
    - On a Pi 5, per-sample coefficient updates must be cheap.

IMPORTANT:
    This entire module is DROPPED if no dual-mic HAT is available.
    The system works without it.

NO torch imports — only numpy.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NLMSFilter:
    """Adaptive NLMS filter for residual noise suppression.

    Usage:
        nlms = NLMSFilter(filter_length=256)

        # In audio callback (per sample or per frame):
        cleaned = nlms.process_frame(enhanced, reference)
    """

    def __init__(
        self,
        filter_length: int = 256,
        step_size: float = 0.1,
        regularization: float = 1e-6,
    ):
        """
        Parameters
        ----------
        filter_length  : number of taps in the adaptive filter
        step_size      : NLMS step size (μ). Range (0, 2). Higher = faster
                         adaptation but less stable. 0.1 is conservative.
        regularization : small constant to prevent division by zero (δ)
        """
        self.filter_length = filter_length
        self.mu = step_size
        self.delta = regularization

        # ── Filter coefficients ──
        self._weights = np.zeros(filter_length, dtype=np.float64)

        # ── Reference signal buffer (double buffered to avoid allocations) ──
        # By keeping a buffer of size 2*L, we can always take a contiguous slice of size L.
        self._ref_buffer = np.zeros(filter_length * 2, dtype=np.float64)
        self._idx = filter_length

    def process_sample(self, enhanced: float, reference: float) -> float:
        """Process a single sample.

        Parameters
        ----------
        enhanced  : one sample from the neural net output
        reference : one sample from the reference mic

        Returns
        -------
        cleaned : the enhanced sample with predicted noise subtracted

        NLMS update equations:
            y[n] = w^T · x[n]              (predicted noise)
            e[n] = enhanced[n] - y[n]       (error = cleaned output)
            w[n+1] = w[n] + μ · e[n] · x[n] / (x[n]^T · x[n] + δ)
        """
        self._idx -= 1
        if self._idx < 0:
            # Wrap around: copy the newest L-1 samples to the end of the buffer
            self._ref_buffer[self.filter_length : 2 * self.filter_length - 1] = \
                self._ref_buffer[0 : self.filter_length - 1]
            self._idx = self.filter_length - 1

        self._ref_buffer[self._idx] = float(reference)
        
        # Contiguous view of the last L samples (newest first)
        x = self._ref_buffer[self._idx : self._idx + self.filter_length]

        # Predict noise at primary mic
        predicted_noise = np.dot(self._weights, x)

        # Error (= cleaned output)
        error = float(enhanced) - predicted_noise

        # Adapt weights
        norm = np.dot(x, x) + self.delta
        self._weights += self.mu * error * x / norm

        return float(error)

    def process_frame(
        self,
        enhanced: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Process a frame of samples.

        Parameters
        ----------
        enhanced  : float32 array of shape (hop,) — neural net output
        reference : float32 array of shape (hop,) — reference mic input

        Returns
        -------
        cleaned : float32 array of shape (hop,)
        """
        n = len(enhanced)
        output = np.zeros(n, dtype=np.float32)

        for i in range(n):
            output[i] = self.process_sample(enhanced[i], reference[i])

        return output

    def reset(self) -> None:
        """Reset filter coefficients and buffer."""
        self._weights[:] = 0.0
        self._ref_buffer[:] = 0.0

    @property
    def converged(self) -> bool:
        """Rough check: has the filter learned something non-trivial?"""
        return float(np.max(np.abs(self._weights))) > 1e-4
