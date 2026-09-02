"""
impulse_gate.py — Transient detector + hold-and-release attenuator.

Catches impulsive transients (gunshots, door slams, hammer strikes) that
the neural net didn't fully suppress, and applies extra attenuation
during and briefly after the transient.

How it works:
    1. Compute frame energy (sum of squared samples).
    2. Compare to a rolling median of recent frame energies.
    3. If the ratio exceeds a threshold → GATE FIRES.
    4. While fired: apply attenuation to the model output.
    5. After the transient passes: smooth release ramp (~150ms).

Why not just let the model handle transients?
    The model is trained mostly on stationary noise.  Impulsive noise
    has very different statistics — short, high-energy bursts.  The
    model can learn to handle them with fine-tuning (Phase 6), but the
    gate provides immediate extra suppression as a safety net.

States exposed to dashboard:
    "idle"      — no transient detected
    "fired"     — transient detected, full attenuation applied
    "releasing" — transient over, attenuation ramping back to 1.0

NO torch imports — only numpy.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ImpulseGate:
    """Per-frame transient detector with smooth release.

    Usage:
        gate = ImpulseGate(sr=16000, hop=256)

        # In audio callback:
        enhanced = gate.process(enhanced_samples)
        state = gate.state  # "idle" | "fired" | "releasing"
    """

    def __init__(
        self,
        sr: int = 16000,
        hop: int = 256,
        threshold_ratio: float = 10.0,
        hold_ms: float = 50.0,
        release_ms: float = 150.0,
        attenuation_db: float = -20.0,
        history_frames: int = 30,
    ):
        """
        Parameters
        ----------
        sr               : sample rate
        hop              : hop size (frame length in samples)
        threshold_ratio  : energy must exceed median × this to fire
        hold_ms          : hold full attenuation for this long after spike
        release_ms       : release ramp duration after hold
        attenuation_db   : how much to attenuate when fired (dB)
        history_frames   : how many past frames for rolling median
        """
        self.sr = sr
        self.hop = hop
        self.threshold_ratio = threshold_ratio
        self.attenuation_linear = 10.0 ** (attenuation_db / 20.0)

        # Timing in frames
        self.hold_frames = int(hold_ms * sr / (hop * 1000.0))
        self.release_frames = int(release_ms * sr / (hop * 1000.0))

        # ── Rolling energy history (ring buffer) ──
        self.history_frames = history_frames
        self._energy_history = np.zeros(history_frames, dtype=np.float64)
        self._history_idx = 0
        self._history_filled = False

        # ── Gate state machine ──
        self._state = "idle"  # "idle" | "fired" | "releasing"
        self._hold_counter = 0
        self._release_counter = 0
        self._current_gain = 1.0  # 1.0 = no attenuation

    @property
    def state(self) -> str:
        """Current gate state: 'idle', 'fired', or 'releasing'."""
        return self._state

    @property
    def current_gain(self) -> float:
        """Current attenuation factor (1.0 = no attenuation)."""
        return self._current_gain

    def _frame_energy(self, samples: np.ndarray) -> float:
        """Compute frame energy (sum of squares)."""
        return float(np.sum(samples.astype(np.float64) ** 2))

    def _update_history(self, energy: float) -> float:
        """Add energy to history and return rolling median."""
        self._energy_history[self._history_idx] = energy
        self._history_idx = (self._history_idx + 1) % self.history_frames
        if not self._history_filled and self._history_idx == 0:
            self._history_filled = True

        if self._history_filled:
            return float(np.median(self._energy_history))
        else:
            # Not enough history yet — use median of what we have
            valid = self._energy_history[:self._history_idx]
            if len(valid) < 3:
                return energy  # not enough data to compare
            return float(np.median(valid))

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process a frame of enhanced audio, applying transient gating.

        Parameters
        ----------
        samples : float32 array of shape (hop,)

        Returns
        -------
        gated : float32 array of shape (hop,) — possibly attenuated
        """
        energy = self._frame_energy(samples)
        median_energy = self._update_history(energy)

        # ── State machine ──
        if self._state == "idle":
            # Check for transient
            if median_energy > 1e-10 and energy > median_energy * self.threshold_ratio:
                self._state = "fired"
                self._hold_counter = self.hold_frames
                self._current_gain = self.attenuation_linear
                logger.debug("Impulse gate FIRED (energy=%.4f, median=%.4f)", energy, median_energy)

        elif self._state == "fired":
            self._hold_counter -= 1
            self._current_gain = self.attenuation_linear
            if self._hold_counter <= 0:
                self._state = "releasing"
                self._release_counter = self.release_frames

        elif self._state == "releasing":
            self._release_counter -= 1
            # Smooth ramp from attenuation_linear back to 1.0
            t = 1.0 - (self._release_counter / max(self.release_frames, 1))
            self._current_gain = self.attenuation_linear + t * (1.0 - self.attenuation_linear)

            if self._release_counter <= 0:
                self._state = "idle"
                self._current_gain = 1.0
                logger.debug("Impulse gate released → idle")

        # ── Apply gain ──
        if abs(self._current_gain - 1.0) < 1e-6:
            return samples  # fast path: no attenuation
        else:
            return (samples * self._current_gain).astype(np.float32)

    def reset(self) -> None:
        """Reset gate to idle state."""
        self._state = "idle"
        self._hold_counter = 0
        self._release_counter = 0
        self._current_gain = 1.0
        self._energy_history[:] = 0.0
        self._history_idx = 0
        self._history_filled = False
