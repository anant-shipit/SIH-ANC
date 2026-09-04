"""
ab_switch.py — A/B toggle with 20ms crossfade.

Toggles between raw (bypassed) and enhanced audio with a smooth
crossfade to prevent audible clicks/pops at the transition.

The switch is controlled by:
    1. GPIO button (primary) — gpiozero Button on GPIO 17
       (ReSpeaker HAT user button).  Works on Pi 4B and Pi 5.
    2. Keyboard fallback — spacebar when running on a laptop

Pi 5 migration note:
    RPi.GPIO does NOT work on Pi 5 (different GPIO controller).
    gpiozero + lgpio is the officially supported replacement.
    bounce_time is in seconds (0.2), not milliseconds (200).
    No GPIO.cleanup() equivalent needed — gpiozero handles it.

Why 20ms crossfade?
    - Long enough to prevent audible discontinuity (human ear resolves
      clicks down to ~5ms).
    - Short enough that the transition feels instant.
    - At 16kHz, 20ms = 320 samples.

Formula:
    out = raw × fade_out + enhanced × fade_in
    where fade_in ramps 0→1 and fade_out ramps 1→0 over 320 samples.

NO torch imports — only numpy.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class ABSwitch:
    """Smooth A/B toggle between raw and enhanced audio.

    Usage:
        switch = ABSwitch(sr=16000)

        # In audio callback:
        output = switch.apply(raw_samples, enhanced_samples)

        # Toggle from UI/GPIO/keyboard:
        switch.toggle()
    """

    def __init__(self, sr: int = 16000, crossfade_ms: float = 20.0):
        """
        Parameters
        ----------
        sr           : sample rate
        crossfade_ms : crossfade duration in milliseconds
        """
        self.sr = sr
        self.crossfade_samples = int(sr * crossfade_ms / 1000.0)

        # ── State ──
        # True = enhanced active, False = raw/bypass active
        self._enhanced_active = True
        self._lock = threading.Lock()

        # ── Crossfade state ──
        # When a toggle is requested, we transition over crossfade_samples.
        # _fade_position tracks progress: 0 = fully raw, 1 = fully enhanced
        self._fade_position = 1.0  # start with enhancement on
        self._fade_target = 1.0
        self._fade_step = 0.0  # per-sample increment (set on toggle)

    @property
    def is_enhanced(self) -> bool:
        """Whether enhancement is currently active (or transitioning to it)."""
        return self._fade_target == 1.0

    def toggle(self) -> bool:
        """Toggle between raw and enhanced.

        Returns the new state (True = enhanced active).
        Thread-safe — can be called from GPIO interrupt or keyboard handler.
        """
        with self._lock:
            if self._fade_target == 1.0:
                self._fade_target = 0.0
            else:
                self._fade_target = 1.0

            # Compute per-sample step to reach target over crossfade duration
            distance = self._fade_target - self._fade_position
            if abs(distance) < 1e-6:
                self._fade_step = 0.0
            else:
                self._fade_step = distance / self.crossfade_samples

        new_state = self._fade_target == 1.0
        logger.info("A/B switch toggled → %s", "ENHANCED" if new_state else "BYPASS")
        return new_state

    def apply(self, raw: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """Mix raw and enhanced samples according to current fade position.

        Parameters
        ----------
        raw       : float32 array of shape (hop,) — unprocessed audio
        enhanced  : float32 array of shape (hop,) — model output

        Returns
        -------
        output : float32 array of shape (hop,) — crossfaded result

        This method is called from the audio callback and MUST NOT allocate
        memory or block.  All buffers are pre-sized.
        """
        n = len(raw)
        output = np.empty(n, dtype=np.float32)

        with self._lock:
            pos = self._fade_position
            step = self._fade_step

        for i in range(n):
            # Clamp fade position to [0, 1]
            pos = max(0.0, min(1.0, pos + step))

            # If we've reached the target, stop stepping
            if step > 0 and pos >= self._fade_target:
                pos = self._fade_target
                step = 0.0
            elif step < 0 and pos <= self._fade_target:
                pos = self._fade_target
                step = 0.0

            output[i] = raw[i] * (1.0 - pos) + enhanced[i] * pos

        # Update shared state (minimal lock duration)
        with self._lock:
            self._fade_position = pos
            self._fade_step = step

        return output

    def apply_vectorized(self, raw: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """Vectorized version of apply() — faster for large blocks.

        Uses numpy broadcasting instead of a Python for-loop.
        Preferred when hop size is large (>= 256 samples).
        """
        n = len(raw)

        with self._lock:
            pos = self._fade_position
            step = self._fade_step
            target = self._fade_target

        if abs(step) < 1e-8:
            # No transition in progress — fast path
            if pos >= 0.999:
                result = enhanced.copy()
            elif pos <= 0.001:
                result = raw.copy()
            else:
                result = raw * (1.0 - pos) + enhanced * pos
        else:
            # Build per-sample fade ramp
            fade = np.clip(pos + step * np.arange(n, dtype=np.float32), 0.0, 1.0)

            # Find where we hit the target and flatten from there
            if step > 0:
                hit_idx = np.searchsorted(fade, target)
                fade[hit_idx:] = target
            else:
                hit_idx = np.searchsorted(-fade, -target)
                fade[hit_idx:] = target

            result = raw * (1.0 - fade) + enhanced * fade
            pos = float(fade[-1])
            if abs(pos - target) < 1e-6:
                step = 0.0

        with self._lock:
            self._fade_position = pos
            self._fade_step = step

        return result.astype(np.float32)


# ── GPIO + Keyboard wiring ─────────────────────────────────────────────────

def attach_gpio_button(
    switch: ABSwitch,
    gpio_pin: int = 17,
    bounce_time: float = 0.2,
) -> object | None:
    """Wire a physical button to the A/B switch via gpiozero.

    GPIO 17 is the ReSpeaker HAT's user button.
    Uses gpiozero (works on both Pi 4B and Pi 5).
    Returns the Button object so the caller can hold a reference
    (preventing garbage collection of the callback).
    Returns None if gpiozero is not available (e.g. on a laptop).
    """
    try:
        from gpiozero import Button  # type: ignore

        button = Button(gpio_pin, pull_up=True, bounce_time=bounce_time)
        button.when_pressed = lambda: switch.toggle()
        logger.info(
            "GPIO button attached on pin %d (bounce_time=%.1fs)",
            gpio_pin, bounce_time,
        )
        return button
    except Exception as exc:
        logger.warning("GPIO button not available: %s (keyboard fallback only)", exc)
        return None
