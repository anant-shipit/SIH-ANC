"""
test_mixer.py — Verify that mixer.py produces correct SNRs and protects against clipping.

Tests:
    1. SNR accuracy within ±0.5 dB for a range of target SNRs.
    2. Same-divisor clipping protection: no sample exceeds [-1, 1].
    3. Impulsive mode places noise without tiling.
    4. Silent signals don't produce NaN or Inf.
    5. Noise looping works for short noise files.
    6. measure_snr agrees with the requested snr_db.
"""
from __future__ import annotations

import numpy as np
import pytest

from sih26052.data.mixer import (
    MixerConfig,
    mix_at_snr,
    measure_snr,
    _rms,
    _fit_noise_to_length,
    _clip_protect,
    _scale_noise_to_snr,
)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSNRAccuracy:
    """The actual SNR of mixed output should be within ±0.5 dB of target."""

    @pytest.mark.parametrize("target_snr", [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    def test_snr_accuracy(self, target_snr: float, make_sine, make_noise):
        clean = make_sine(duration_s=3.0)
        noise = make_noise(duration_s=3.0)

        noisy, clean_out = mix_at_snr(clean, noise, target_snr)
        actual_snr = measure_snr(clean_out, noisy)

        assert abs(actual_snr - target_snr) < 0.5, (
            f"Target SNR={target_snr} dB, got {actual_snr:.2f} dB "
            f"(error={abs(actual_snr - target_snr):.2f} dB)"
        )


class TestClippingProtection:
    """No sample in the output should exceed [-0.99, 0.99]."""

    def test_no_clipping_at_low_snr(self, make_sine, make_noise):
        """At −5 dB SNR, noise is loud → must clip-protect."""
        clean = make_sine(duration_s=2.0)
        noise = make_noise(duration_s=2.0) * 5.0  # loud noise

        noisy, clean_out = mix_at_snr(clean, noise, -5.0)

        assert np.max(np.abs(noisy)) <= 1.0, "noisy exceeds [-1, 1]"
        assert np.max(np.abs(clean_out)) <= 1.0, "clean exceeds [-1, 1]"

    def test_same_divisor_invariant(self, make_sine, make_noise):
        """Both clean and noisy must be scaled by the same factor.

        Verify by checking that the SNR is preserved even after
        clip protection.
        """
        clean = make_sine(duration_s=2.0) * 0.8
        noise = make_noise(duration_s=2.0) * 3.0
        target_snr = 5.0

        noisy, clean_out = mix_at_snr(clean, noise, target_snr)
        actual_snr = measure_snr(clean_out, noisy)

        # Even with aggressive clipping, SNR should be preserved
        assert abs(actual_snr - target_snr) < 0.5

    def test_already_quiet_signals(self, make_sine, make_noise):
        """Signals well below 1.0 should not be rescaled."""
        clean = make_sine(duration_s=1.0) * 0.1
        noise = make_noise(duration_s=1.0) * 0.01

        noisy, clean_out = mix_at_snr(clean, noise, 10.0)

        # Peak should be roughly the same as input (no inflation)
        assert np.max(np.abs(noisy)) < 0.5


class TestImpulsiveMode:
    """Impulsive noise should be placed once, not tiled."""

    def test_impulse_not_tiled(self, make_sine):
        """A short impulse placed in a long buffer should occupy one contiguous region."""
        rng = np.random.default_rng(42)
        target_len = 48000  # 3 seconds at 16kHz
        impulse = np.ones(1600, dtype=np.float32)  # 100ms of ones

        placed = _fit_noise_to_length(impulse, target_len, impulsive=True, rng=rng)

        # The placed buffer should have exactly one contiguous block of non-zero values
        active = np.abs(placed) > 1e-8
        transitions = np.sum(np.diff(active.astype(int)) == 1)
        assert transitions <= 1, (
            f"Impulsive noise has {transitions} onset transitions — looks tiled"
        )

        # The active region should be exactly the impulse length
        assert np.sum(active) == len(impulse), (
            f"Expected {len(impulse)} active samples, got {np.sum(active)}"
        )

    def test_impulse_output_correct_length(self, make_sine, make_noise):
        """Output must match clean signal length."""
        clean = make_sine(duration_s=2.0)
        impulse = make_noise(duration_s=0.5)

        noisy, clean_out = mix_at_snr(clean, impulse, 5.0, impulsive=True)
        assert len(noisy) == len(clean)
        assert len(clean_out) == len(clean)


class TestEdgeCases:
    def test_silent_clean(self, make_noise):
        """Silent clean signal should not produce NaN."""
        clean = np.zeros(16000, dtype=np.float32)
        noise = make_noise(duration_s=1.0)

        noisy, clean_out = mix_at_snr(clean, noise, 5.0)

        assert np.all(np.isfinite(noisy))
        assert np.all(np.isfinite(clean_out))

    def test_silent_noise(self, make_sine):
        """Silent noise should not produce NaN."""
        clean = make_sine(duration_s=1.0)
        noise = np.zeros(16000, dtype=np.float32)

        noisy, clean_out = mix_at_snr(clean, noise, 5.0)

        assert np.all(np.isfinite(noisy))
        assert np.all(np.isfinite(clean_out))

    def test_noise_shorter_than_clean(self, make_sine, make_noise):
        """Short noise should be looped to match clean length."""
        clean = make_sine(duration_s=3.0)
        noise = make_noise(duration_s=0.5)  # much shorter

        noisy, clean_out = mix_at_snr(clean, noise, 10.0)
        assert len(noisy) == len(clean)

    def test_noise_longer_than_clean(self, make_sine, make_noise):
        """Long noise should be cropped to match clean length."""
        clean = make_sine(duration_s=1.0)
        noise = make_noise(duration_s=5.0)  # much longer

        noisy, clean_out = mix_at_snr(clean, noise, 10.0)
        assert len(noisy) == len(clean)


class TestFitNoiseToLength:
    def test_exact_length(self, make_noise):
        noise = make_noise(duration_s=1.0)
        result = _fit_noise_to_length(noise, len(noise))
        assert len(result) == len(noise)

    def test_impulsive_placement(self):
        """Impulsive mode should produce a buffer with mostly zeros."""
        noise = np.ones(100, dtype=np.float32)
        result = _fit_noise_to_length(noise, 10000, impulsive=True)
        assert len(result) == 10000
        # Most of the buffer should be zero
        zero_fraction = np.mean(result == 0.0)
        assert zero_fraction > 0.9, f"Only {zero_fraction*100:.0f}% zeros"


class TestRMS:
    def test_sine_rms(self):
        """RMS of a sine wave with amplitude A is A/√2."""
        amp = 0.5
        t = np.arange(16000, dtype=np.float32) / 16000
        sine = amp * np.sin(2 * np.pi * 440 * t)
        expected = amp / np.sqrt(2)
        assert abs(_rms(sine) - expected) < 0.01

    def test_silence_rms(self):
        assert _rms(np.zeros(100, dtype=np.float32)) == 0.0
