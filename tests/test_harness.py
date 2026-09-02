"""
test_harness.py — Verify the evaluation harness identity test and metrics.

Tests:
    1. Identity test: clean vs clean → PESQ ≈ 4.5, SI-SNR ≈ 100+ dB
    2. SI-SNR of identical signals is very high.
    3. SI-SNR of uncorrelated signals is near 0.
    4. Alignment finds a known offset.
"""
from __future__ import annotations

import numpy as np
import pytest

from sih26052.eval.metrics import si_snr, compute_all_metrics
from sih26052.eval.alignment import find_delay, align_signals


# ── Helpers ────────────────────────────────────────────────────────────────

def make_speech_like(duration_s: float = 2.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Generate a speech-like signal (modulated noise) for metric tests."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    # Mix of tones to simulate speech formants
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
        + 0.05 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    # Amplitude modulation to simulate speech envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return (signal * envelope).astype(np.float32)


# ── SI-SNR tests ───────────────────────────────────────────────────────────

class TestSISNR:
    def test_identical_signals(self):
        """SI-SNR of identical signals should be very high (>80 dB)."""
        signal = make_speech_like()
        snr = si_snr(signal, signal)
        assert snr > 80.0, f"Expected SI-SNR > 80 dB for identical signals, got {snr:.1f}"

    def test_uncorrelated_signals(self):
        """SI-SNR of uncorrelated signals should be low (strongly negative or near 0)."""
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(99)
        ref = rng1.standard_normal(32000).astype(np.float32)
        est = rng2.standard_normal(32000).astype(np.float32)
        snr = si_snr(ref, est)
        # For truly uncorrelated signals, SI-SNR should be very negative
        # (estimate has no projection onto reference)
        assert snr < 5.0, f"Expected low SI-SNR for uncorrelated, got {snr:.1f}"

    def test_known_snr(self):
        """Adding noise at known power should give predictable SI-SNR."""
        rng = np.random.default_rng(42)
        ref = make_speech_like(duration_s=3.0)
        noise = rng.standard_normal(len(ref)).astype(np.float32) * 0.01
        est = ref + noise
        snr = si_snr(ref, est)
        # Noise is very quiet relative to signal, SI-SNR should be high
        assert snr > 20.0, f"Expected SI-SNR > 20 with tiny noise, got {snr:.1f}"

    def test_scale_invariant(self):
        """SI-SNR should not change if estimate is scaled."""
        ref = make_speech_like()
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(len(ref)).astype(np.float32) * 0.1
        est = ref + noise

        snr1 = si_snr(ref, est)
        snr2 = si_snr(ref, est * 2.0)  # double the volume

        assert abs(snr1 - snr2) < 0.5, (
            f"SI-SNR changed from {snr1:.2f} to {snr2:.2f} with 2× scaling"
        )


# ── Alignment tests ───────────────────────────────────────────────────────

class TestAlignment:
    def test_known_offset_positive(self):
        """Detect a known positive delay."""
        ref = make_speech_like(duration_s=3.0)
        delay_samples = 50
        # Simulate a delayed estimate: pad with zeros at the start
        est = np.concatenate([np.zeros(delay_samples, dtype=np.float32), ref])

        detected = find_delay(ref, est)
        assert abs(detected - (-delay_samples)) < 5, (
            f"Expected delay ≈ {-delay_samples}, got {detected}"
        )

    def test_known_offset_negative(self):
        """Detect a known negative delay (estimate is ahead)."""
        ref = make_speech_like(duration_s=3.0)
        delay_samples = 30
        # Estimate starts early: skip first N samples of ref
        est = ref[delay_samples:]

        detected = find_delay(ref, est)
        assert abs(detected - delay_samples) < 5, (
            f"Expected delay ≈ {delay_samples}, got {detected}"
        )

    def test_no_delay(self):
        """No delay should be detected for aligned signals."""
        ref = make_speech_like(duration_s=2.0)
        delay = find_delay(ref, ref.copy())
        assert abs(delay) < 3, f"Expected ~0 delay, got {delay}"

    def test_align_signals_produces_same_length(self):
        """align_signals should return two arrays of the same length."""
        ref = make_speech_like(duration_s=2.0)
        est = np.concatenate([np.zeros(20, dtype=np.float32), ref])

        ref_a, est_a, delay = align_signals(ref, est)
        assert len(ref_a) == len(est_a)


# ── Identity test (PESQ gate) ─────────────────────────────────────────────

def _pesq_available() -> bool:
    """Check if pesq library is importable."""
    try:
        import pesq  # noqa: F401
        return True
    except ImportError:
        return False


class TestIdentityGate:
    """The identity test: clean vs clean should yield near-perfect scores.

    This is the Phase 2 gate — if this fails, all downstream evaluations
    are suspect because the metric pipeline itself is broken.
    """

    def test_si_snr_identity(self):
        """clean→clean should give SI-SNR > 80 dB."""
        signal = make_speech_like(duration_s=3.0)
        result = compute_all_metrics(signal, signal)
        assert result.si_snr > 80.0, f"Identity SI-SNR = {result.si_snr:.1f} dB"

    @pytest.mark.skipif(
        not _pesq_available(),
        reason="pesq library not installed"
    )
    def test_pesq_identity(self):
        """clean→clean PESQ should be ≈ 4.5."""
        signal = make_speech_like(duration_s=3.0)
        result = compute_all_metrics(signal, signal)
        assert result.pesq is not None, "PESQ returned None for identity test"
        assert result.pesq > 4.0, f"Identity PESQ = {result.pesq:.2f}, expected > 4.0"

