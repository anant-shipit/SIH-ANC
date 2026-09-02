"""
augment.py — On-the-fly audio augmentations for training diversity.

These transforms widen the variety of training examples *without* mixing
new noise.  They simulate real-world recording variation:

    - Random gain:     models don't memorise a single volume level.
    - Band-limiting:   simulates cheap mics that roll off above 4 kHz.
    - Mild clipping:   simulates ADC saturation in the field.
    - Reverb:          simulates enclosed spaces (vehicles, bunkers).

All augmentations are applied *before* mixing (to the clean signal) so
that the (noisy, clean) pair stays aligned.  The mixer produces the SNR
relationship; augment widens the clean signal variety.

Design note:
    These are intentionally simple numpy implementations.  We don't
    need GPU-accelerated augmentation for 48K-param models on a 4060.

Usage:
    from sih26052.data.augment import AugmentPipeline

    aug = AugmentPipeline(prob=0.5, seed=42)
    augmented = aug(clean_waveform, sr=16000)
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── Individual augmentation functions ──────────────────────────────────────

def random_gain(
    audio: np.ndarray,
    rng: np.random.Generator,
    min_db: float = -6.0,
    max_db: float = 6.0,
) -> np.ndarray:
    """Apply a random gain between [min_db, max_db].

    Why:
        Speech recordings vary in volume depending on mic distance,
        preamp settings, and talker loudness.  Training at a fixed
        volume makes the model fragile to level changes.
    """
    gain_db = rng.uniform(min_db, max_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return audio * gain_linear


def band_limit(
    audio: np.ndarray,
    rng: np.random.Generator,
    sr: int = 16000,
    min_cutoff_hz: float = 3500.0,
    max_cutoff_hz: float = 7500.0,
) -> np.ndarray:
    """Apply a random low-pass filter to simulate cheap microphones.

    Why:
        Field mics (especially body-worn ones) often have poor high-frequency
        response.  This teaches the model not to hallucinate HF content.

    Implementation:
        Simple FIR via scipy's firwin.  Order 63 is plenty for this purpose.
    """
    from scipy.signal import firwin, lfilter

    cutoff = rng.uniform(min_cutoff_hz, max_cutoff_hz)
    nyquist = sr / 2.0
    if cutoff >= nyquist:
        return audio  # nothing to cut

    normalized_cutoff = cutoff / nyquist
    # 63-tap FIR — good tradeoff between sharpness and artifacts
    b = firwin(63, normalized_cutoff, window="hamming")
    return lfilter(b, 1.0, audio).astype(np.float32)


def mild_clipping(
    audio: np.ndarray,
    rng: np.random.Generator,
    min_threshold: float = 0.5,
    max_threshold: float = 0.95,
) -> np.ndarray:
    """Soft-clip the signal at a random threshold.

    Why:
        ADC saturation in field equipment clips loud transients.
        The model needs to have seen clipped inputs to handle them.

    We use tanh soft-clipping rather than hard clip because tanh produces
    smoother harmonics that are more realistic.
    """
    threshold = rng.uniform(min_threshold, max_threshold)
    # Normalise peak to 1, apply tanh, scale back
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    normalised = audio / peak
    # Scale so that `threshold` of the dynamic range is in the tanh linear zone
    scale = 1.0 / threshold
    clipped = np.tanh(normalised * scale)
    # Restore original peak (approximately)
    return (clipped * peak * threshold).astype(np.float32)


def add_reverb(
    audio: np.ndarray,
    rng: np.random.Generator,
    sr: int = 16000,
    min_rt60_ms: float = 100.0,
    max_rt60_ms: float = 500.0,
    wet_ratio: float = 0.3,
) -> np.ndarray:
    """Add simple synthetic reverb using an exponential decay impulse response.

    Why:
        Defense environments include vehicles, bunkers, and corridors with
        significant reverberation.  This augmentation teaches the model
        to handle reverberant speech without confusing it with noise.

    This is NOT a physically accurate room simulator.  It's a fast
    approximation: white noise shaped by an exponential decay envelope,
    convolved with the signal.  Good enough for augmentation purposes.
    """
    rt60_ms = rng.uniform(min_rt60_ms, max_rt60_ms)
    rt60_samples = int(rt60_ms * sr / 1000.0)

    # Generate impulse response: exponential decay from 1 to ~0.001 (−60 dB)
    t = np.arange(rt60_samples, dtype=np.float32)
    decay = np.exp(-6.908 * t / rt60_samples)  # 6.908 = ln(1000) ≈ 60 dB decay
    ir = rng.standard_normal(rt60_samples).astype(np.float32) * decay
    ir[0] = 1.0  # direct path

    # Normalise IR energy
    ir /= np.sqrt(np.sum(ir ** 2) + 1e-8)

    # Convolve
    from scipy.signal import fftconvolve
    reverbed = fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)

    # Wet/dry mix
    out = (1.0 - wet_ratio) * audio + wet_ratio * reverbed
    return out.astype(np.float32)


# ── Pipeline ───────────────────────────────────────────────────────────────

class AugmentPipeline:
    """Apply a random subset of augmentations to an audio signal.

    Each augmentation is applied independently with probability *prob*.
    Set prob=0.0 to disable all augmentations (e.g. for validation).

    Parameters
    ----------
    prob : float
        Independent probability of applying each augmentation.
    seed : int | None
        RNG seed for reproducibility.
    enable_reverb : bool
        Whether to include reverb (expensive, can be disabled for speed).
    """

    def __init__(
        self,
        prob: float = 0.5,
        seed: int | None = None,
        enable_reverb: bool = True,
    ):
        self.prob = prob
        self.rng = np.random.default_rng(seed)
        self.enable_reverb = enable_reverb

        # Build the augmentation list.  Order matters slightly:
        # gain first (affects clipping threshold), reverb last (expensive).
        self._augmentations: list[tuple[str, Callable]] = [
            ("gain", lambda a, r, sr: random_gain(a, r)),
            ("band_limit", lambda a, r, sr: band_limit(a, r, sr)),
            ("clipping", lambda a, r, sr: mild_clipping(a, r)),
        ]
        if enable_reverb:
            self._augmentations.append(
                ("reverb", lambda a, r, sr: add_reverb(a, r, sr)),
            )

    def __call__(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply random augmentations to *audio*.

        Returns the augmented waveform (same length, float32).
        """
        out = audio.astype(np.float32, copy=True)
        applied = []

        for name, fn in self._augmentations:
            if self.rng.random() < self.prob:
                out = fn(out, self.rng, sr)
                applied.append(name)

        if applied:
            logger.debug("Applied augmentations: %s", ", ".join(applied))

        return out

    def __repr__(self) -> str:
        return (
            f"AugmentPipeline(prob={self.prob}, "
            f"augments={[n for n, _ in self._augmentations]})"
        )
