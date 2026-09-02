"""
mixer.py — Mix clean speech with noise at a random SNR.

This is the core data synthesis engine.  It produces (noisy, clean) pairs
that the model trains on.

Design decisions:
    1. **Same-divisor clipping protection.**
       We find the divisor that prevents *both* the clean and noisy signals
       from clipping, and apply it to both.  This preserves the exact SNR
       relationship — if you scaled them independently, the SNR would drift.

    2. **Impulsive noise is NOT tiled.**
       Short transients (gunshots, door slams) are placed at random
       positions within the speech, not repeated.  Tiling would create
       unrealistic periodic patterns that the model memorises instead of
       generalising.

    3. **SNR range −5 to +20 dB.**
       −5 dB is almost unintelligibly noisy (battlefield close to source).
       +20 dB is near-clean (distant engine rumble).
       Uniform random within this range.

Usage:
    from sih26052.data.mixer import mix_at_snr, MixerConfig, mix_pair

    noisy, clean = mix_at_snr(clean_wav, noise_wav, snr_db=5.0)
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────

class MixResult(NamedTuple):
    """Output of a single mix operation."""
    noisy: np.ndarray        # float32, same length as clean
    clean: np.ndarray        # float32, the reference signal
    snr_db: float            # actual SNR used
    noise_class: str         # label for the noise source (e.g. "gunshot")


@dataclasses.dataclass
class MixerConfig:
    """Controls the mixing behaviour."""
    snr_range: tuple[float, float] = (-5.0, 20.0)   # uniform draw (dB)
    target_sr: int = 16000
    # If True, impulsive noises are placed once at a random offset.
    # If False, noise is looped/truncated to match speech length.
    impulsive_mode: bool = False
    # RNG seed for reproducibility (None = random each call)
    seed: int | None = None


# ── Core mixing ────────────────────────────────────────────────────────────

def _rms(x: np.ndarray) -> float:
    """Root mean square.  Returns 0.0 for silence."""
    val = np.sqrt(np.mean(x ** 2))
    return float(val) if np.isfinite(val) else 0.0


def _scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale *noise* so that SNR(clean, noise) = snr_db.

    The clean signal is NOT modified — only the noise is rescaled.
    If either signal is silent, return noise unchanged (nothing to scale).
    """
    rms_clean = _rms(clean)
    rms_noise = _rms(noise)

    if rms_clean < 1e-8 or rms_noise < 1e-8:
        return noise

    # SNR = 20·log10(rms_clean / rms_noise)
    # ⇒ rms_noise_target = rms_clean / 10^(snr_db/20)
    target_rms_noise = rms_clean / (10.0 ** (snr_db / 20.0))
    scale = target_rms_noise / rms_noise
    return noise * scale


def _fit_noise_to_length(
    noise: np.ndarray,
    target_len: int,
    *,
    impulsive: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Make *noise* exactly *target_len* samples long.

    For stationary noise:
        If shorter than target → loop (tile + truncate).
        If longer → random crop.

    For impulsive noise (impulsive=True):
        Place the noise at a random offset within a silence buffer.
        No tiling — a single occurrence is more realistic.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_len = len(noise)

    if impulsive:
        # Single occurrence at random offset inside a zero buffer
        out = np.zeros(target_len, dtype=np.float32)
        if n_len >= target_len:
            # Noise is longer than speech — take a random chunk
            start = rng.integers(0, n_len - target_len + 1)
            out[:] = noise[start:start + target_len]
        else:
            # Place at random position
            max_start = target_len - n_len
            start = rng.integers(0, max_start + 1)
            out[start:start + n_len] = noise
        return out

    # ── Stationary noise: loop or crop ──
    if n_len >= target_len:
        start = rng.integers(0, n_len - target_len + 1)
        return noise[start:start + target_len].copy()
    else:
        # Loop the noise until it's long enough, then truncate
        reps = (target_len // n_len) + 1
        looped = np.tile(noise, reps)
        return looped[:target_len].copy()


def _clip_protect(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-divisor clipping protection.

    Find the single scalar that brings the louder of |noisy| and |clean|
    to exactly 0.99, then apply it to BOTH signals.  This preserves the
    SNR exactly.

    Why 0.99 and not 1.0?
        Leaves 0.01 headroom for floating-point rounding so the resulting
        WAV file never hits full-scale (which some DACs convert to a click).
    """
    peak = max(np.max(np.abs(clean)), np.max(np.abs(noisy)))
    if peak < 1e-8:
        return clean, noisy

    headroom = 0.99
    if peak > headroom:
        scale = headroom / peak
        return clean * scale, noisy * scale
    return clean, noisy


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    impulsive: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a clean speech signal with noise at a specified SNR.

    Parameters
    ----------
    clean     : 1-D float32 waveform (16 kHz mono)
    noise     : 1-D float32 waveform (any length, same sample rate)
    snr_db    : desired signal-to-noise ratio in dB
    impulsive : if True, noise is placed once instead of looped
    rng       : numpy Generator for reproducibility

    Returns
    -------
    noisy  : float32 waveform, same length as clean
    clean  : float32 waveform (may be rescaled for clip protection)

    The actual SNR of the output should be within ±0.5 dB of snr_db
    unless the clean signal is near-silent.
    """
    if rng is None:
        rng = np.random.default_rng()

    clean = clean.astype(np.float32, copy=True)
    noise = noise.astype(np.float32, copy=True)

    # Step 1: fit noise to clean's length
    noise_fitted = _fit_noise_to_length(
        noise, len(clean), impulsive=impulsive, rng=rng,
    )

    # Step 2: scale noise to achieve target SNR
    noise_scaled = _scale_noise_to_snr(clean, noise_fitted, snr_db)

    # Step 3: sum
    noisy = clean + noise_scaled

    # Step 4: same-divisor clipping protection
    clean, noisy = _clip_protect(clean, noisy)

    return noisy, clean


def mix_pair(
    clean_path: Path,
    noise_path: Path,
    config: MixerConfig | None = None,
    noise_class: str = "unknown",
) -> MixResult:
    """Load files from disk and mix them.  Convenience wrapper around mix_at_snr.

    Parameters
    ----------
    clean_path   : path to a pre-processed 16 kHz mono WAV
    noise_path   : path to a pre-processed noise WAV
    config       : MixerConfig (defaults to uniform SNR in [−5, +20] dB)
    noise_class  : human label like "gunshot", "engine", "stationary"

    Returns
    -------
    MixResult with (noisy, clean, snr_db, noise_class)
    """
    if config is None:
        config = MixerConfig()

    rng = np.random.default_rng(config.seed)

    # ── Load ──
    clean, sr_c = sf.read(str(clean_path), dtype="float32")
    noise, sr_n = sf.read(str(noise_path), dtype="float32")

    assert sr_c == config.target_sr, (
        f"Clean file {clean_path} has sr={sr_c}, expected {config.target_sr}. "
        "Run preprocess.py first."
    )
    assert sr_n == config.target_sr, (
        f"Noise file {noise_path} has sr={sr_n}, expected {config.target_sr}. "
        "Run preprocess.py first."
    )

    # ── Mono safety ──
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # ── Random SNR ──
    snr_db = rng.uniform(config.snr_range[0], config.snr_range[1])

    noisy, clean_out = mix_at_snr(
        clean, noise, snr_db,
        impulsive=config.impulsive_mode, rng=rng,
    )

    return MixResult(
        noisy=noisy,
        clean=clean_out,
        snr_db=float(snr_db),
        noise_class=noise_class,
    )


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the actual SNR between clean and the noise component.

    This is used in tests to verify that mix_at_snr produces accurate SNRs.

    SNR = 20 · log10(rms(clean) / rms(noise))
    where noise = noisy − clean
    """
    noise = noisy - clean
    rms_c = _rms(clean)
    rms_n = _rms(noise)
    if rms_n < 1e-10:
        return float("inf")
    return 20.0 * np.log10(rms_c / rms_n)
