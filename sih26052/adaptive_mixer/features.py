"""
features.py — Feature extraction for the AdaptiveMixingPolicy.

Extracts lightweight acoustic features from speech and noise signals
that inform the mixing policy about signal characteristics.

These are numpy-only operations — no torch dependency.

Usage:
    from sih26052.adaptive_mixer.features import extract_speech_features, extract_noise_features

    speech_feats = extract_speech_features(clean_wav, sr=16000)
    noise_feats = extract_noise_features(noise_wav, sr=16000, category="gunfire")
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _rms(x: np.ndarray) -> float:
    """Root Mean Square energy."""
    if len(x) == 0:
        return 0.0
    val = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return float(val) if np.isfinite(val) else 0.0


def _spectral_centroid(x: np.ndarray, sr: int = 16000, n_fft: int = 512) -> float:
    """Spectral centroid in Hz."""
    if len(x) < n_fft:
        return 0.0
    spec = np.abs(np.fft.rfft(x[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    total_energy = np.sum(spec)
    if total_energy < 1e-10:
        return 0.0
    return float(np.sum(freqs * spec) / total_energy)


def _spectral_bandwidth(x: np.ndarray, sr: int = 16000, n_fft: int = 512) -> float:
    """Spectral bandwidth in Hz."""
    if len(x) < n_fft:
        return 0.0
    spec = np.abs(np.fft.rfft(x[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    total_energy = np.sum(spec)
    if total_energy < 1e-10:
        return 0.0
    centroid = np.sum(freqs * spec) / total_energy
    return float(np.sqrt(np.sum(spec * (freqs - centroid) ** 2) / total_energy))


def _spectral_rolloff(x: np.ndarray, sr: int = 16000, n_fft: int = 512,
                      rolloff: float = 0.85) -> float:
    """Frequency below which `rolloff` fraction of energy is contained."""
    if len(x) < n_fft:
        return 0.0
    spec = np.abs(np.fft.rfft(x[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    total_energy = np.sum(spec)
    if total_energy < 1e-10:
        return 0.0
    cumsum = np.cumsum(spec)
    idx = np.searchsorted(cumsum, rolloff * total_energy)
    return float(freqs[min(idx, len(freqs) - 1)])


def _spectral_flatness(x: np.ndarray, n_fft: int = 512) -> float:
    """Spectral flatness (0 = tonal, 1 = noise-like)."""
    if len(x) < n_fft:
        return 0.0
    spec = np.abs(np.fft.rfft(x[:n_fft])) + 1e-10
    geo_mean = np.exp(np.mean(np.log(spec)))
    arith_mean = np.mean(spec)
    return float(geo_mean / arith_mean) if arith_mean > 1e-10 else 0.0


def _zero_crossing_rate(x: np.ndarray) -> float:
    """Zero crossing rate (crossings per sample)."""
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (len(x) - 1))


def _speech_activity_ratio(x: np.ndarray, frame_len: int = 512,
                           hop_len: int = 256, threshold_db: float = -30.0) -> float:
    """Fraction of frames with energy above threshold (Voice Activity Ratio)."""
    if len(x) < frame_len:
        return 1.0
    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames <= 0:
        return 1.0

    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
    )
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    max_e = np.max(energies)
    if max_e < 1e-10:
        return 0.0
    threshold = max_e * (10.0 ** (threshold_db / 10.0))
    return float(np.mean(energies > threshold))


def _transient_ratio(x: np.ndarray, frame_len: int = 512,
                     hop_len: int = 256) -> float:
    """Ratio of transient (high energy spike) frames to total frames."""
    if len(x) < frame_len:
        return 0.0
    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames <= 3:
        return 0.0

    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
    )
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    median_e = np.median(energies)
    if median_e < 1e-10:
        return 0.0
    # A frame is "transient" if its energy is > 10× the median
    return float(np.mean(energies > 10.0 * median_e))


def extract_speech_features(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract 8-dimensional feature vector from a clean speech waveform.

    Features:
        0: RMS energy (log scale)
        1: Duration in seconds
        2: Spectral centroid (normalized to [0, 1])
        3: Spectral bandwidth (normalized to [0, 1])
        4: Spectral rolloff (normalized to [0, 1])
        5: Spectral flatness [0, 1]
        6: Zero crossing rate [0, 1]
        7: Speech activity ratio [0, 1]
    """
    rms = _rms(wav)
    log_rms = float(np.log10(rms + 1e-8))
    duration = len(wav) / sr
    centroid = _spectral_centroid(wav, sr) / (sr / 2)  # normalize to Nyquist
    bandwidth = _spectral_bandwidth(wav, sr) / (sr / 2)
    rolloff = _spectral_rolloff(wav, sr) / (sr / 2)
    flatness = _spectral_flatness(wav)
    zcr = _zero_crossing_rate(wav)
    sar = _speech_activity_ratio(wav)

    return np.array([
        log_rms, duration, centroid, bandwidth,
        rolloff, flatness, zcr, sar,
    ], dtype=np.float32)


def extract_noise_features(wav: np.ndarray, sr: int = 16000,
                           category: str = "unknown",
                           category_to_idx: Optional[dict] = None,
                           n_categories: int = 6) -> np.ndarray:
    """Extract 10-dimensional feature vector from a noise waveform.

    Features:
        0: RMS energy (log scale)
        1: Duration in seconds
        2: Spectral centroid (normalized)
        3: Spectral bandwidth (normalized)
        4: Spectral rolloff (normalized)
        5: Spectral flatness [0, 1]
        6: Zero crossing rate
        7: Transient ratio [0, 1]
        8: Category index (one-hot encoded as float)
        9: Dataset source index (0 for unknown)
    """
    rms = _rms(wav)
    log_rms = float(np.log10(rms + 1e-8))
    duration = len(wav) / sr
    centroid = _spectral_centroid(wav, sr) / (sr / 2)
    bandwidth = _spectral_bandwidth(wav, sr) / (sr / 2)
    rolloff = _spectral_rolloff(wav, sr) / (sr / 2)
    flatness = _spectral_flatness(wav)
    zcr = _zero_crossing_rate(wav)
    transient = _transient_ratio(wav)

    if category_to_idx is not None:
        cat_idx = float(category_to_idx.get(category, 0))
    else:
        default_mapping = {
            "background": 0, "environmental": 1, "gunfire": 2,
            "urban": 3, "alarm": 4, "human": 5,
        }
        cat_idx = float(default_mapping.get(category, 0))

    return np.array([
        log_rms, duration, centroid, bandwidth,
        rolloff, flatness, zcr, transient,
        cat_idx / max(n_categories - 1, 1),  # normalize to [0, 1]
        0.0,  # dataset source (placeholder)
    ], dtype=np.float32)
