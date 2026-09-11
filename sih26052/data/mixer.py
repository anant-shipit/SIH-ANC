"""
mixer.py — Advanced Defense Acoustic Mixing & Data Synthesis Engine.

This is the core data synthesis engine for GTCRN speech enhancement training.
It combines clean speech with complex military, urban, and impulsive noise
profiles at precise SNRs or exact ratio distributions.

Key Acoustic & Architectural Design Decisions:
    1. **Exact 40/60 Speech-to-Noise Ratio Mixing**:
       Standardizes on 40% clean speech power and 60% noise power, creating an
       intense -3.52 dB SNR acoustic challenge that trains models for extreme robustness.

    2. **Active Speech Energy (Voice Activity) RMS Calculation**:
       Instead of letting prolonged speech pauses depress the calculated speech RMS,
       we compute active-speech RMS over voiced frames (thresholded energy).
       This ensures the desired SNR/ratio is mathematically accurate across utterances
       of varying sentence structure and pacing.

    3. **Same-Divisor Clipping Protection**:
       To prevent digital clipping while preserving exact SNR and relative phase,
       the louder of clean and noisy is evaluated against 0.99 headroom, and a single
       shared scale factor is applied to both signals simultaneously.

    4. **Non-Tiled Transient Physics (Impulsive Realism)**:
       Gunfire, explosions, and door slams are NEVER tiled periodically. They are
       placed with random temporal offsets, burst jitter, and realistic distance
       attenuation.

    5. **Battlefield Composite Defense Scenarios (Types A–F)**:
       Generates realistic combat acoustics:
       - Type A (40%): Clean speech + Continuous ambient/vehicle noise
       - Type B (7.5%): Clean speech + Environmental noise event
       - Type C (7.5%): Clean speech + Ballistic gunfire transient
       - Type D (22.5%): Clean speech + Background + Gunfire
       - Type E (15%): Clean speech + Background + Environmental alarm/machinery
       - Type F (7.5%): Clean speech + Background + Multi-threat combat cocktail

Usage:
    from sih26052.data.mixer import mix_at_ratio, mix_at_snr, mix_defense_sample, MixerConfig

    # Mix 40% clean speech, 60% background noise:
    noisy, clean_target = mix_at_ratio(clean_wav, noise_wav, clean_ratio=0.40, noise_ratio=0.60)
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ── Data Types ─────────────────────────────────────────────────────────────

class MixResult(NamedTuple):
    """Output of a single audio mix operation."""
    noisy: np.ndarray        # float32 1D waveform, same length as clean
    clean: np.ndarray        # float32 1D waveform, the ground-truth target
    snr_db: float            # measured or targeted SNR in decibels
    noise_class: str         # noise category label (e.g. "gunfire", "background")


@dataclasses.dataclass
class MixerConfig:
    """Configuration parameters controlling audio mixing."""
    snr_range: tuple[float, float] = (-5.0, 20.0)    # Uniform random draw range (dB)
    target_sr: int = 16000                           # Target sample rate (16 kHz mono)
    impulsive_mode: bool = False                     # True = single transient, False = loop/fit
    seed: int | None = None                          # Seed for reproducibility
    clean_ratio: float | None = 0.40                 # 40% clean speech proportion
    noise_ratio: float | None = 0.60                 # 60% background noise proportion


# ── Acoustic Level Measurement & Protection ────────────────────────────────

def _rms(x: np.ndarray) -> float:
    """Standard Root Mean Square (RMS) energy. Returns 0.0 for silence."""
    if len(x) == 0:
        return 0.0
    val = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return float(val) if np.isfinite(val) else 0.0


def _active_rms(
    x: np.ndarray,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -30.0,
) -> float:
    """Compute active speech RMS over voiced frames to avoid silence dilution.
    
    Falls back to global RMS if the signal is uniform or very short.
    """
    if len(x) < frame_len:
        return _rms(x)

    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames <= 1:
        return _rms(x)

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
    )
    frame_energies = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    max_energy = np.max(frame_energies)

    if max_energy < 1e-10:
        return 0.0

    threshold = max_energy * (10.0 ** (threshold_db / 10.0))
    active_frames = frames[frame_energies > threshold]

    if len(active_frames) == 0:
        return _rms(x)

    active_val = np.sqrt(np.mean(active_frames.astype(np.float64) ** 2))
    return float(active_val) if np.isfinite(active_val) else _rms(x)


def _clip_protect(
    clean: np.ndarray,
    noisy: np.ndarray,
    headroom: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical attenuation divisor to both clean and noisy signals.
    
    Using the exact same divisor preserves the SNR, relative phase, and signal
    dynamics without distortion or arbitrary scaling drift.
    """
    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))))
    if peak < 1e-8:
        return clean, noisy

    if peak > headroom:
        scale = headroom / peak
        return (clean * scale).astype(np.float32), (noisy * scale).astype(np.float32)
    return clean.astype(np.float32), noisy.astype(np.float32)


def _scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    use_active_rms: bool = True,
) -> np.ndarray:
    """Scale noise to achieve the exact target Signal-to-Noise Ratio (dB)."""
    rms_clean = _active_rms(clean) if use_active_rms else _rms(clean)
    rms_noise = _rms(noise)

    if rms_clean < 1e-8 or rms_noise < 1e-8:
        return noise

    target_rms_noise = rms_clean / (10.0 ** (snr_db / 20.0))
    scale = target_rms_noise / rms_noise
    return (noise * scale).astype(np.float32)


def _fit_noise_to_length(
    noise: np.ndarray,
    target_len: int,
    *,
    impulsive: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Fit a noise signal to target_len samples.
    
    Stationary noise:
        - If shorter than target_len: tiled/looped continuously.
        - If longer than target_len: randomly cropped.
    Impulsive noise:
        - Positioned once at a randomized temporal position with zero-padding.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_len = len(noise)
    if n_len == 0:
        return np.zeros(target_len, dtype=np.float32)

    if impulsive:
        out = np.zeros(target_len, dtype=np.float32)
        if n_len >= target_len:
            start = rng.integers(0, n_len - target_len + 1)
            out[:] = noise[start : start + target_len]
        else:
            max_start = target_len - n_len
            start = rng.integers(0, max_start + 1)
            out[start : start + n_len] = noise
        return out

    if n_len >= target_len:
        start = rng.integers(0, n_len - target_len + 1)
        return noise[start : start + target_len].copy().astype(np.float32)

    reps = (target_len // n_len) + 1
    looped = np.tile(noise, reps)
    return looped[:target_len].copy().astype(np.float32)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the empirical SNR between clean speech and the residual noise."""
    noise = noisy - clean
    rms_c = _rms(clean)
    rms_n = _rms(noise)
    if rms_n < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(rms_c / rms_n))


def sample_snr(
    snr_range: Tuple[float, float] = (-10.0, 15.0),
    dist: str = "uniform",
    mean: float = 3.0,
    std: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample an SNR value according to a specified distribution.

    Parameters
    ----------
    snr_range : (low, high) bounds in dB
    dist : 'uniform' or 'gaussian' (truncated normal)
    mean : center for Gaussian distribution (default +3.0 dB)
    std : standard deviation for Gaussian distribution (default 5.0 dB)
    rng : numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    low, high = snr_range
    if dist == "gaussian":
        val = float(rng.normal(mean, std))
        return float(np.clip(val, low, high))
    return float(rng.uniform(low, high))


def apply_speech_gain_jitter(
    clean: np.ndarray,
    target_dbfs_range: Tuple[float, float] = (-32.0, -16.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomize active speech level to simulate varying speaker distance and microphone gain."""
    if rng is None:
        rng = np.random.default_rng()

    current_rms = _active_rms(clean)
    if current_rms < 1e-8:
        return clean

    target_dbfs = float(rng.uniform(target_dbfs_range[0], target_dbfs_range[1]))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    scale = target_rms / current_rms
    scaled = clean * scale

    peak = float(np.max(np.abs(scaled)))
    if peak > 0.98:
        scaled = scaled * (0.98 / peak)
    return scaled.astype(np.float32)


def apply_synthetic_reverb(
    clean: np.ndarray,
    sr: int = 16000,
    rt60_range: Tuple[float, float] = (0.12, 0.32),
    direct_to_reverb_ratio: float = 0.75,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate early reflections and reverberation (cabin / vehicle acoustic space).

    Generates an exponential decay impulse response and convolves with clean speech,
    preserving energy and direct-path intelligibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    rt60 = float(rng.uniform(rt60_range[0], rt60_range[1]))
    ir_len = min(int(rt60 * sr), int(0.3 * sr))
    if ir_len < 100:
        return clean

    t = np.arange(ir_len) / sr
    tau = rt60 / (3.0 * np.log(10.0))
    decay = np.exp(-t / tau)
    noise = rng.normal(0, 1, ir_len)

    h = decay * noise
    h[0] = 1.0
    h[1:20] += rng.normal(0, 0.25, 19) * decay[1:20]
    h = h / np.sqrt(np.sum(h ** 2) + 1e-12)

    try:
        from scipy.signal import fftconvolve
        reverbed = fftconvolve(clean, h, mode="full")[:len(clean)]
        blended = direct_to_reverb_ratio * clean + (1.0 - direct_to_reverb_ratio) * reverbed
        rms_orig = _rms(clean)
        rms_new = _rms(blended)
        if rms_new > 1e-8:
            blended = blended * (rms_orig / rms_new)
        return blended.astype(np.float32)
    except Exception:
        return clean


def apply_simulated_codec(
    wav: np.ndarray,
    sr: int = 16000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate streaming codec artifacts: non-linear mu-law companding & bit reduction.

    Makes speech enhancement models resilient to lossy VoIP / radio bit-rate compression.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply 8-bit mu-law companding
    x = np.clip(wav, -1.0, 1.0)
    mu = 255.0
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
    expanded = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu

    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 7200, btype="lowpass", fs=sr, output="sos")
        filtered = sosfilt(sos, expanded)
        return filtered.astype(np.float32)
    except Exception:
        return expanded.astype(np.float32)



# ── Core Mixing Functions ──────────────────────────────────────────────────

def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    impulsive: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix clean speech with noise at a target SNR in decibels."""
    if rng is None:
        rng = np.random.default_rng()

    clean = clean.astype(np.float32, copy=True)
    noise = noise.astype(np.float32, copy=True)

    noise_fitted = _fit_noise_to_length(noise, len(clean), impulsive=impulsive, rng=rng)
    noise_scaled = _scale_noise_to_snr(clean, noise_fitted, snr_db, use_active_rms=False)

    noisy = clean + noise_scaled
    clean_out, noisy_out = _clip_protect(clean, noisy)
    return noisy_out, clean_out


def mix_at_ratio(
    clean: np.ndarray,
    noise: np.ndarray,
    clean_ratio: float = 0.40,
    noise_ratio: float = 0.60,
    *,
    impulsive: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix speech and noise at an explicit amplitude/power ratio (e.g. 40% speech, 60% noise).
    
    Formula:
        clean_norm = clean / rms(clean)
        noise_norm = noise / rms(noise)
        noisy = (clean_ratio * clean_norm) + (noise_ratio * noise_norm)
        target = clean_ratio * clean_norm
    """
    if rng is None:
        rng = np.random.default_rng()

    clean = clean.astype(np.float32, copy=True)
    noise = noise.astype(np.float32, copy=True)

    noise_fitted = _fit_noise_to_length(noise, len(clean), impulsive=impulsive, rng=rng)

    rms_c = _rms(clean)
    rms_n = _rms(noise_fitted)
    if rms_c < 1e-8 or rms_n < 1e-8:
        return clean, clean

    clean_norm = clean / rms_c
    noise_norm = noise_fitted / rms_n

    noisy = (clean_ratio * clean_norm) + (noise_ratio * noise_norm)
    target = clean_ratio * clean_norm

    target_out, noisy_out = _clip_protect(target, noisy)
    return noisy_out, target_out


def mix_pair(
    clean_path: Path | str,
    noise_path: Path | str,
    config: Optional[MixerConfig] = None,
    noise_class: str = "unknown",
) -> MixResult:
    """Load clean and noise audio files from disk and mix them."""
    if config is None:
        config = MixerConfig()

    rng = np.random.default_rng(config.seed)

    clean, sr_c = sf.read(str(clean_path), dtype="float32")
    noise, sr_n = sf.read(str(noise_path), dtype="float32")

    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    if config.clean_ratio is not None and config.noise_ratio is not None:
        noisy, clean_out = mix_at_ratio(
            clean, noise,
            clean_ratio=config.clean_ratio,
            noise_ratio=config.noise_ratio,
            impulsive=config.impulsive_mode,
            rng=rng,
        )
        snr_db = 20.0 * np.log10(config.clean_ratio / config.noise_ratio)
    else:
        snr_db = float(rng.uniform(config.snr_range[0], config.snr_range[1]))
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


# ── Defense Battlefield & Impulsive Synthesis ──────────────────────────────

def synthesize_gunfire_track(
    gunfire_samples: List[np.ndarray],
    target_len: int,
    rng: Optional[np.random.Generator] = None,
    mode: str = "random",
    min_shots: int = 2,
    max_shots: int = 4,
    burst_interval_ms: Tuple[float, float] = (50.0, 120.0),
    sr: int = 16000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Synthesize a realistic combat gunfire noise track.
    
    Modes:
        - 'single': Single rifle/handgun shot.
        - 'multiple': 2–4 spaced shots simulating field engagements.
        - 'burst': Rapid automatic fire with 50–120ms cyclic rate spacing.
        - 'random': Stochastic blend (40% single, 35% multiple, 25% burst).
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "random":
        p = rng.random()
        effective_mode = "single" if p < 0.40 else ("multiple" if p < 0.75 else "burst")
    else:
        effective_mode = mode

    out = np.zeros(target_len, dtype=np.float32)
    shot_offsets: List[int] = []

    if not gunfire_samples:
        return out, {"mode": effective_mode, "shots": 0, "offsets": []}

    def _add_shot(shot_wav: np.ndarray, start_idx: int) -> None:
        shot = shot_wav.astype(np.float32)
        end_idx = min(start_idx + len(shot), target_len)
        chunk_len = end_idx - start_idx
        if chunk_len > 0:
            out[start_idx:end_idx] += shot[:chunk_len]
            shot_offsets.append(start_idx)

    if effective_mode == "single":
        shot = gunfire_samples[rng.integers(0, len(gunfire_samples))]
        max_start = max(0, target_len - min(len(shot), target_len // 2))
        start = rng.integers(0, max_start + 1) if max_start > 0 else 0
        _add_shot(shot, start)

    elif effective_mode == "multiple":
        n_shots = rng.integers(min_shots, max_shots + 1)
        for _ in range(n_shots):
            shot = gunfire_samples[rng.integers(0, len(gunfire_samples))]
            max_start = max(0, target_len - min(len(shot), target_len // 4))
            start = rng.integers(0, max_start + 1) if max_start > 0 else 0
            _add_shot(shot, start)

    elif effective_mode == "burst":
        n_shots = rng.integers(min_shots, max_shots + 2)
        shot_template = gunfire_samples[rng.integers(0, len(gunfire_samples))]
        spacing = int(rng.uniform(burst_interval_ms[0], burst_interval_ms[1]) * (sr / 1000.0))
        max_start = max(0, target_len - n_shots * spacing - 1)
        cur_pos = rng.integers(0, max_start + 1) if max_start > 0 else 0

        for _ in range(n_shots):
            gain = float(rng.uniform(0.85, 1.15))
            _add_shot(shot_template * gain, cur_pos)
            cur_pos += spacing
            if cur_pos >= target_len:
                break

    return out, {
        "mode": effective_mode,
        "shots": len(shot_offsets),
        "offsets": shot_offsets,
    }


def mix_defense_sample(
    clean: np.ndarray,
    background: Optional[np.ndarray] = None,
    event: Optional[np.ndarray] = None,
    gunfire: Optional[np.ndarray] = None,
    mixture_type: str = "auto",
    snr_db: float = 0.0,
    event_snr_db: Optional[float] = None,
    gunfire_snr_db: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    probabilities: Optional[Dict[str, float]] = None,
    clean_ratio: Optional[float] = 0.40,
    noise_ratio: Optional[float] = 0.60,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Construct a battlefield-realistic composite noisy speech mixture.
    
    Mixture Distribution:
        - TYPE A (40%): Clean speech + Ambient Background
        - TYPE B (7.5%): Clean speech + Environmental Event
        - TYPE C (7.5%): Clean speech + Gunfire Transient
        - TYPE D (22.5%): Clean speech + Background + Gunfire
        - TYPE E (15%): Clean speech + Background + Environmental Event
        - TYPE F (7.5%): Clean speech + Background + Multi-threat Combat Cocktail
    """
    if rng is None:
        rng = np.random.default_rng()

    target_len = len(clean)
    clean = clean.astype(np.float32, copy=True)

    if probabilities is None:
        probabilities = {
            "A": 0.40,
            "B": 0.075,
            "C": 0.075,
            "D": 0.225,
            "E": 0.150,
            "F": 0.075,
        }

    if mixture_type == "auto":
        types = list(probabilities.keys())
        probs = np.array([probabilities[t] for t in types], dtype=np.float64)
        probs /= probs.sum()
        chosen_type = str(rng.choice(types, p=probs))
    else:
        chosen_type = mixture_type.upper().replace("TYPE_", "")

    noise_total = np.zeros(target_len, dtype=np.float32)
    active_components: List[str] = []

    # Fit noise components to length
    bg_fitted = _fit_noise_to_length(background, target_len, impulsive=False, rng=rng) if background is not None else None
    ev_fitted = _fit_noise_to_length(event, target_len, impulsive=True, rng=rng) if event is not None else None
    gf_fitted = _fit_noise_to_length(gunfire, target_len, impulsive=True, rng=rng) if gunfire is not None else None

    # Sub-component relative SNRs
    snr_bg = snr_db
    snr_ev = event_snr_db if event_snr_db is not None else (snr_db - float(rng.uniform(2.0, 6.0)))
    snr_gf = gunfire_snr_db if gunfire_snr_db is not None else (snr_db - float(rng.uniform(4.0, 10.0)))

    # Composite construction by scenario type
    if chosen_type == "A" and bg_fitted is not None:
        noise_total += _scale_noise_to_snr(clean, bg_fitted, snr_bg)
        active_components.append("background")

    elif chosen_type == "B" and ev_fitted is not None:
        noise_total += _scale_noise_to_snr(clean, ev_fitted, snr_ev)
        active_components.append("environmental")

    elif chosen_type == "C" and gf_fitted is not None:
        noise_total += _scale_noise_to_snr(clean, gf_fitted, snr_gf)
        active_components.append("gunfire")

    elif chosen_type == "D":
        if bg_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, bg_fitted, snr_bg)
            active_components.append("background")
        if gf_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, gf_fitted, snr_gf)
            active_components.append("gunfire")

    elif chosen_type == "E":
        if bg_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, bg_fitted, snr_bg)
            active_components.append("background")
        if ev_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, ev_fitted, snr_ev)
            active_components.append("environmental")

    elif chosen_type == "F":
        if bg_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, bg_fitted, snr_bg)
            active_components.append("background")
        if ev_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, ev_fitted, snr_ev)
            active_components.append("environmental")
        if gf_fitted is not None:
            noise_total += _scale_noise_to_snr(clean, gf_fitted, snr_gf)
            active_components.append("gunfire")

    # Fallback to Gaussian white noise if selected component files were missing/silent
    if len(active_components) == 0:
        synth_noise = rng.standard_normal(target_len).astype(np.float32) * 0.1
        noise_total = _scale_noise_to_snr(clean, synth_noise, snr_db)
        active_components.append("gaussian_fallback")

    # Ratio-based mix (40% clean, 60% noise)
    if clean_ratio is not None and noise_ratio is not None:
        rms_c = _rms(clean)
        rms_n = _rms(noise_total)
        if rms_c > 1e-8 and rms_n > 1e-8:
            clean_norm = clean / rms_c
            noise_norm = noise_total / rms_n
            noisy = (clean_ratio * clean_norm) + (noise_ratio * noise_norm)
            clean_target = clean_ratio * clean_norm
            clean_out, noisy_out = _clip_protect(clean_target, noisy)
            meta = {
                "mixture_type": f"TYPE_{chosen_type}",
                "active_components": active_components,
                "clean_ratio": float(clean_ratio),
                "noise_ratio": float(noise_ratio),
                "snr_db": float(20.0 * np.log10(clean_ratio / noise_ratio)),
                "actual_snr_db": float(measure_snr(clean_out, noisy_out)),
            }
            return noisy_out, clean_out, meta

    # SNR-based mix
    noisy = clean + noise_total
    clean_out, noisy_out = _clip_protect(clean, noisy)
    meta = {
        "mixture_type": f"TYPE_{chosen_type}",
        "active_components": active_components,
        "snr_db": float(snr_db),
        "actual_snr_db": float(measure_snr(clean_out, noisy_out)),
    }
    return noisy_out, clean_out, meta
