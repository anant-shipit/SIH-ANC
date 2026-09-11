"""
full_metrics.py — Comprehensive speech enhancement metrics (all 14).

Computes before-GTCRN (input) and after-GTCRN (output) metrics for
each sample, along with the improvement delta.

Metrics:
    1.  SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    2.  SI-SDRi (SI-SDR Improvement)
    3.  SDR (Signal-to-Distortion Ratio)
    4.  SDRi (SDR Improvement)
    5.  SNR (Signal-to-Noise Ratio)
    6.  SNRi (SNR Improvement)
    7.  PESQ (Perceptual Evaluation of Speech Quality)
    8.  STOI (Short-Time Objective Intelligibility)
    9.  ESTOI (Extended STOI)
    10. Segmental SNR
    11. MSE (Mean Squared Error)
    12. MAE (Mean Absolute Error)
    13. Spectral Convergence
    14. DNSMOS (if available)

Usage:
    from sih26052.eval.full_metrics import compute_full_metrics

    results = compute_full_metrics(clean, noisy, enhanced, sr=16000)
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FullMetricResult(NamedTuple):
    """Complete metric results for one (clean, noisy, enhanced) triplet."""
    # SI-SDR
    si_sdr_input: float
    si_sdr_output: float
    si_sdr_improvement: float
    # SDR
    sdr_input: float
    sdr_output: float
    sdr_improvement: float
    # SNR
    snr_input: float
    snr_output: float
    snr_improvement: float
    # Perceptual
    pesq_input: Optional[float]
    pesq_output: Optional[float]
    stoi_input: Optional[float]
    stoi_output: Optional[float]
    estoi_input: Optional[float]
    estoi_output: Optional[float]
    # Spectral
    seg_snr_input: float
    seg_snr_output: float
    spectral_convergence: float
    # Error
    mse: float
    mae: float
    # DNSMOS
    dnsmos: Optional[float]


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio."""
    ref = reference.astype(np.float64) - np.mean(reference)
    est = estimate.astype(np.float64) - np.mean(estimate)

    dot = np.sum(ref * est)
    s_target = (dot / (np.sum(ref ** 2) + 1e-8)) * ref
    e_noise = est - s_target

    return float(10.0 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8)
    ))


def _sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-Distortion Ratio (not scale-invariant)."""
    ref = reference.astype(np.float64)
    noise = estimate.astype(np.float64) - ref
    ref_pow = np.sum(ref ** 2)
    noise_pow = np.sum(noise ** 2)
    if noise_pow < 1e-10:
        return 100.0
    return float(10.0 * np.log10(ref_pow / (noise_pow + 1e-8)))


def _snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-Noise Ratio."""
    noise = estimate.astype(np.float64) - reference.astype(np.float64)
    sig_pow = np.sum(reference.astype(np.float64) ** 2)
    noise_pow = np.sum(noise ** 2)
    if noise_pow < 1e-10:
        return 100.0
    return float(10.0 * np.log10(sig_pow / (noise_pow + 1e-8)))


def _segmental_snr(reference: np.ndarray, estimate: np.ndarray,
                   frame_len: int = 512, hop: int = 256) -> float:
    """Segmental SNR: average SNR over short frames."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    noise = est - ref

    n_frames = max(1, (len(ref) - frame_len) // hop + 1)
    seg_snrs = []

    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(ref):
            break
        ref_frame = ref[start:end]
        noise_frame = noise[start:end]
        ref_pow = np.sum(ref_frame ** 2)
        noise_pow = np.sum(noise_frame ** 2)
        if ref_pow > 1e-10 and noise_pow > 1e-10:
            seg_snrs.append(10.0 * np.log10(ref_pow / noise_pow))

    if not seg_snrs:
        return 0.0
    # Clip extreme values
    seg_snrs = np.clip(seg_snrs, -30.0, 60.0)
    return float(np.mean(seg_snrs))


def _spectral_convergence(reference: np.ndarray, estimate: np.ndarray,
                          n_fft: int = 512) -> float:
    """Spectral convergence: ||X_ref - X_est|| / ||X_ref||."""
    ref_spec = np.abs(np.fft.rfft(reference.astype(np.float64), n=n_fft))
    est_spec = np.abs(np.fft.rfft(estimate.astype(np.float64), n=n_fft))
    ref_norm = np.linalg.norm(ref_spec)
    if ref_norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(ref_spec - est_spec) / ref_norm)


def _pesq(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> Optional[float]:
    """PESQ (wideband)."""
    try:
        from pesq import pesq
        return float(pesq(sr, reference, estimate, "wb"))
    except Exception:
        return None


def _stoi(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> Optional[float]:
    """STOI."""
    try:
        from pystoi import stoi
        return float(stoi(reference, estimate, sr, extended=False))
    except Exception:
        return None


def _estoi(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> Optional[float]:
    """Extended STOI."""
    try:
        from pystoi import stoi
        return float(stoi(reference, estimate, sr, extended=True))
    except Exception:
        return None


def compute_full_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
) -> FullMetricResult:
    """Compute all 14 speech enhancement metrics.

    Parameters
    ----------
    clean    : 1D float array — clean reference speech
    noisy    : 1D float array — noisy input (before GTCRN)
    enhanced : 1D float array — enhanced output (after GTCRN)
    sr       : sample rate

    Returns
    -------
    FullMetricResult with input/output/improvement for all metrics.
    """
    # Ensure same length
    min_len = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:min_len].astype(np.float32)
    noisy = noisy[:min_len].astype(np.float32)
    enhanced = enhanced[:min_len].astype(np.float32)

    # SI-SDR
    si_sdr_in = _si_sdr(clean, noisy)
    si_sdr_out = _si_sdr(clean, enhanced)

    # SDR
    sdr_in = _sdr(clean, noisy)
    sdr_out = _sdr(clean, enhanced)

    # SNR
    snr_in = _snr(clean, noisy)
    snr_out = _snr(clean, enhanced)

    # Segmental SNR
    seg_snr_in = _segmental_snr(clean, noisy)
    seg_snr_out = _segmental_snr(clean, enhanced)

    # Spectral Convergence (output only — lower is better)
    spec_conv = _spectral_convergence(clean, enhanced)

    # MSE / MAE
    mse = float(np.mean((clean - enhanced) ** 2))
    mae = float(np.mean(np.abs(clean - enhanced)))

    # PESQ
    pesq_in = _pesq(clean, noisy, sr)
    pesq_out = _pesq(clean, enhanced, sr)

    # STOI
    stoi_in = _stoi(clean, noisy, sr)
    stoi_out = _stoi(clean, enhanced, sr)

    # ESTOI
    estoi_in = _estoi(clean, noisy, sr)
    estoi_out = _estoi(clean, enhanced, sr)

    # DNSMOS (placeholder — requires API or model)
    dnsmos = None

    return FullMetricResult(
        si_sdr_input=si_sdr_in,
        si_sdr_output=si_sdr_out,
        si_sdr_improvement=si_sdr_out - si_sdr_in,
        sdr_input=sdr_in,
        sdr_output=sdr_out,
        sdr_improvement=sdr_out - sdr_in,
        snr_input=snr_in,
        snr_output=snr_out,
        snr_improvement=snr_out - snr_in,
        pesq_input=pesq_in,
        pesq_output=pesq_out,
        stoi_input=stoi_in,
        stoi_output=stoi_out,
        estoi_input=estoi_in,
        estoi_output=estoi_out,
        seg_snr_input=seg_snr_in,
        seg_snr_output=seg_snr_out,
        spectral_convergence=spec_conv,
        mse=mse,
        mae=mae,
        dnsmos=dnsmos,
    )
