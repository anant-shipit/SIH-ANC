"""
per_snr_eval.py — Evaluate GTCRN performance at each SNR level.

Generates clean+noise mixtures at specific SNR levels and evaluates
GTCRN enhancement quality at each level independently.

Usage:
    from sih26052.eval.per_snr_eval import evaluate_per_snr

    results = evaluate_per_snr(model, clean_files, noise_files,
                               snr_levels=[-10, -5, 0, 5, 10, 15])
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

from sih26052.data.mixer import mix_at_snr
from sih26052.eval.full_metrics import FullMetricResult, compute_full_metrics

logger = logging.getLogger(__name__)


def evaluate_per_snr(
    enhance_fn: Callable[[np.ndarray], np.ndarray],
    clean_files: List[str | Path],
    noise_files: List[str | Path],
    snr_levels: List[float] = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
    n_samples_per_snr: int = 20,
    crop_samples: int = 32000,
    sr: int = 16000,
    seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """Evaluate enhancement at each SNR level.

    Parameters
    ----------
    enhance_fn : function that takes a noisy waveform and returns enhanced waveform
    clean_files : list of clean speech file paths
    noise_files : list of noise file paths
    snr_levels : SNR values to evaluate at (dB)
    n_samples_per_snr : number of test pairs per SNR level
    crop_samples : waveform length for evaluation

    Returns
    -------
    Dict mapping SNR → aggregated metric dict.
    """
    rng = np.random.default_rng(seed)
    results: Dict[float, Dict[str, float]] = {}

    for snr_db in snr_levels:
        logger.info("Evaluating at SNR = %.1f dB (%d samples)...", snr_db, n_samples_per_snr)
        metrics_lists: Dict[str, List[float]] = {
            "si_sdr_input": [], "si_sdr_output": [], "si_sdr_improvement": [],
            "sdr_input": [], "sdr_output": [], "sdr_improvement": [],
            "snr_input": [], "snr_output": [], "snr_improvement": [],
            "pesq_input": [], "pesq_output": [],
            "stoi_input": [], "stoi_output": [],
            "seg_snr_input": [], "seg_snr_output": [],
            "mse": [], "mae": [], "spectral_convergence": [],
        }

        for i in range(n_samples_per_snr):
            # Random clean + noise pair
            clean_path = clean_files[rng.integers(0, len(clean_files))]
            noise_path = noise_files[rng.integers(0, len(noise_files))]

            try:
                clean, sr_c = sf.read(str(clean_path), dtype="float32")
                noise, sr_n = sf.read(str(noise_path), dtype="float32")
            except Exception as e:
                logger.debug("Failed to load audio: %s", e)
                continue

            if clean.ndim > 1:
                clean = clean.mean(axis=1)
            if noise.ndim > 1:
                noise = noise.mean(axis=1)

            # Crop
            if len(clean) > crop_samples:
                start = rng.integers(0, len(clean) - crop_samples + 1)
                clean = clean[start:start + crop_samples]
            elif len(clean) < crop_samples:
                clean = np.pad(clean, (0, crop_samples - len(clean)))

            # Mix at target SNR
            noisy, clean_target = mix_at_snr(clean, noise, snr_db, rng=rng)

            # Enhance
            try:
                enhanced = enhance_fn(noisy)
            except Exception as e:
                logger.debug("Enhancement failed: %s", e)
                continue

            # Compute full metrics
            m = compute_full_metrics(clean_target, noisy, enhanced, sr=sr)

            metrics_lists["si_sdr_input"].append(m.si_sdr_input)
            metrics_lists["si_sdr_output"].append(m.si_sdr_output)
            metrics_lists["si_sdr_improvement"].append(m.si_sdr_improvement)
            metrics_lists["sdr_input"].append(m.sdr_input)
            metrics_lists["sdr_output"].append(m.sdr_output)
            metrics_lists["sdr_improvement"].append(m.sdr_improvement)
            metrics_lists["snr_input"].append(m.snr_input)
            metrics_lists["snr_output"].append(m.snr_output)
            metrics_lists["snr_improvement"].append(m.snr_improvement)
            if m.pesq_input is not None:
                metrics_lists["pesq_input"].append(m.pesq_input)
            if m.pesq_output is not None:
                metrics_lists["pesq_output"].append(m.pesq_output)
            if m.stoi_input is not None:
                metrics_lists["stoi_input"].append(m.stoi_input)
            if m.stoi_output is not None:
                metrics_lists["stoi_output"].append(m.stoi_output)
            metrics_lists["seg_snr_input"].append(m.seg_snr_input)
            metrics_lists["seg_snr_output"].append(m.seg_snr_output)
            metrics_lists["mse"].append(m.mse)
            metrics_lists["mae"].append(m.mae)
            metrics_lists["spectral_convergence"].append(m.spectral_convergence)

        # Aggregate
        results[snr_db] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in metrics_lists.items()
        }
        results[snr_db]["n_samples"] = float(len(metrics_lists["si_sdr_input"]))

        logger.info(
            "  SNR=%+6.1f dB: SI-SDRi=%+.2f dB | PESQ=%.3f | STOI=%.3f",
            snr_db,
            results[snr_db].get("si_sdr_improvement", 0),
            results[snr_db].get("pesq_output", 0),
            results[snr_db].get("stoi_output", 0),
        )

    return results


def format_per_snr_table(results: Dict[float, Dict[str, float]]) -> str:
    """Format per-SNR results as a readable ASCII table."""
    header = (
        f"{'SNR (dB)':>10} | {'SI-SDR In':>10} | {'SI-SDR Out':>10} | {'SI-SDRi':>8} | "
        f"{'PESQ':>6} | {'STOI':>6} | {'MSE':>8}"
    )
    lines = [header, "-" * len(header)]

    for snr in sorted(results.keys()):
        r = results[snr]
        lines.append(
            f"{snr:>+10.1f} | {r.get('si_sdr_input', 0):>10.2f} | "
            f"{r.get('si_sdr_output', 0):>10.2f} | {r.get('si_sdr_improvement', 0):>+8.2f} | "
            f"{r.get('pesq_output', 0):>6.3f} | {r.get('stoi_output', 0):>6.3f} | "
            f"{r.get('mse', 0):>8.6f}"
        )

    return "\n".join(lines)
