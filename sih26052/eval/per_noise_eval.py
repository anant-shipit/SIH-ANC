"""
per_noise_eval.py — Evaluate GTCRN performance by noise category.

Generates clean+noise mixtures using specific noise categories and evaluates
GTCRN enhancement quality for each category independently.

Usage:
    from sih26052.eval.per_noise_eval import evaluate_per_noise_type

    results = evaluate_per_noise_type(model, clean_files, noise_pools)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

from sih26052.data.mixer import mix_at_snr
from sih26052.eval.full_metrics import compute_full_metrics

logger = logging.getLogger(__name__)


def evaluate_per_noise_type(
    enhance_fn: Callable[[np.ndarray], np.ndarray],
    clean_files: List[str | Path],
    noise_pools: Dict[str, List[str | Path]],
    snr_db: float = 0.0,
    n_samples_per_type: int = 20,
    crop_samples: int = 32000,
    sr: int = 16000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate enhancement by noise category.

    Parameters
    ----------
    enhance_fn : function that takes a noisy waveform and returns enhanced waveform
    clean_files : list of clean speech file paths
    noise_pools : dict mapping noise category → list of noise file paths
    snr_db : fixed SNR for fair comparison across noise types
    n_samples_per_type : number of test pairs per noise category

    Returns
    -------
    Dict mapping noise_type → aggregated metric dict.
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, float]] = {}

    for noise_type, noise_files in noise_pools.items():
        if not noise_files:
            logger.warning("No files for noise type '%s', skipping.", noise_type)
            continue

        logger.info("Evaluating noise type '%s' (%d files, %d samples at %.1f dB)...",
                     noise_type, len(noise_files), n_samples_per_type, snr_db)

        metrics_lists: Dict[str, List[float]] = {
            "si_sdr_input": [], "si_sdr_output": [], "si_sdr_improvement": [],
            "sdr_improvement": [],
            "snr_input": [], "snr_output": [], "snr_improvement": [],
            "pesq_input": [], "pesq_output": [],
            "stoi_input": [], "stoi_output": [],
            "mse": [], "mae": [], "spectral_convergence": [],
        }

        for i in range(n_samples_per_type):
            clean_path = clean_files[rng.integers(0, len(clean_files))]
            noise_path = noise_files[rng.integers(0, len(noise_files))]

            try:
                clean, _ = sf.read(str(clean_path), dtype="float32")
                noise, _ = sf.read(str(noise_path), dtype="float32")
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

            # Determine if impulsive
            is_impulsive = noise_type in ("gunfire", "gunshot", "impulsive")

            # Mix
            noisy, clean_target = mix_at_snr(
                clean, noise, snr_db, impulsive=is_impulsive, rng=rng,
            )

            try:
                enhanced = enhance_fn(noisy)
            except Exception as e:
                logger.debug("Enhancement failed: %s", e)
                continue

            m = compute_full_metrics(clean_target, noisy, enhanced, sr=sr)

            metrics_lists["si_sdr_input"].append(m.si_sdr_input)
            metrics_lists["si_sdr_output"].append(m.si_sdr_output)
            metrics_lists["si_sdr_improvement"].append(m.si_sdr_improvement)
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
            metrics_lists["mse"].append(m.mse)
            metrics_lists["mae"].append(m.mae)
            metrics_lists["spectral_convergence"].append(m.spectral_convergence)

        results[noise_type] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in metrics_lists.items()
        }
        results[noise_type]["n_samples"] = len(metrics_lists["si_sdr_input"])

        logger.info(
            "  %s: SI-SDRi=%+.2f dB | PESQ=%.3f | STOI=%.3f",
            noise_type,
            results[noise_type].get("si_sdr_improvement", 0),
            results[noise_type].get("pesq_output", 0),
            results[noise_type].get("stoi_output", 0),
        )

    return results


def format_per_noise_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format per-noise-type results as a readable ASCII table."""
    header = (
        f"{'Noise Type':>20} | {'SI-SDR In':>10} | {'SI-SDR Out':>10} | {'SI-SDRi':>8} | "
        f"{'PESQ':>6} | {'STOI':>6} | {'N':>4}"
    )
    lines = [header, "-" * len(header)]

    for noise_type in sorted(results.keys()):
        r = results[noise_type]
        lines.append(
            f"{noise_type:>20} | {r.get('si_sdr_input', 0):>10.2f} | "
            f"{r.get('si_sdr_output', 0):>10.2f} | "
            f"{r.get('si_sdr_improvement', 0):>+8.2f} | "
            f"{r.get('pesq_output', 0):>6.3f} | {r.get('stoi_output', 0):>6.3f} | "
            f"{int(r.get('n_samples', 0)):>4d}"
        )

    return "\n".join(lines)
