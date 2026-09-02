"""
validate.py — Per-epoch validation using the eval harness.

Runs after each training epoch to:
    1. Compute PESQ, STOI, SI-SNR on the fixed validation set.
    2. Split results by subset (stationary, impulsive, real).
    3. Detect catastrophic forgetting: if impulsive improves while
       stationary regresses → alert.

Usage:
    from sih26052.train.validate import validate_epoch

    results = validate_epoch(model, val_manifest, epoch=5)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def validate_epoch(
    model: nn.Module,
    val_manifest: str | Path,
    device: torch.device,
    epoch: int,
    nfft: int = 512,
    hop: int = 256,
    sr: int = 16000,
    max_pairs: int | None = None,
) -> dict:
    """Run validation after one training epoch.

    Parameters
    ----------
    model        : the GTCRN model in eval mode
    val_manifest : path to the validation JSONL manifest
    device       : torch device
    epoch        : current epoch number (for logging)
    max_pairs    : limit for quick validation during development

    Returns
    -------
    dict with per-subset and overall metrics
    """
    from sih26052.eval.harness import run_eval_harness

    model.eval()
    window = torch.hann_window(nfft).to(device)

    def enhance_fn(noisy_wav: np.ndarray) -> np.ndarray:
        """Enhance a waveform using the model."""
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy_wav).unsqueeze(0).to(device)

            # STFT
            noisy_stft = torch.stft(
                noisy_t, nfft, hop, window=window, return_complex=False
            )

            # Enhance
            pred_stft = model(noisy_stft)

            # ISTFT
            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enhanced = torch.istft(
                pred_complex, nfft, hop, window=window, length=len(noisy_wav)
            )

            return enhanced.squeeze(0).cpu().numpy()

    results = run_eval_harness(
        val_manifest,
        enhance_fn=enhance_fn,
        sr=sr,
        max_pairs=max_pairs,
        align=True,
    )

    # Log the table
    logger.info("Epoch %d validation results:\n%s", epoch, results.format_table())

    # Build summary dict
    summary = {
        "epoch": epoch,
        "overall_pesq_out": results.overall.pesq_out_mean,
        "overall_stoi_out": results.overall.stoi_out_mean,
        "overall_si_snr_out": results.overall.si_snr_out_mean,
        "overall_pesq_delta": results.overall.pesq_delta,
    }

    for subset_name, sm in results.subsets.items():
        summary[f"{subset_name}_pesq_out"] = sm.pesq_out_mean
        summary[f"{subset_name}_stoi_out"] = sm.stoi_out_mean

    return summary


def check_catastrophic_forgetting(
    history: list[dict],
    watch_subset: str = "stationary",
    improve_subset: str = "impulsive",
    threshold: float = 0.05,
) -> bool:
    """Check if one subset is improving at the expense of another.

    Returns True if forgetting is detected.
    """
    if len(history) < 3:
        return False

    recent = history[-3:]
    watch_trend = [h.get(f"{watch_subset}_pesq_out", 0) for h in recent]
    improve_trend = [h.get(f"{improve_subset}_pesq_out", 0) for h in recent]

    watch_declining = watch_trend[-1] < watch_trend[0] - threshold
    improve_increasing = improve_trend[-1] > improve_trend[0] + threshold

    if watch_declining and improve_increasing:
        logger.warning(
            "⚠️  CATASTROPHIC FORGETTING DETECTED: "
            "'%s' declining (%.3f → %.3f) while '%s' improving (%.3f → %.3f). "
            "Consider reducing impulsive proportion.",
            watch_subset, watch_trend[0], watch_trend[-1],
            improve_subset, improve_trend[0], improve_trend[-1],
        )
        return True

    return False
