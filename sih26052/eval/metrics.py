"""
metrics.py — PESQ, STOI, and SI-SNR metric wrappers.

Three metrics, three views of quality:

    PESQ  (Perceptual Evaluation of Speech Quality)
        Correlates with human MOS scores.  Range: −0.5 to 4.5.
        Uses the `pesq` library in wideband (wb) mode at 16 kHz.
        Expensive (~0.5s per pair), so we try/except and count skips.

    STOI  (Short-Time Objective Intelligibility)
        Measures how intelligible speech is.  Range: 0.0 to 1.0.
        Uses the `pystoi` library.

    SI-SNR  (Scale-Invariant Signal-to-Noise Ratio)
        Model-agnostic quality measure.  Higher is better.
        Simple 5-line implementation — also reused as training loss.
        Scale-invariant: immune to overall gain differences.

Usage:
    from sih26052.eval.metrics import pesq_score, stoi_score, si_snr

    p = pesq_score(clean, enhanced, sr=16000)
    s = stoi_score(clean, enhanced, sr=16000)
    snr = si_snr(clean, enhanced)
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── SI-SNR ─────────────────────────────────────────────────────────────────
# This is the simplest metric.  5 lines of math, no external dependencies.
# Also reused as a training loss component (see train/loss.py).

def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

    Parameters
    ----------
    reference : 1-D float array — the clean target
    estimate  : 1-D float array — the enhanced output (same length)

    Returns
    -------
    SI-SNR in dB.  Higher is better.
    For identical signals: ~100+ dB (limited by float precision).
    For uncorrelated signals: ~0 dB.

    Math:
        s_target = <estimate, reference> / ||reference||² · reference
        e_noise  = estimate − s_target
        SI-SNR   = 10 · log10(||s_target||² / ||e_noise||²)

    Why scale-invariant?
        Because s_target is the projection of estimate onto reference,
        it doesn't matter if the estimate is 2× louder.  This makes
        SI-SNR robust to gain mismatches between model output and target.
    """
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)

    # Remove mean (zero-mean SI-SNR variant, standard practice)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    # s_target: projection of estimate onto reference direction
    dot = np.sum(ref * est)
    s_target = (dot / (np.sum(ref ** 2) + 1e-8)) * ref

    # e_noise: everything that isn't the target
    e_noise = est - s_target

    # SI-SNR in dB
    si_snr_val = 10.0 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8)
    )
    return float(si_snr_val)


# ── PESQ wrapper ───────────────────────────────────────────────────────────

def pesq_score(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int = 16000,
) -> float | None:
    """Compute PESQ (wideband) score.

    Returns None if the computation fails (e.g. too short, silence).
    We don't crash on failure because PESQ is fragile with edge cases —
    the harness counts skips separately.
    """
    try:
        from pesq import pesq as _pesq
        # pesq library expects int16-range or float in [-1, 1]
        score = _pesq(sr, reference, degraded, "wb")
        return float(score)
    except Exception as exc:
        logger.debug("PESQ failed: %s", exc)
        return None


# ── STOI wrapper ───────────────────────────────────────────────────────────

def stoi_score(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int = 16000,
) -> float | None:
    """Compute STOI (Short-Time Objective Intelligibility) score.

    Returns None if the computation fails.
    Range: 0.0 (unintelligible) to 1.0 (perfectly clear).
    """
    try:
        from pystoi import stoi as _stoi
        score = _stoi(reference, degraded, sr, extended=False)
        return float(score)
    except Exception as exc:
        logger.debug("STOI failed: %s", exc)
        return None


# ── All-in-one ─────────────────────────────────────────────────────────────

class MetricResult(NamedTuple):
    """Results from computing all three metrics on one pair."""
    pesq: float | None
    stoi: float | None
    si_snr: float


def compute_all_metrics(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int = 16000,
) -> MetricResult:
    """Compute PESQ, STOI, and SI-SNR for a single (reference, degraded) pair.

    Ensures both signals are the same length (truncates the longer one).
    """
    # Length alignment — truncate to shorter
    min_len = min(len(reference), len(degraded))
    ref = reference[:min_len].astype(np.float32)
    deg = degraded[:min_len].astype(np.float32)

    return MetricResult(
        pesq=pesq_score(ref, deg, sr),
        stoi=stoi_score(ref, deg, sr),
        si_snr=si_snr(ref, deg),
    )
