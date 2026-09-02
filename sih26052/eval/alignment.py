"""
alignment.py — Cross-correlation delay finder.

When comparing model output against the clean reference, the output may
be shifted by a constant number of samples due to STFT windowing.  If we
don't correct for this, PESQ and SI-SNR will be systematically wrong.

This module finds and applies that shift.

How it works:
    1. Cross-correlate the clean and enhanced signals.
    2. The lag at the peak of the cross-correlation is the delay.
    3. Shift the enhanced signal by that many samples.

Why cross-correlation and not just counting STFT hops?
    Because the actual delay depends on the OLA implementation details
    (window type, padding, etc.).  Measuring it empirically is more robust
    than computing it from first principles and getting it wrong.

Usage:
    from sih26052.eval.alignment import find_delay, align_signals

    delay = find_delay(reference, enhanced)
    enhanced_aligned = align_signals(reference, enhanced)
"""
from __future__ import annotations

import numpy as np


def find_delay(
    reference: np.ndarray,
    estimate: np.ndarray,
    max_delay_samples: int = 1024,
) -> int:
    """Find the sample delay between *reference* and *estimate*.

    Parameters
    ----------
    reference : 1-D clean signal
    estimate  : 1-D enhanced signal (potentially shifted)
    max_delay_samples : maximum expected delay (limits search range)

    Returns
    -------
    delay : int
        Positive means estimate is delayed relative to reference.
        Negative means estimate is ahead.

    Implementation:
        Uses numpy's correlate in 'full' mode, restricted to
        ±max_delay_samples to avoid spurious matches far from zero lag.
    """
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)

    # Truncate both to same length for correlation
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    # Normalise to prevent overflow in large signals
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    est = est / (np.max(np.abs(est)) + 1e-8)

    # Full cross-correlation
    corr = np.correlate(ref, est, mode="full")
    # corr has length 2*min_len - 1
    # Zero-lag is at index min_len - 1

    zero_lag_idx = min_len - 1

    # Restrict search to ±max_delay_samples around zero lag
    search_start = max(0, zero_lag_idx - max_delay_samples)
    search_end = min(len(corr), zero_lag_idx + max_delay_samples + 1)

    search_region = corr[search_start:search_end]
    peak_in_region = np.argmax(search_region)
    peak_idx = search_start + peak_in_region

    # Delay = peak_idx - zero_lag_idx
    # Positive: estimate is delayed (need to shift it left)
    delay = peak_idx - zero_lag_idx
    return int(delay)


def align_signals(
    reference: np.ndarray,
    estimate: np.ndarray,
    max_delay_samples: int = 1024,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align *estimate* to *reference* by compensating for sample delay.

    Returns
    -------
    ref_aligned : reference, possibly truncated
    est_aligned : estimate, shifted and truncated to match
    delay       : the detected delay in samples

    Both outputs have the same length.
    """
    delay = find_delay(reference, estimate, max_delay_samples)

    if delay > 0:
        # Estimate is delayed → trim the first `delay` samples from estimate
        # and the last `delay` samples from reference
        est_aligned = estimate[delay:]
        ref_aligned = reference[:len(est_aligned)]
    elif delay < 0:
        # Estimate is ahead → trim the first `|delay|` samples from reference
        abs_delay = abs(delay)
        ref_aligned = reference[abs_delay:]
        est_aligned = estimate[:len(ref_aligned)]
    else:
        # No delay
        min_len = min(len(reference), len(estimate))
        ref_aligned = reference[:min_len]
        est_aligned = estimate[:min_len]

    # Final length match (safety)
    min_len = min(len(ref_aligned), len(est_aligned))
    return ref_aligned[:min_len], est_aligned[:min_len], delay
