"""
reward.py — Reward computation for the AdaptiveMixingPolicy.

The reward measures how much the GTCRN model improves on held-out
validation data after training on the adaptive mixer's chosen distribution.

Reward design:
    - Positive reward for validation SI-SDR improvement
    - Bonus for improving on hard examples (low-SNR, gunfire)
    - Penalty for degrading performance on previously strong areas
    - Baseline subtraction for variance reduction

Usage:
    from sih26052.adaptive_mixer.reward import compute_reward, RewardTracker

    tracker = RewardTracker()
    reward = compute_reward(current_metrics, previous_metrics, tracker)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardTracker:
    """Running baseline for REINFORCE variance reduction.

    Maintains an exponential moving average of past rewards
    to subtract from the raw reward signal.
    """

    def __init__(self, ema_alpha: float = 0.1, window_size: int = 50):
        self.ema_alpha = ema_alpha
        self.baseline = 0.0
        self.reward_history: deque = deque(maxlen=window_size)
        self.n_updates = 0

    def update(self, reward: float) -> float:
        """Update baseline and return the advantage (reward - baseline)."""
        self.reward_history.append(reward)
        self.n_updates += 1

        if self.n_updates == 1:
            self.baseline = reward
        else:
            self.baseline = (
                self.ema_alpha * reward + (1 - self.ema_alpha) * self.baseline
            )

        advantage = reward - self.baseline
        return advantage

    @property
    def mean_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history))

    @property
    def std_reward(self) -> float:
        if len(self.reward_history) < 2:
            return 1.0
        return float(np.std(self.reward_history)) + 1e-8


def compute_reward(
    current_metrics: Dict[str, float],
    previous_metrics: Optional[Dict[str, float]] = None,
    per_snr_metrics: Optional[Dict[float, Dict[str, float]]] = None,
    per_noise_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute reward for the adaptive mixing policy.

    The reward encourages the policy to find training distributions
    that maximize GTCRN's generalization on validation data.

    Parameters
    ----------
    current_metrics : dict with keys like 'si_snr', 'pesq', 'stoi'
    previous_metrics : metrics from the previous policy update step
    per_snr_metrics : optional per-SNR breakdown
    per_noise_metrics : optional per-noise-type breakdown
    weights : optional weights for different reward components

    Returns
    -------
    reward : float — scalar reward value
    """
    if weights is None:
        weights = {
            "si_snr_improvement": 0.4,   # Main objective
            "pesq": 0.3,                  # Perceptual quality
            "stoi": 0.2,                  # Intelligibility
            "hard_example_bonus": 0.1,    # Bonus for hard examples
        }

    reward = 0.0

    # ── 1. SI-SNR Improvement Component ──
    if previous_metrics is not None:
        si_snr_delta = current_metrics.get("si_snr", 0) - previous_metrics.get("si_snr", 0)
        reward += weights["si_snr_improvement"] * si_snr_delta
    else:
        # Absolute SI-SNR (normalized)
        reward += weights["si_snr_improvement"] * (
            current_metrics.get("si_snr_improvement", 0) / 10.0
        )

    # ── 2. PESQ Component ──
    pesq = current_metrics.get("pesq", 0)
    if pesq > 0:
        reward += weights["pesq"] * (pesq - 2.0) / 2.5  # Normalize PESQ to ~[-1, 1]

    # ── 3. STOI Component ──
    stoi = current_metrics.get("stoi", 0)
    if stoi > 0:
        reward += weights["stoi"] * (stoi - 0.5) * 2.0  # Normalize STOI to ~[-1, 1]

    # ── 4. Hard Example Bonus ──
    if per_snr_metrics is not None:
        # Bonus for improving on low-SNR conditions
        hard_snrs = [-10.0, -5.0, 0.0]
        hard_improvements = []
        for snr in hard_snrs:
            snr_data = per_snr_metrics.get(snr, {})
            si_snri = snr_data.get("si_snr_improvement", 0)
            hard_improvements.append(si_snri)

        if hard_improvements:
            hard_bonus = np.mean(hard_improvements) / 10.0
            reward += weights["hard_example_bonus"] * hard_bonus

    if per_noise_metrics is not None:
        # Bonus for improving on gunfire (typically hardest)
        gunfire_data = per_noise_metrics.get("gunfire", {})
        gunfire_si_snri = gunfire_data.get("si_snr_improvement", 0)
        reward += weights["hard_example_bonus"] * 0.5 * gunfire_si_snri / 10.0

    return float(reward)


def compute_hard_example_score(
    per_snr_metrics: Dict[float, Dict[str, float]],
    per_noise_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Identify where GTCRN is weakest — used to guide the mixing policy.

    Returns a dict mapping (snr_bin, noise_category) → difficulty score.
    Higher score = GTCRN performs worse = needs more training examples.
    """
    scores = {}

    # Per-SNR difficulty
    for snr, metrics in per_snr_metrics.items():
        si_snri = metrics.get("si_snr_improvement", 0)
        # Lower improvement = harder = higher difficulty score
        scores[f"snr_{snr}"] = max(0.0, 5.0 - si_snri)

    # Per-noise difficulty
    for noise_type, metrics in per_noise_metrics.items():
        si_snri = metrics.get("si_snr_improvement", 0)
        scores[f"noise_{noise_type}"] = max(0.0, 5.0 - si_snri)

    return scores


def compute_curriculum_level(
    epoch: int,
    total_epochs: int,
    current_metrics: Dict[str, float],
) -> float:
    """Compute a curriculum difficulty level [0, 1].

    0 = easy (high SNR, simple noise)
    1 = hard (low SNR, complex composite noise)

    Progression: starts easy, increases with epochs and model capability.
    """
    # Time-based progression (sigmoid warmup)
    progress = epoch / max(total_epochs, 1)
    time_level = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.3)))

    # Performance-based adjustment
    si_snri = current_metrics.get("si_snr_improvement", 0)
    if si_snri > 10.0:
        # Model is doing well, push harder
        perf_bonus = 0.1
    elif si_snri < 2.0:
        # Model is struggling, ease up
        perf_bonus = -0.1
    else:
        perf_bonus = 0.0

    return float(np.clip(time_level + perf_bonus, 0.0, 1.0))
