"""
policy.py — AdaptiveMixingPolicy neural network.

A lightweight MLP that predicts mixing parameters to optimize GTCRN training.
This model is TRAINING-ONLY and is NOT deployed on Raspberry Pi.

Architecture:
    Input: concatenation of speech features (8), noise features (10), and
           GTCRN performance features (12) = 30 dimensions
    Shared trunk: 30 → 128 → 64 (ReLU, LayerNorm)
    Heads:
        - SNR head: 64 → N_snrs (categorical over SNR bins)
        - Category head: 64 → N_categories (categorical over noise types)
        - Gain head: 64 → 1 (continuous, sigmoid → [0, 1])
        - Overlap head: 64 → 1 (continuous, sigmoid → [0, 1])

Training: REINFORCE with validation-improvement reward.

Usage:
    policy = AdaptiveMixingPolicy(n_noise_categories=3)
    params = policy.sample(speech_feats, noise_feats, perf_feats)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class MixingParameters(NamedTuple):
    """Output of the AdaptiveMixingPolicy."""
    snr_idx: int                # index into SNR bins
    snr_db: float               # actual SNR value in dB
    category_idx: int           # noise category index
    noise_gain: float           # noise gain multiplier [0, 1]
    overlap_ratio: float        # speech/noise overlap ratio [0, 1]
    log_prob: float             # total log probability (for REINFORCE)


class AdaptiveMixingPolicy(nn.Module):
    """Lightweight MLP that predicts mixing parameters for GTCRN training.

    Parameters
    ----------
    n_noise_categories : int
        Number of noise categories to choose from.
    n_snr_bins : int
        Number of discrete SNR bins.
    snr_values : list[float]
        Actual SNR values corresponding to each bin.
    speech_feat_dim : int
        Dimensionality of speech feature vector.
    noise_feat_dim : int
        Dimensionality of noise feature vector.
    perf_feat_dim : int
        Dimensionality of GTCRN performance feature vector.
    hidden_dim : int
        Hidden layer size.
    """

    def __init__(
        self,
        n_noise_categories: int = 3,
        n_snr_bins: int = 6,
        snr_values: Optional[list[float]] = None,
        speech_feat_dim: int = 8,
        noise_feat_dim: int = 10,
        perf_feat_dim: int = 12,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.n_noise_categories = n_noise_categories
        self.n_snr_bins = n_snr_bins
        self.snr_values = snr_values or [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

        input_dim = speech_feat_dim + noise_feat_dim + perf_feat_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        mid_dim = hidden_dim // 2  # 64

        # Prediction heads
        self.snr_head = nn.Linear(mid_dim, n_snr_bins)
        self.category_head = nn.Linear(mid_dim, n_noise_categories)
        self.gain_head = nn.Linear(mid_dim, 1)
        self.overlap_head = nn.Linear(mid_dim, 1)

        # Initialize with uniform priors
        nn.init.zeros_(self.snr_head.bias)
        nn.init.zeros_(self.category_head.bias)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "AdaptiveMixingPolicy: %d params, %d categories, %d SNR bins",
            total_params, n_noise_categories, n_snr_bins,
        )

    def forward(
        self,
        speech_feats: torch.Tensor,
        noise_feats: torch.Tensor,
        perf_feats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: compute logits and continuous outputs.

        Parameters
        ----------
        speech_feats : (B, 8) — speech acoustic features
        noise_feats  : (B, 10) — noise acoustic features
        perf_feats   : (B, 12) — GTCRN performance features

        Returns
        -------
        dict with:
            snr_logits     : (B, N_snrs)
            cat_logits     : (B, N_cats)
            noise_gain     : (B, 1) — sigmoid output
            overlap_ratio  : (B, 1) — sigmoid output
        """
        x = torch.cat([speech_feats, noise_feats, perf_feats], dim=-1)
        h = self.trunk(x)

        return {
            "snr_logits": self.snr_head(h),
            "cat_logits": self.category_head(h),
            "noise_gain": torch.sigmoid(self.gain_head(h)),
            "overlap_ratio": torch.sigmoid(self.overlap_head(h)),
        }

    def sample(
        self,
        speech_feats: torch.Tensor,
        noise_feats: torch.Tensor,
        perf_feats: torch.Tensor,
    ) -> Tuple[MixingParameters, Dict[str, Any]]:
        """Sample mixing parameters from the policy.

        Uses categorical sampling for SNR and category,
        and the deterministic sigmoid output for gain and overlap.

        Returns both the sampled parameters and the intermediate
        tensors needed for REINFORCE gradient computation.
        """
        outputs = self.forward(speech_feats, noise_feats, perf_feats)

        # Sample SNR
        snr_dist = Categorical(logits=outputs["snr_logits"])
        snr_idx = snr_dist.sample()
        snr_log_prob = snr_dist.log_prob(snr_idx)

        # Sample category
        cat_dist = Categorical(logits=outputs["cat_logits"])
        cat_idx = cat_dist.sample()
        cat_log_prob = cat_dist.log_prob(cat_idx)

        # Continuous outputs (no sampling needed)
        noise_gain = outputs["noise_gain"].squeeze(-1)
        overlap_ratio = outputs["overlap_ratio"].squeeze(-1)

        # Total log probability
        total_log_prob = snr_log_prob + cat_log_prob

        params = MixingParameters(
            snr_idx=int(snr_idx.item()),
            snr_db=self.snr_values[int(snr_idx.item())],
            category_idx=int(cat_idx.item()),
            noise_gain=float(noise_gain.item()),
            overlap_ratio=float(overlap_ratio.item()),
            log_prob=float(total_log_prob.item()),
        )

        intermediates = {
            "snr_log_prob": snr_log_prob,
            "cat_log_prob": cat_log_prob,
            "total_log_prob": total_log_prob,
            "snr_dist": snr_dist,
            "cat_dist": cat_dist,
        }

        return params, intermediates

    def get_action_distribution(
        self,
        speech_feats: torch.Tensor,
        noise_feats: torch.Tensor,
        perf_feats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get the full probability distributions without sampling.

        Useful for analysis and logging.
        """
        outputs = self.forward(speech_feats, noise_feats, perf_feats)
        return {
            "snr_probs": F.softmax(outputs["snr_logits"], dim=-1),
            "cat_probs": F.softmax(outputs["cat_logits"], dim=-1),
            "noise_gain": outputs["noise_gain"],
            "overlap_ratio": outputs["overlap_ratio"],
        }
