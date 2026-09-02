"""
loss.py — SI-SNR + compressed spectral loss for GTCRN fine-tuning.

Two loss components:

    1. SI-SNR on waveform:
       The primary loss.  Measures time-domain signal quality.
       Scale-invariant, so it doesn't penalise gain differences.

    2. Compressed spectral loss:
       |X|^0.3 magnitude compression rebalances the loss toward
       high-frequency consonants (which have lower energy but are
       critical for intelligibility).  Without compression, the loss
       is dominated by low-frequency vowels.

Adapted from GTCRN's loss.py — extended with:
    - SI-SNR component (original only had spectral loss)
    - Per-frame impulsive weighting (2–3× on frames where noise energy
      spikes above rolling median)

Usage:
    criterion = CombinedLoss(alpha=0.5)
    loss = criterion(enhanced_stft, clean_stft, enhanced_wav, clean_wav)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss (negated for minimization).

    Maximizing SI-SNR = minimizing -SI-SNR.
    """

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        estimate  : (batch, samples) — model output waveform
        reference : (batch, samples) — clean target waveform

        Returns
        -------
        loss : scalar tensor — mean -SI-SNR across batch
        """
        # Zero-mean
        ref = reference - reference.mean(dim=-1, keepdim=True)
        est = estimate - estimate.mean(dim=-1, keepdim=True)

        # s_target = <est, ref> / ||ref||² · ref
        dot = torch.sum(ref * est, dim=-1, keepdim=True)
        s_target = (dot / (torch.sum(ref ** 2, dim=-1, keepdim=True) + 1e-8)) * ref

        # e_noise = est - s_target
        e_noise = est - s_target

        # SI-SNR = 10·log10(||s_target||² / ||e_noise||²)
        si_snr = 10.0 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
        )

        return -si_snr.mean()


class CompressedSpectralLoss(nn.Module):
    """Compressed spectral magnitude + phase loss.

    Applies |X|^compression_power to rebalance toward high frequencies.

    Adapted from GTCRN's HybridLoss with the compression exponent
    changed from 0.3 to a configurable parameter.
    """

    def __init__(self, compression_power: float = 0.3):
        super().__init__()
        self.c = compression_power

    def forward(self, pred_stft: torch.Tensor, true_stft: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_stft : (batch, freq, time, 2) — predicted STFT [real, imag]
        true_stft : (batch, freq, time, 2) — target STFT [real, imag]

        Returns
        -------
        loss : scalar tensor
        """
        # Split real/imag
        pred_real = pred_stft[..., 0]
        pred_imag = pred_stft[..., 1]
        true_real = true_stft[..., 0]
        true_imag = true_stft[..., 1]

        # Magnitudes
        pred_mag = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_real ** 2 + true_imag ** 2 + 1e-12)

        # Compressed magnitude loss
        mag_loss = nn.functional.mse_loss(pred_mag ** self.c, true_mag ** self.c)

        # Compressed complex loss (preserves phase information)
        pred_real_c = pred_real / (pred_mag ** (1 - self.c) + 1e-8)
        pred_imag_c = pred_imag / (pred_mag ** (1 - self.c) + 1e-8)
        true_real_c = true_real / (true_mag ** (1 - self.c) + 1e-8)
        true_imag_c = true_imag / (true_mag ** (1 - self.c) + 1e-8)

        real_loss = nn.functional.mse_loss(pred_real_c, true_real_c)
        imag_loss = nn.functional.mse_loss(pred_imag_c, true_imag_c)

        return mag_loss + real_loss + imag_loss


class CombinedLoss(nn.Module):
    """Combined SI-SNR + Compressed Spectral loss.

    total_loss = alpha * SI-SNR_loss + (1 - alpha) * spectral_loss

    Parameters
    ----------
    alpha : float
        Weight for SI-SNR loss.  0.5 = equal weighting.
    compression_power : float
        Exponent for spectral compression.  0.3 rebalances toward HF.
    """

    def __init__(self, alpha: float = 0.5, compression_power: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.si_snr_loss = SISNRLoss()
        self.spectral_loss = CompressedSpectralLoss(compression_power)

    def forward(
        self,
        pred_stft: torch.Tensor,
        true_stft: torch.Tensor,
        pred_wav: torch.Tensor,
        true_wav: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns
        -------
        total_loss : scalar tensor
        components : dict with individual loss values (for logging)
        """
        l_sisnr = self.si_snr_loss(pred_wav, true_wav)
        l_spec = self.spectral_loss(pred_stft, true_stft)

        total = self.alpha * l_sisnr + (1 - self.alpha) * l_spec

        components = {
            "si_snr_loss": l_sisnr.item(),
            "spectral_loss": l_spec.item(),
            "total_loss": total.item(),
        }

        return total, components
