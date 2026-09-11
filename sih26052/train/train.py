"""
train.py — Optimized GTCRN training pipeline.

Single, definitive training script for GTCRN speech enhancement.
All other models (DeepFilterNet, FullSubNet, SEtrain) have been removed.

Key training optimizations over baseline:
    1. AdamW optimizer with decoupled weight decay
    2. Cosine Annealing with Warm Restarts (cyclic LR)
    3. Linear warmup for first N epochs
    4. Exponential Moving Average (EMA) of model weights
    5. SpecAugment (frequency + time masking on STFT)
    6. SNR curriculum (easy → hard over epochs)
    7. Multi-resolution STFT loss (512, 1024, 2048)
    8. Gradient accumulation for effective larger batch sizes
    9. Patience-based early stopping
   10. Top-K checkpoint saving

Usage:
    python sih26052/train/train.py                          # defaults
    python sih26052/train/train.py --epochs 100             # long run
    python sih26052/train/train.py --config configs/gtcrn_defense.yaml
    python -m sih26052.train.train --epochs 60 --batch-size 16

CRITICAL: Uses hann_window(512).pow(0.5) to match the original GTCRN pretrained weights.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root and GTCRN model directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GTCRN_DIR = REPO_ROOT / "models" / "gtcrn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GTCRN_DIR) not in sys.path:
    sys.path.insert(0, str(GTCRN_DIR))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = REPO_ROOT.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gtcrn import GTCRN
from sih26052.eval.metrics import si_snr

try:
    from pesq import pesq as compute_pesq
except ImportError:
    compute_pesq = None

try:
    from pystoi import stoi as compute_stoi
except ImportError:
    compute_stoi = None

logger = logging.getLogger("gtcrn_train")

# ── STFT Constants ─────────────────────────────────────────────────────────
# CRITICAL: The original GTCRN was trained with sqrt-Hann window.
# Using standard Hann window creates a distribution mismatch that
# degrades pretrained model performance.
NFFT = 512
HOP = 256
WIN_LEN = 512


def get_window(device: torch.device) -> torch.Tensor:
    """Return the sqrt-Hann window used by the original GTCRN.

    The original infer.py and loss.py both use:
        torch.hann_window(512).pow(0.5)

    This MUST be used for both training and inference to maintain
    compatibility with the pretrained checkpoint.
    """
    return torch.hann_window(WIN_LEN).pow(0.5).to(device)


# ── SpecAugment ────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking on STFT features.

    Prevents overfitting by randomly zeroing out contiguous frequency
    bands and time steps during training. Proven effective for speech
    models even at very small parameter counts.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, stft_real: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment masking.

        Parameters
        ----------
        stft_real : (B, F, T, 2) — STFT in real format

        Returns
        -------
        Masked STFT of same shape
        """
        if not self.training:
            return stft_real

        B, F, T, C = stft_real.shape
        masked = stft_real.clone()

        for _ in range(self.n_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, F), (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            masked[:, f0:f0 + f, :, :] = 0.0

        for _ in range(self.n_time_masks):
            t = torch.randint(0, min(self.time_mask_param, T), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            masked[:, :, t0:t0 + t, :] = 0.0

        return masked


# ── Exponential Moving Average ─────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * current

    Use EMA weights for validation and final checkpoint — they smooth
    out training noise and consistently outperform raw weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy shadow weights into the model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ── Loss Functions ─────────────────────────────────────────────────────────

class GTCRNHybridLoss(nn.Module):
    """Hybrid loss matching the original GTCRN loss.py.

    Components:
        1. Compressed spectral loss: |X|^0.3 magnitude + compressed RI
           Weights: 30*(real_loss + imag_loss) + 70*mag_loss
        2. SI-SNR loss in time domain

    This exactly matches models/gtcrn/loss.py::HybridLoss.
    """

    def __init__(self, compress_power: float = 0.3, lambda_ri: float = 30.0,
                 lambda_mag: float = 70.0, lambda_sisnr: float = 1.0,
                 asym_penalty: float = 0.0, silence_penalty: float = 0.0):
        super().__init__()
        self.c = compress_power
        self.lambda_ri = lambda_ri
        self.lambda_mag = lambda_mag
        self.lambda_sisnr = lambda_sisnr
        self.asym_penalty = asym_penalty
        self.silence_penalty = silence_penalty

    def forward(self, pred_stft: torch.Tensor, true_stft: torch.Tensor,
                window: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        pred_stft : (B, F, T, 2) — predicted STFT [real, imag]
        true_stft : (B, F, T, 2) — target STFT [real, imag]
        window    : sqrt-Hann window for iSTFT reconstruction

        Returns
        -------
        total_loss : scalar tensor
        components : dict with individual loss values
        """
        pred_real = pred_stft[..., 0]
        pred_imag = pred_stft[..., 1]
        true_real = true_stft[..., 0]
        true_imag = true_stft[..., 1]

        # Magnitudes
        pred_mag = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_real ** 2 + true_imag ** 2 + 1e-12)

        # Compressed spectral losses
        mag_loss = nn.functional.mse_loss(pred_mag ** self.c, true_mag ** self.c)
        real_loss = nn.functional.mse_loss(
            pred_real / (pred_mag ** (1 - self.c) + 1e-8),
            true_real / (true_mag ** (1 - self.c) + 1e-8),
        )
        imag_loss = nn.functional.mse_loss(
            pred_imag / (pred_mag ** (1 - self.c) + 1e-8),
            true_imag / (true_mag ** (1 - self.c) + 1e-8),
        )

        sisnr_val = 0.0
        if self.lambda_sisnr > 0.0:
            # SI-SNR in time domain
            y_pred = torch.istft(
                pred_real + 1j * pred_imag, NFFT, HOP, WIN_LEN, window=window,
            )
            y_true = torch.istft(
                true_real + 1j * true_imag, NFFT, HOP, WIN_LEN, window=window,
            )

            # SI-SNR computation (matching original loss.py)
            s_target = (
                torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
                / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + 1e-8)
            )
            sisnr = -torch.log10(
                torch.norm(s_target, dim=-1, keepdim=True) ** 2
                / (torch.norm(y_pred - s_target, dim=-1, keepdim=True) ** 2 + 1e-8)
                + 1e-8
            ).mean()
            sisnr_val = sisnr.item()
            total = self.lambda_ri * (real_loss + imag_loss) + self.lambda_mag * mag_loss + self.lambda_sisnr * sisnr
        else:
            total = self.lambda_ri * (real_loss + imag_loss) + self.lambda_mag * mag_loss

        asym_val = 0.0
        if self.asym_penalty > 0.0:
            # Asymmetric penalty: heavily penalize under-estimating speech (False Negatives -> boosts Recall)
            under_est = torch.relu(true_mag ** self.c - pred_mag ** self.c)
            asym_loss = torch.mean(under_est ** 2) * self.lambda_mag
            total = total + self.asym_penalty * asym_loss
            asym_val = asym_loss.item()

        silence_val = 0.0
        if self.silence_penalty > 0.0:
            # Silence penalty: heavily penalize residual noise leakage during speech pauses (False Positives -> boosts Specificity)
            true_frame_e = torch.mean(true_mag ** 2, dim=1, keepdim=True) + 1e-12
            peak_e = torch.max(true_frame_e, dim=2, keepdim=True)[0]
            silence_mask = (true_frame_e < (peak_e * 2e-4)).float()  # Frames below ~ -37 dB
            silence_leak = (pred_mag ** self.c) * silence_mask
            silence_loss = torch.mean(silence_leak ** 2) * self.lambda_mag
            total = total + self.silence_penalty * silence_loss
            silence_val = silence_loss.item()

        return total, {
            "total": total.item(),
            "ri_loss": (real_loss + imag_loss).item(),
            "mag_loss": mag_loss.item(),
            "sisnr_loss": sisnr_val,
            "asym_loss": asym_val,
            "silence_loss": silence_val,
        }




class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for capturing details at different time-frequency scales.

    Computes spectral convergence + log magnitude loss at multiple FFT sizes.
    Larger FFTs capture low-frequency detail; smaller FFTs capture transients.

    This complements the primary hybrid loss and improves phase estimation.
    """

    def __init__(self, fft_sizes: list[int] = (512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred_wav: torch.Tensor, true_wav: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_wav : (B, T) predicted waveform
        true_wav : (B, T) target waveform

        Returns
        -------
        loss : scalar tensor — averaged multi-resolution loss
        """
        min_len = min(pred_wav.shape[-1], true_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        true_wav = true_wav[..., :min_len]

        total = torch.tensor(0.0, device=pred_wav.device)

        for fft_size in self.fft_sizes:
            hop_size = fft_size // 4
            win = torch.hann_window(fft_size, device=pred_wav.device)

            pred_stft = torch.stft(pred_wav, fft_size, hop_size, fft_size,
                                   window=win, return_complex=True)
            true_stft = torch.stft(true_wav, fft_size, hop_size, fft_size,
                                   window=win, return_complex=True)

            pred_mag = pred_stft.abs()
            true_mag = true_stft.abs()

            # Spectral convergence loss
            sc_loss = torch.norm(true_mag - pred_mag, p="fro") / (torch.norm(true_mag, p="fro") + 1e-8)

            # Log magnitude loss
            log_loss = nn.functional.l1_loss(
                torch.log(pred_mag + 1e-8),
                torch.log(true_mag + 1e-8),
            )

            total = total + sc_loss + log_loss

        return total / len(self.fft_sizes)


# ── SNR Curriculum ─────────────────────────────────────────────────────────

class SNRCurriculum:
    """Curriculum learning for SNR: start easy, get harder.

    Begins training with easier (higher) SNRs and gradually expands to
    include harder (lower) SNR conditions. This helps the model learn
    basic denoising first before tackling extreme noise.
    """

    def __init__(
        self,
        start_range: tuple[float, float] = (5.0, 15.0),
        end_range: tuple[float, float] = (-10.0, 15.0),
        ramp_epochs: int = 10,
    ):
        self.start_low, self.start_high = start_range
        self.end_low, self.end_high = end_range
        self.ramp_epochs = ramp_epochs

    def get_snr_range(self, epoch: int) -> tuple[float, float]:
        """Get the SNR range for the given epoch."""
        if epoch >= self.ramp_epochs:
            return (self.end_low, self.end_high)

        progress = epoch / max(self.ramp_epochs, 1)
        low = self.start_low + (self.end_low - self.start_low) * progress
        high = self.start_high + (self.end_high - self.start_high) * progress
        return (low, high)


# ── Config Loading ─────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = str(REPO_ROOT / "configs" / "gtcrn_defense.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Config file not found: %s. Using defaults.", config_path)
        return {}

    try:
        import yaml
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed. Using defaults.")
        return {}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GTCRN — optimized speech enhancement training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str,
                        default=str(REPO_ROOT / "configs" / "gtcrn_defense.yaml"),
                        help="YAML config file path")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="AdamW weight decay (decoupled)")
    parser.add_argument("--grad-clip", type=float, default=3.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="Virtual dataset size per epoch")

    # Scheduler
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear LR warmup epochs")
    parser.add_argument("--scheduler", type=str, default="cosine_warm_restarts",
                        choices=["cosine", "cosine_warm_restarts"],
                        help="LR scheduler type")
    parser.add_argument("--T0", type=int, default=10,
                        help="CosineAnnealingWarmRestarts T_0 (restart period)")

    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor (0 to disable)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-K checkpoints by PESQ")

    # SpecAugment
    parser.add_argument("--spec-augment", action="store_true", default=True,
                        help="Enable SpecAugment during training")
    parser.add_argument("--no-spec-augment", dest="spec_augment", action="store_false")
    parser.add_argument("--freq-mask-param", type=int, default=15)
    parser.add_argument("--time-mask-param", type=int, default=20)

    # Multi-resolution STFT loss
    parser.add_argument("--multi-res-loss", action="store_true", default=True,
                        help="Enable multi-resolution STFT loss")
    parser.add_argument("--no-multi-res-loss", dest="multi_res_loss", action="store_false")
    parser.add_argument("--multi-res-weight", type=float, default=0.1,
                        help="Weight for multi-resolution STFT loss")

    # Loss warmup
    parser.add_argument("--loss-warmup-epochs", type=int, default=3,
                        help="Epochs of spectral-only loss before adding SI-SNR")

    # SNR curriculum
    parser.add_argument("--snr-curriculum", action="store_true", default=True,
                        help="Enable SNR curriculum (easy→hard)")
    parser.add_argument("--no-snr-curriculum", dest="snr_curriculum", action="store_false")
    parser.add_argument("--snr-ramp-epochs", type=int, default=10,
                        help="Epochs to ramp from easy to full SNR range")

    # Data
    parser.add_argument("--clean-dir", type=str, default=None,
                        help="Clean speech directory or manifest JSON")
    parser.add_argument("--noise-manifest", type=str, default=None,
                        help="Noise pools JSON manifest")
    parser.add_argument("--clean-train-dir", type=str, default=None)
    parser.add_argument("--noisy-train-dir", type=str, default=None)
    parser.add_argument("--clean-val-dir", type=str, default=None)
    parser.add_argument("--noisy-val-dir", type=str, default=None)

    # Mixing
    parser.add_argument("--snr-range", type=float, nargs=2, default=[-10.0, 15.0])
    parser.add_argument("--clean-ratio", type=float, default=0.40)
    parser.add_argument("--noise-ratio", type=float, default=0.60)
    parser.add_argument("--composite-mixing", action="store_true", default=True,
                        help="Use composite defense mixing (Types A-F)")

    # Checkpoint & Fine-Tuning
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"))
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved checkpoint (with optimizer state)")
    parser.add_argument("--finetune-from", type=str, default=None,
                        help="Path to checkpoint for Stage-2 fine-tuning (fresh optimizer/scheduler)")

    # Enhanced Loss & Metric Tuning
    parser.add_argument("--si-snr-weight", type=float, default=1.0,
                        help="Weight for SI-SNR loss (e.g. 0.2 reduces phase distortion during fine-tuning)")
    parser.add_argument("--asym-penalty", type=float, default=0.0,
                        help="Asymmetric penalty on speech under-estimation (boosts Recall/Accuracy above 90%)")
    parser.add_argument("--silence-penalty", type=float, default=0.0,
                        help="Silence penalty on residual noise leakage (boosts Specificity above 85-90%)")
    parser.add_argument("--spectral-floor", type=float, default=0.0,
                        help="Anti-gating spectral floor for streaming codecs (e.g. 0.025 = -32 dB)")

    # Enhanced Data Augmentations & Codec-in-the-loop
    parser.add_argument("--snr-dist", type=str, default="uniform",
                        choices=["uniform", "gaussian"],
                        help="SNR draw distribution ('uniform' or 'gaussian')")
    parser.add_argument("--gain-jitter", action="store_true", default=False,
                        help="Enable dynamic clean speech gain jitter (-32 to -16 dBFS)")
    parser.add_argument("--simulate-reverb", action="store_true", default=False,
                        help="Simulate cabin/room reverberation on clean speech")
    parser.add_argument("--simulate-codec", action="store_true", default=False,
                        help="Simulate lossy streaming codec companding and quantization during training")


    # Output
    parser.add_argument("--save-dir", type=str,
                        default=str(REPO_ROOT / "models" / "checkpoints"))
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name for structured logging")


    # Audio
    parser.add_argument("--crop-samples", type=int, default=32000)

    # Validation & Evaluation
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--eval-snrs", type=float, nargs="+",
                        default=[-10, -5, 0, 5, 10, 15])
    parser.add_argument("--eval-per-snr", action="store_true", default=True)
    parser.add_argument("--eval-per-noise", action="store_true", default=True)
    parser.add_argument("--n-eval-samples", type=int, default=15)

    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ── Model Loading ──────────────────────────────────────────────────────────

def load_gtcrn(checkpoint_path: str | Path | None,
               device: torch.device) -> nn.Module:
    """Load GTCRN model with optional pretrained weights."""
    model = GTCRN().to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info("Loading pretrained weights from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        cleaned = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        logger.info("GTCRN model loaded (%d params)",
                    sum(p.numel() for p in model.parameters()))
    else:
        logger.warning("No checkpoint at %s. Using random initialization.", checkpoint_path)

    return model


def forward_gtcrn(model: nn.Module, noisy_t: torch.Tensor,
                  window: torch.Tensor,
                  spec_augment: Optional[SpecAugment] = None,
                  spectral_floor: float = 0.0,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GTCRN forward pass: waveform → STFT → model → iSTFT → waveform.

    Returns both the enhanced waveform and the predicted STFT (for loss computation).
    """
    orig_len = noisy_t.shape[-1]

    # STFT with sqrt-Hann window (matching original GTCRN)
    stft_c = torch.stft(noisy_t, n_fft=NFFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, return_complex=True)
    stft_real = torch.view_as_real(stft_c)  # (B, F, T, 2)

    # SpecAugment (training only)
    if spec_augment is not None:
        stft_real = spec_augment(stft_real)

    # Model forward
    enh_stft_real = model(stft_real)  # (B, F, T, 2)

    # Anti-gating comfort noise floor for streaming codecs
    if spectral_floor > 0.0:
        enh_stft_real = enh_stft_real + stft_real * spectral_floor

    # iSTFT reconstruction
    enh_stft_c = torch.complex(enh_stft_real[..., 0], enh_stft_real[..., 1])
    enh_t = torch.istft(enh_stft_c, n_fft=NFFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, length=orig_len)

    return enh_t, enh_stft_real



# ── Dataset Construction ───────────────────────────────────────────────────

def build_train_dataset(args, config: dict, snr_range: Optional[tuple] = None):
    """Build training dataset using SpeechEnhancementDataset with on-the-fly mixing."""
    from sih26052.train.dataset import SpeechEnhancementDataset

    actual_snr_range = snr_range if snr_range else tuple(args.snr_range)

    # Try to load from manifests first
    clean_manifest = Path(args.clean_dir) if args.clean_dir else (
        REPO_ROOT / "data" / "manifests" / "clean_train.json"
    )
    noise_manifest = Path(args.noise_manifest) if args.noise_manifest else (
        REPO_ROOT / "data" / "manifests" / "noise_pools.json"
    )

    # Load clean file list
    if clean_manifest.exists() and clean_manifest.suffix == ".json":
        with open(clean_manifest) as f:
            clean_files = json.load(f)
        clean_dir = clean_files
    elif clean_manifest.is_dir():
        clean_dir = str(clean_manifest)
    else:
        clean_dir = str(REPO_ROOT / "Datasets" / "VoiceBank" / "clean_trainset_28spk_wav")

    # Load noise pools
    if noise_manifest.exists():
        with open(noise_manifest) as f:
            noise_pools = json.load(f)
        noise_dirs = noise_pools.get("train", noise_pools)
    else:
        noise_dirs = {}
        processed = REPO_ROOT / "data" / "processed" / "defense_noise" / "train"
        if processed.exists():
            for cat_dir in processed.iterdir():
                if cat_dir.is_dir():
                    noise_dirs[cat_dir.name] = str(cat_dir)
        if not noise_dirs:
            logger.warning("No noise pools found. Using white noise fallback.")
            noise_dirs = {"synthetic": []}

    # Mixing probabilities from config
    mixing_probs = None
    cfg_mix = config.get("mixing", {}).get("probabilities", {})
    if cfg_mix:
        mixing_probs = {
            "A": cfg_mix.get("type_a_clean_background", 0.40),
            "B": cfg_mix.get("type_b_clean_environmental", 0.075),
            "C": cfg_mix.get("type_c_clean_gunfire", 0.075),
            "D": cfg_mix.get("type_d_clean_background_gunfire", 0.225),
            "E": cfg_mix.get("type_e_clean_background_environmental", 0.15),
            "F": cfg_mix.get("type_f_clean_background_multiple_events", 0.075),
        }

    dataset = SpeechEnhancementDataset(
        clean_dir=clean_dir,
        noise_dirs=noise_dirs,
        crop_samples=args.crop_samples,
        snr_range=actual_snr_range,
        composite_mixing=args.composite_mixing,
        mixing_probabilities=mixing_probs,
        clean_ratio=args.clean_ratio,
        noise_ratio=args.noise_ratio,
        n_pairs=args.n_pairs,
        seed=args.seed,
        augment=True,
        snr_dist=getattr(args, "snr_dist", "uniform"),
        gain_jitter=getattr(args, "gain_jitter", False),
        simulate_reverb=getattr(args, "simulate_reverb", False),
        simulate_codec=getattr(args, "simulate_codec", False),
    )

    return dataset


def build_val_dataset(args):
    """Build validation dataset using PairedSpeechDataset (fixed pairs)."""
    from sih26052.data.dccrn_dataset import PairedSpeechDataset

    clean_val = args.clean_val_dir or str(
        REPO_ROOT / "Datasets" / "VoiceBank" / "clean_testset_wav"
    )
    noisy_val = args.noisy_val_dir or str(
        REPO_ROOT / "Datasets" / "VoiceBank" / "noisy_testset_wav"
    )

    return PairedSpeechDataset(
        clean_dir=clean_val,
        noisy_dir=noisy_val,
        crop_samples=args.crop_samples,
        is_train=False,
        max_samples=args.val_samples if args.val_samples > 0 else None,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, val_loader: DataLoader, window: torch.Tensor,
             device: torch.device) -> dict:
    """Evaluate GTCRN on validation data.

    Returns dict with loss, PESQ, STOI, SI-SNR, and SI-SNR improvement.
    """
    model.eval()
    val_losses = []
    pesq_scores = []
    stoi_scores = []
    sisnr_scores = []
    sisnr_input_scores = []
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    noise_supp_scores = []

    from sih26052.eval.metrics import compute_classification_metrics

    criterion = GTCRNHybridLoss().to(device)

    with torch.no_grad():
        for batch in val_loader:
            # Handle both dataset formats
            if isinstance(batch, dict):
                noisy_t = batch["noisy"].to(device)
                clean_t = batch["clean"].to(device)
            else:
                noisy_t, clean_t = batch[0].to(device), batch[1].to(device)

            # Forward (no SpecAugment during eval)
            enh_t, pred_stft = forward_gtcrn(model, noisy_t, window)

            # Clean STFT for loss
            clean_stft_c = torch.stft(clean_t, n_fft=NFFT, hop_length=HOP,
                                      win_length=WIN_LEN, window=window,
                                      return_complex=True)
            clean_stft = torch.view_as_real(clean_stft_c)

            loss, _ = criterion(pred_stft, clean_stft, window)
            val_losses.append(loss.item())

            # Per-sample metrics
            for b in range(clean_t.shape[0]):
                clean_b = clean_t[b].squeeze().cpu().numpy()
                enh_b = enh_t[b].squeeze().cpu().numpy()
                noisy_b = noisy_t[b].squeeze().cpu().numpy()

                min_len = min(len(clean_b), len(enh_b), len(noisy_b))
                clean_b = clean_b[:min_len]
                enh_b = enh_b[:min_len]
                noisy_b = noisy_b[:min_len]

                sisnr_scores.append(si_snr(clean_b, enh_b))
                sisnr_input_scores.append(si_snr(clean_b, noisy_b))

                # Classification & VAD metrics (Accuracy, Recall, Precision, F1)
                clf = compute_classification_metrics(clean_b, enh_b)
                acc_scores.append(clf["accuracy"])
                prec_scores.append(clf["precision"])
                rec_scores.append(clf["recall"])
                f1_scores.append(clf["f1_score"])
                noise_supp_scores.append(clf["noise_suppression"])

                if compute_pesq is not None:
                    try:
                        p = compute_pesq(16000, clean_b, enh_b, "wb")
                        if not np.isnan(p):
                            pesq_scores.append(p)
                    except Exception:
                        pass

                if compute_stoi is not None:
                    try:
                        s = compute_stoi(clean_b, enh_b, 16000)
                        if not np.isnan(s):
                            stoi_scores.append(s)
                    except Exception:
                        pass

    results = {
        "loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "pesq": float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        "stoi": float(np.mean(stoi_scores)) if stoi_scores else 0.0,
        "si_snr": float(np.mean(sisnr_scores)) if sisnr_scores else 0.0,
        "si_snr_input": float(np.mean(sisnr_input_scores)) if sisnr_input_scores else 0.0,
        "accuracy": float(np.mean(acc_scores)) if acc_scores else 0.0,
        "precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
        "recall": float(np.mean(rec_scores)) if rec_scores else 0.0,
        "f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "noise_suppression": float(np.mean(noise_supp_scores)) if noise_supp_scores else 0.0,
    }
    results["si_snr_improvement"] = results["si_snr"] - results["si_snr_input"]

    return results



# ── Checkpoint Manager ─────────────────────────────────────────────────────

class CheckpointManager:
    """Manages top-K checkpoints by score, plus latest checkpoint."""

    def __init__(self, save_dir: Path, top_k: int = 3):
        self.save_dir = save_dir
        self.top_k = top_k
        self.checkpoints: list[tuple[float, Path]] = []  # (score, path)

    def save(self, state: dict, score: float, epoch: int, is_latest: bool = True) -> Optional[Path]:
        """Save checkpoint if it's in top-K. Always saves latest."""
        # Always save latest
        if is_latest:
            latest_path = self.save_dir / "checkpoint_latest.pth"
            torch.save(state, latest_path)

        # Check if in top-K
        if len(self.checkpoints) < self.top_k or score > self.checkpoints[-1][0]:
            ckpt_path = self.save_dir / f"checkpoint_best_epoch{epoch:03d}.pth"
            torch.save(state, ckpt_path)
            self.checkpoints.append((score, ckpt_path))
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Remove excess checkpoints
            while len(self.checkpoints) > self.top_k:
                _, old_path = self.checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()
                    logger.debug("Removed old checkpoint: %s", old_path.name)

            # Also save as checkpoint_best.pth (symlink to actual best)
            best_path = self.save_dir / "checkpoint_best.pth"
            if self.checkpoints:
                torch.save(
                    torch.load(str(self.checkpoints[0][1]), map_location="cpu", weights_only=False),
                    best_path,
                )

            return ckpt_path
        return None

    @property
    def best_score(self) -> float:
        return self.checkpoints[0][0] if self.checkpoints else -float("inf")

    @property
    def best_path(self) -> Optional[Path]:
        return self.checkpoints[0][1] if self.checkpoints else None


# ── LR Scheduler with Warmup ──────────────────────────────────────────────

class WarmupScheduler:
    """Wraps a base scheduler with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup from lr/10 to lr
            warmup_factor = 0.1 + 0.9 * (self.current_epoch / max(self.warmup_epochs, 1))
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * warmup_factor
        else:
            self.base_scheduler.step()

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Main Training Loop ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # Faster convolutions

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load config
    config = load_config(args.config)

    # Experiment directory
    if args.experiment_name:
        save_dir = REPO_ROOT / "experiments" / args.experiment_name
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    exp_config = {
        "model": "GTCRN",
        "model_params": "23.67K",
        "model_mmacs": "33.0",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "T0": args.T0,
        "grad_clip": args.grad_clip,
        "grad_accum_steps": args.grad_accum_steps,
        "ema_decay": args.ema_decay,
        "patience": args.patience,
        "top_k_checkpoints": args.top_k,
        "n_pairs": args.n_pairs,
        "crop_samples": args.crop_samples,
        "snr_range": args.snr_range,
        "snr_curriculum": args.snr_curriculum,
        "snr_ramp_epochs": args.snr_ramp_epochs,
        "clean_ratio": args.clean_ratio,
        "noise_ratio": args.noise_ratio,
        "composite_mixing": args.composite_mixing,
        "spec_augment": args.spec_augment,
        "multi_res_loss": args.multi_res_loss,
        "multi_res_weight": args.multi_res_weight,
        "loss_warmup_epochs": args.loss_warmup_epochs,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "stft": {"nfft": NFFT, "hop": HOP, "win_len": WIN_LEN, "window": "sqrt_hann"},
        "loss": {"type": "HybridLoss + MultiResSTFT", "lambda_ri": 30, "lambda_mag": 70,
                 "compress_power": 0.3},
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # ── 1. Build Datasets ──
    logger.info("Building datasets...")

    # SNR curriculum setup
    snr_curriculum = None
    if args.snr_curriculum:
        snr_curriculum = SNRCurriculum(
            start_range=(5.0, 15.0),
            end_range=tuple(args.snr_range),
            ramp_epochs=args.snr_ramp_epochs,
        )
        initial_snr = snr_curriculum.get_snr_range(1)
        logger.info("SNR curriculum enabled: starting at [%.1f, %.1f] dB, "
                     "ramping to [%.1f, %.1f] dB over %d epochs",
                     initial_snr[0], initial_snr[1],
                     args.snr_range[0], args.snr_range[1], args.snr_ramp_epochs)
    else:
        initial_snr = None

    train_dataset = build_train_dataset(args, config, snr_range=initial_snr)
    val_dataset = build_val_dataset(args)

    # Custom collate for SpeechEnhancementDataset (returns dicts)
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            return {
                "noisy": torch.stack([b["noisy"] for b in batch]),
                "clean": torch.stack([b["clean"] for b in batch]),
                "snr_db": [b["snr_db"] for b in batch],
                "noise_class": [b["noise_class"] for b in batch],
            }
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Train: %d virtual pairs | Val: %d fixed pairs",
                len(train_dataset), len(val_dataset))

    # ── 2. Model & Training Components ──
    window = get_window(device)
    logger.info("STFT window: sqrt-Hann (matching original GTCRN pretrained weights)")

    model_path = args.finetune_from if args.finetune_from else args.checkpoint
    model = load_gtcrn(model_path, device)

    # Loss functions
    criterion = GTCRNHybridLoss(
        lambda_sisnr=getattr(args, "si_snr_weight", 1.0),
        asym_penalty=getattr(args, "asym_penalty", 0.0),
        silence_penalty=getattr(args, "silence_penalty", 0.0),
    ).to(device)
    if getattr(args, "asym_penalty", 0.0) > 0.0:
        logger.info("Asymmetric speech-preservation penalty active (weight=%.2f)", args.asym_penalty)
    if getattr(args, "silence_penalty", 0.0) > 0.0:
        logger.info("Silence suppression penalty active (weight=%.2f)", args.silence_penalty)
    multi_res_criterion = MultiResolutionSTFTLoss().to(device) if args.multi_res_loss else None
    if multi_res_criterion:
        logger.info("Multi-resolution STFT loss enabled (weight=%.2f)", args.multi_res_weight)

    # SpecAugment
    spec_aug = SpecAugment(
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
    ) if args.spec_augment else None
    if spec_aug:
        logger.info("SpecAugment enabled (freq_mask=%d, time_mask=%d)",
                     args.freq_mask_param, args.time_mask_param)

    # Optimizer: AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    logger.info("Optimizer: AdamW (lr=%.1e, weight_decay=%.1e)", args.lr, args.weight_decay)

    # Scheduler with warmup
    if args.scheduler == "cosine_warm_restarts":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T0, T_mult=2, eta_min=args.lr * 0.01,
        )
        logger.info("Scheduler: CosineAnnealingWarmRestarts (T_0=%d, T_mult=2)", args.T0)
    else:
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
        logger.info("Scheduler: CosineAnnealingLR (T_max=%d)", args.epochs)

    scheduler = WarmupScheduler(optimizer, args.warmup_epochs, base_scheduler)
    logger.info("LR warmup: %d epochs (lr/10 → lr)", args.warmup_epochs)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        logger.info("EMA enabled (decay=%.4f)", args.ema_decay)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(save_dir, top_k=args.top_k)
    logger.info("Saving top-%d checkpoints to %s", args.top_k, save_dir)

    # Resume from checkpoint
    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_score" in ckpt:
            best_score = ckpt["best_score"]
            best_epoch = ckpt.get("best_epoch", 0)
        if "ema_state_dict" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema_state_dict"])

    # ── 3. Initial Evaluation ──
    logger.info("Evaluating initial checkpoint...")
    init_val = evaluate(model, val_loader, window, device)
    logger.info(
        "Initial: PESQ=%.4f | STOI=%.4f | SI-SNR=%.2f dB | SI-SNRi=%.2f dB",
        init_val["pesq"], init_val["stoi"], init_val["si_snr"],
        init_val["si_snr_improvement"],
    )
    if best_score == -float("inf"):
        best_score = init_val["pesq"] if init_val["pesq"] > 0 else init_val["si_snr"]

    # ── 4. Training Loop ──
    logger.info("=" * 70)
    logger.info("Starting %d training epochs...", args.epochs)
    logger.info("  Effective batch size: %d (batch=%d × accum=%d)",
                args.batch_size * args.grad_accum_steps, args.batch_size, args.grad_accum_steps)
    logger.info("=" * 70)

    history = []
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if spec_aug is not None:
            spec_aug.train()

        total_loss = 0.0
        loss_components_sum: Dict[str, float] = {}
        n_batches = 0
        t0 = time.time()

        # Update SNR curriculum
        if snr_curriculum is not None:
            current_snr = snr_curriculum.get_snr_range(epoch)
            if hasattr(train_dataset, 'snr_range'):
                train_dataset.snr_range = current_snr
            if epoch <= args.snr_ramp_epochs:
                logger.info("  SNR range: [%.1f, %.1f] dB", current_snr[0], current_snr[1])

        # Loss warmup: spectral-only for first N epochs
        use_sisnr = (epoch > args.loss_warmup_epochs)
        criterion.lambda_sisnr = getattr(args, "si_snr_weight", 1.0) if use_sisnr else 0.0
        if not use_sisnr and epoch == 1:
            logger.info("  Loss warmup: spectral-only (SI-SNR disabled for %d epochs)",
                        args.loss_warmup_epochs)

        max_batches = (
            args.max_batches_per_epoch if args.max_batches_per_epoch > 0
            else len(train_loader)
        )
        pbar = tqdm(train_loader, total=min(max_batches, len(train_loader)),
                    desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            # Handle both dataset formats
            if isinstance(batch, dict):
                noisy_t = batch["noisy"].to(device)
                clean_t = batch["clean"].to(device)
            else:
                noisy_t = batch[0].to(device)
                clean_t = batch[1].to(device)

            # Forward pass with SpecAugment and comfort noise floor
            enh_t, pred_stft = forward_gtcrn(
                model, noisy_t, window,
                spec_augment=spec_aug,
                spectral_floor=getattr(args, "spectral_floor", 0.0),
            )

            # Clean STFT for loss
            clean_stft_c = torch.stft(clean_t, n_fft=NFFT, hop_length=HOP,
                                      win_length=WIN_LEN, window=window,
                                      return_complex=True)
            clean_stft = torch.view_as_real(clean_stft_c)

            # Primary loss (lambda_sisnr dynamically scaled)
            loss, components = criterion(pred_stft, clean_stft, window)

            # Multi-resolution STFT loss
            if multi_res_criterion is not None:
                mr_loss = multi_res_criterion(enh_t, clean_t)
                loss = loss + args.multi_res_weight * mr_loss
                components["multi_res_loss"] = mr_loss.item()

            # Scale loss for gradient accumulation
            loss = loss / args.grad_accum_steps
            loss.backward()

            # Gradient step (accumulation-aware)
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) >= max_batches:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item() * args.grad_accum_steps  # un-scale for logging
            n_batches += 1

            for k, v in components.items():
                loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v

            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum_steps:.4f}",
                "avg": f"{(total_loss / n_batches):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        epoch_time = time.time() - t0

        avg_components = {
            k: v / max(n_batches, 1)
            for k, v in loss_components_sum.items()
        }

        # ── Validation (with EMA weights if available) ──
        original_state = None
        if ema is not None:
            # Save original weights, apply EMA for validation
            original_state = copy.deepcopy(model.state_dict())
            ema.apply(model)

        val_res = evaluate(model, val_loader, window, device)

        if ema is not None and original_state is not None:
            # Restore original weights after validation
            model.load_state_dict(original_state)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch [%2d/%2d] (%.1fs) lr=%.1e | Loss: %.4f "
            "| Val PESQ: %.4f | Val STOI: %.4f | Val SI-SNR: %.2f dB (Δ%.2f dB) "
            "| Acc: %.1f%% | Rec: %.3f | Prec: %.3f | F1: %.3f",
            epoch, args.epochs, epoch_time, current_lr, train_loss,
            val_res["pesq"], val_res["stoi"], val_res["si_snr"],
            val_res["si_snr_improvement"],
            val_res.get("accuracy", 0.0), val_res.get("recall", 0.0),
            val_res.get("precision", 0.0), val_res.get("f1_score", 0.0),
        )
        logger.info(
            "  Components: RI=%.4f | Mag=%.4f | SI-SNR=%.4f%s",
            avg_components.get("ri_loss", 0), avg_components.get("mag_loss", 0),
            avg_components.get("sisnr_loss", 0),
            f" | MultiRes={avg_components.get('multi_res_loss', 0):.4f}" if multi_res_criterion else "",
        )

        # History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ri_loss": avg_components.get("ri_loss", 0),
            "train_mag_loss": avg_components.get("mag_loss", 0),
            "train_sisnr_loss": avg_components.get("sisnr_loss", 0),
            "train_multi_res_loss": avg_components.get("multi_res_loss", 0),
            "val_loss": val_res["loss"],
            "val_pesq": val_res["pesq"],
            "val_stoi": val_res["stoi"],
            "val_si_snr": val_res["si_snr"],
            "val_si_snr_input": val_res["si_snr_input"],
            "val_si_snr_improvement": val_res["si_snr_improvement"],
            "val_accuracy": val_res.get("accuracy", 0.0),
            "val_precision": val_res.get("precision", 0.0),
            "val_recall": val_res.get("recall", 0.0),
            "val_f1_score": val_res.get("f1_score", 0.0),
            "val_noise_suppression": val_res.get("noise_suppression", 0.0),
            "lr": current_lr,
            "epoch_time_s": epoch_time,
        })

        # ── Checkpoint ──
        score = val_res["pesq"] if val_res["pesq"] > 0 else val_res["si_snr"]

        ckpt_state = {
            "epoch": epoch,
            "best_score": max(score, ckpt_manager.best_score),
            "best_epoch": epoch if score > ckpt_manager.best_score else best_epoch,
            "val_pesq": val_res["pesq"],
            "val_stoi": val_res["stoi"],
            "val_si_snr": val_res["si_snr"],
            "val_si_snr_improvement": val_res["si_snr_improvement"],
            "val_accuracy": val_res.get("accuracy", 0.0),
            "val_precision": val_res.get("precision", 0.0),
            "val_recall": val_res.get("recall", 0.0),
            "val_f1_score": val_res.get("f1_score", 0.0),
            "val_noise_suppression": val_res.get("noise_suppression", 0.0),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": exp_config,
        }
        if ema is not None:
            ckpt_state["ema_state_dict"] = ema.state_dict()

        saved_path = ckpt_manager.save(ckpt_state, score, epoch)
        if saved_path is not None:
            best_epoch = epoch
            patience_counter = 0
            logger.info(
                "🔥 New best at epoch %d (score: %.4f) → %s",
                epoch, score, saved_path.name,
            )
            if ema is not None:
                ema_save_path = save_dir / "best_ema.pt"
                torch.save({
                    "model_state_dict": copy.deepcopy(ema.shadow),
                    "epoch": epoch,
                    "best_score": score,
                    "val_pesq": val_res["pesq"],
                    "val_stoi": val_res["stoi"],
                    "val_si_snr": val_res["si_snr"],
                    "val_accuracy": val_res.get("accuracy", 0.0),
                    "val_precision": val_res.get("precision", 0.0),
                    "val_recall": val_res.get("recall", 0.0),
                    "val_f1_score": val_res.get("f1_score", 0.0),
                    "config": exp_config,
                }, ema_save_path)
        else:

            patience_counter += 1

        # ── Early Stopping ──
        if patience_counter >= args.patience:
            logger.info(
                "⏹️  Early stopping triggered after %d epochs without improvement "
                "(best epoch: %d, best score: %.4f)",
                args.patience, best_epoch, ckpt_manager.best_score,
            )
            break

    # ── 5. Final Evaluation ──
    logger.info("=" * 70)
    logger.info("Training complete. Best epoch: %d (score: %.4f)",
                best_epoch, ckpt_manager.best_score)

    # Load best checkpoint for final evaluation
    best_path = save_dir / "checkpoint_best.pth"
    if best_path.exists():
        best_ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        # Apply EMA if saved
        if "ema_state_dict" in best_ckpt:
            temp_ema = EMA(model, decay=args.ema_decay)
            temp_ema.load_state_dict(best_ckpt["ema_state_dict"])
            temp_ema.apply(model)

    final_res = evaluate(model, val_loader, window, device)
    logger.info(
        "Final Best: PESQ=%.4f | STOI=%.4f | SI-SNR=%.2f dB | SI-SNRi=%.2f dB",
        final_res["pesq"], final_res["stoi"], final_res["si_snr"],
        final_res["si_snr_improvement"],
    )

    # ── Per-SNR & Per-Noise Evaluation ──
    per_snr_results = None
    per_noise_results = None

    def enhance_fn(noisy_wav: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t_noisy = torch.from_numpy(noisy_wav.astype(np.float32)).unsqueeze(0).to(device)
            enh_t, _ = forward_gtcrn(model, t_noisy, window)
            return enh_t.squeeze(0).cpu().numpy()

    clean_manifest = REPO_ROOT / "data" / "manifests" / "clean_train.json"
    noise_manifest = REPO_ROOT / "data" / "manifests" / "noise_pools.json"

    if clean_manifest.exists() and noise_manifest.exists():
        try:
            with open(clean_manifest) as f:
                c_files = json.load(f)
            with open(noise_manifest) as f:
                n_pools = json.load(f)
            test_noise_pools = n_pools.get("test", n_pools.get("train", {}))
            all_test_noises = [p for pool in test_noise_pools.values() for p in pool]

            if args.eval_per_snr and all_test_noises:
                from sih26052.eval.per_snr_eval import evaluate_per_snr
                logger.info("Running Per-SNR Evaluation...")
                per_snr_results = evaluate_per_snr(
                    enhance_fn=enhance_fn,
                    clean_files=c_files[:100],
                    noise_files=all_test_noises,
                    snr_levels=args.eval_snrs,
                    n_samples_per_snr=args.n_eval_samples,
                    crop_samples=args.crop_samples,
                )
                with open(save_dir / "per_snr_results.json", "w") as f:
                    json.dump({str(k): v for k, v in per_snr_results.items()}, f, indent=2)

            if args.eval_per_noise and test_noise_pools:
                from sih26052.eval.per_noise_eval import evaluate_per_noise_type
                logger.info("Running Per-Noise-Type Evaluation...")
                per_noise_results = evaluate_per_noise_type(
                    enhance_fn=enhance_fn,
                    clean_files=c_files[:100],
                    noise_pools=test_noise_pools,
                    snr_db=0.0,
                    n_samples_per_type=args.n_eval_samples,
                    crop_samples=args.crop_samples,
                )
                with open(save_dir / "per_noise_results.json", "w") as f:
                    json.dump(per_noise_results, f, indent=2)
        except Exception as e:
            logger.warning("Per-SNR / Per-noise evaluation error: %s", e)

    # ── 6. Save Report ──
    report = {
        "model": "GTCRN",
        "model_params": "23.67K",
        "best_epoch": best_epoch,
        "best_score": ckpt_manager.best_score,
        "total_epochs_trained": epoch,
        "early_stopped": patience_counter >= args.patience,
        "initial_metrics": init_val,
        "final_metrics": final_res,
        "improvement": {
            "pesq": final_res["pesq"] - init_val["pesq"],
            "stoi": final_res["stoi"] - init_val["stoi"],
            "si_snr": final_res["si_snr"] - init_val["si_snr"],
        },
        "training_config": exp_config,
        "per_snr": {str(k): v for k, v in per_snr_results.items()} if per_snr_results else None,
        "per_noise": per_noise_results,
        "history": history,
    }
    report_path = save_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
