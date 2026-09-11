#!/usr/bin/env python3
"""
finetune_gtcrn.py — Stage-2 Perceptual Fine-Tuning for GTCRN.

Takes the best checkpoint from Stage-1 training and applies:
1. Low, non-destructive learning rate (2e-5) with cosine annealing.
2. Perceptual harmonic-preserving loss:
   - Reduced time-domain SI-SNR loss weight (0.2) to eliminate phase distortion.
   - Boosted multi-resolution STFT loss (0.4) to reconstruct speech formants.
3. Advanced Defense acoustic augmentations:
   - Truncated Gaussian SNR sampling (+3 dB mean, 5 dB std) to avoid over-suppression.
   - Dynamic speech gain jitter (-32 to -16 dBFS) for speaker distance invariance.
   - Synthetic room impulse response (early reflections / vehicle cabin acoustics).
4. Saves EMA smoothed weights (best_ema.pt) for peak PESQ/STOI evaluation.

Usage:
    python scripts/finetune_gtcrn.py
    python scripts/finetune_gtcrn.py --checkpoint experiments/gtcrn_optimized_v1/checkpoint_best.pth --epochs 15
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GTCRN_DIR = REPO_ROOT / "models" / "gtcrn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if GTCRN_DIR.exists() and str(GTCRN_DIR) not in sys.path:
    sys.path.insert(0, str(GTCRN_DIR))

from sih26052.train.train import main as train_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("finetune_gtcrn")


def parse_finetune_args():
    parser = argparse.ArgumentParser(
        description="Stage-2 Perceptual Fine-Tuning for GTCRN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best.pth"),
        help="Base checkpoint to fine-tune from",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="Gentle peak learning rate")
    parser.add_argument("--si-snr-weight", type=float, default=0.2, help="SI-SNR loss weight")
    parser.add_argument("--multi-res-weight", type=float, default=0.4, help="Multi-res STFT loss weight")
    parser.add_argument("--snr-dist", type=str, default="gaussian", choices=["uniform", "gaussian"])
    parser.add_argument("--simulate-reverb", action="store_true", default=True)
    parser.add_argument("--no-reverb", dest="simulate_reverb", action="store_false")
    parser.add_argument("--gain-jitter", action="store_true", default=True)
    parser.add_argument("--no-gain-jitter", dest="gain_jitter", action="store_false")
    parser.add_argument("--simulate-codec", action="store_true", default=True, help="Simulate streaming codec compression")
    parser.add_argument("--no-codec", dest="simulate_codec", action="store_false")
    parser.add_argument("--asym-penalty", type=float, default=2.5, help="Asymmetric penalty on speech under-estimation (boosts Recall/Accuracy > 90%)")
    parser.add_argument("--silence-penalty", type=float, default=2.0, help="Silence penalty on residual noise leakage (boosts Specificity > 85-90%)")
    parser.add_argument("--spectral-floor", type=float, default=0.025, help="Anti-gating comfort noise floor for streaming codecs")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="gtcrn_stage3_final",
        help="Experiment name for logging and output",
    )
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--n-pairs", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_finetune_args()
    ckpt_path = Path(args.checkpoint)

    # If checkpoint doesn't exist yet, look for checkpoint_best_epoch*.pth or best available
    if not ckpt_path.exists():
        exp_dir = REPO_ROOT / "experiments" / "gtcrn_optimized_v1"
        if exp_dir.exists():
            candidates = sorted(exp_dir.glob("checkpoint_best*.pth"))
            if candidates:
                ckpt_path = candidates[-1]
                logger.info("Using candidate checkpoint: %s", ckpt_path)

    logger.info("=" * 70)
    logger.info("🚀 Launching GTCRN Stage-3 Perceptual & Silence Fine-Tuning")
    logger.info("  Base checkpoint:       %s", ckpt_path)
    logger.info("  Epochs:                %d", args.epochs)
    logger.info("  Learning rate:         %.1e (Cosine decay)", args.lr)
    logger.info("  SI-SNR weight:         %.2f", args.si_snr_weight)
    logger.info("  Multi-Res weight:      %.2f", args.multi_res_weight)
    logger.info("  Asymmetric Penalty:    %.2f (heavily penalizes FN speech clipping)", args.asym_penalty)
    logger.info("  Silence Penalty:       %.2f (heavily penalizes FP noise in pauses)", args.silence_penalty)
    logger.info("  Spectral Noise Floor:  %.3f (anti-gating comfort noise for codecs)", args.spectral_floor)
    logger.info("  Simulate Codec:        %s", args.simulate_codec)
    logger.info("  SNR Distribution:      %s", args.snr_dist)
    logger.info("  Gain Jitter:           %s", args.gain_jitter)
    logger.info("  Simulate Reverb:       %s", args.simulate_reverb)
    logger.info("=" * 70)

    # Build argv for train.py
    cli_args = [
        "train.py",
        "--finetune-from", str(ckpt_path),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--scheduler", "cosine",
        "--warmup-epochs", "1",
        "--weight-decay", "0.01",
        "--si-snr-weight", str(args.si_snr_weight),
        "--asym-penalty", str(args.asym_penalty),
        "--silence-penalty", str(args.silence_penalty),
        "--spectral-floor", str(args.spectral_floor),
        "--multi-res-loss",
        "--multi-res-weight", str(args.multi_res_weight),
        "--loss-warmup-epochs", "0",
        "--snr-dist", args.snr_dist,
        "--experiment-name", args.experiment_name,
        "--n-pairs", str(args.n_pairs),
        "--val-samples", str(args.val_samples),
    ]

    if args.gain_jitter:
        cli_args.append("--gain-jitter")
    if args.simulate_reverb:
        cli_args.append("--simulate-reverb")
    if args.simulate_codec:
        cli_args.append("--simulate-codec")

    # Overwrite sys.argv and call train main
    sys.argv = cli_args
    train_main()



if __name__ == "__main__":
    main()
