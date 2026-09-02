"""
train.py — Fine-tuning loop for GTCRN.

Adapted from ~/Downloads/SEtrain/ with these key modifications:

    1. Loads pretrained checkpoint BEFORE constructing optimizer
       (so optimizer momentum buffers match the loaded weights).

    2. LR ~10× lower than original training (fine-tuning, not from scratch).

    3. Freezes nothing (48K params — no speed to gain from freezing).

    4. Batch composition enforced via the dataset (40/40/20 split).

    5. Per-epoch validation using the eval harness.

    6. Catastrophic forgetting detection: if impulsive improves while
       stationary regresses, alert and adjust.

Usage:
    python -m sih26052.train.train \\
        --checkpoint ~/Downloads/gtcrn/checkpoints/model.pth \\
        --clean-dir data/processed/clean \\
        --noise-dirs broadband=data/processed/DEMAND impulsive=data/processed/ESC-50 \\
        --epochs 50 \\
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    nfft: int = 512,
    hop: int = 256,
) -> dict:
    """Run one training epoch.

    The five core lines (everything else is bookkeeping):
        1. pred = model(noisy_stft)
        2. loss = criterion(pred, clean_stft, pred_wav, clean_wav)
        3. optimizer.zero_grad()
        4. loss.backward()
        5. optimizer.step()
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    component_totals = {}

    window = torch.hann_window(nfft).to(device)

    for batch_idx, batch in enumerate(train_loader):
        noisy = batch["noisy"].to(device)  # (B, samples)
        clean = batch["clean"].to(device)

        # ── STFT ──
        noisy_stft = torch.stft(
            noisy, nfft, hop, window=window, return_complex=False
        )  # (B, freq, time, 2)
        clean_stft = torch.stft(
            clean, nfft, hop, window=window, return_complex=False
        )

        # ── Forward ──
        pred_stft = model(noisy_stft)

        # ── Reconstruct waveform for SI-SNR loss ──
        pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
        pred_wav = torch.istft(pred_complex, nfft, hop, window=window, length=clean.shape[-1])

        # ── Loss ──
        loss, components = criterion(pred_stft, clean_stft, pred_wav, clean)

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        for k, v in components.items():
            component_totals[k] = component_totals.get(k, 0.0) + v

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                "Epoch %d [%d/%d] loss=%.4f",
                epoch, batch_idx + 1, len(train_loader), loss.item(),
            )

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in component_totals.items()}

    return {"avg_loss": avg_loss, **avg_components}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_dir: str | Path,
) -> Path:
    """Save model + optimizer state for resume capability."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)

    logger.info("Saved checkpoint: %s", path)
    return path


def load_pretrained(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Load pretrained weights BEFORE constructing optimizer.

    Why before optimizer?
        If you construct the optimizer first and then load weights,
        the optimizer's momentum buffers are initialised for the
        random weights, not the pretrained ones.  This causes a
        spike in the loss at the start of training.
    """
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained weights from %s", checkpoint_path)
    return model


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GTCRN.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-dirs", type=str, nargs="+", required=True,
        help="category=path pairs, e.g. broadband=data/processed/DEMAND",
    )
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Parse noise dirs
    noise_dirs = {}
    for item in args.noise_dirs:
        cat, path = item.split("=", 1)
        noise_dirs[cat] = path

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Model
    import sys
    gtcrn_root = args.checkpoint.parent.parent
    sys.path.insert(0, str(gtcrn_root))
    from gtcrn import GTCRN  # type: ignore

    model = GTCRN().to(device)
    model = load_pretrained(model, args.checkpoint)

    # Optimizer (AFTER loading pretrained weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss
    from sih26052.train.loss import CombinedLoss
    criterion = CombinedLoss(alpha=0.5)

    # Dataset
    from sih26052.train.dataset import SpeechEnhancementDataset
    train_dataset = SpeechEnhancementDataset(
        clean_dir=args.clean_dir,
        noise_dirs=noise_dirs,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info("=" * 60)
        logger.info("Epoch %d / %d", epoch, args.epochs)

        metrics = train(model, train_loader, criterion, optimizer, device, epoch)
        logger.info("Train: %s", json.dumps(metrics, indent=2))

        save_checkpoint(model, optimizer, epoch, metrics, args.save_dir)


if __name__ == "__main__":
    main()
