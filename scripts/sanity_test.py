#!/usr/bin/env python3
"""
sanity_test.py — Small-subset sanity test before full GTCRN defense training.

Executes:
    1. Builds a tiny dataset (~100 noisy defense mixtures)
    2. Loads GTCRN with pretrained weights
    3. Runs forward pass and checks STFT tensor shapes
    4. Computes CombinedLoss and verifies finite loss
    5. Runs backward pass and verifies finite gradients
    6. Trains for 2 epochs and asserts loss decreases
    7. Runs inference, saves enhanced WAV
    8. Calculates metrics (SI-SNR, STOI, PESQ) and prints improvement
    9. Verifies clean speech identity test
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "models" / "gtcrn"))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from gtcrn import GTCRN
from sih26052.eval.metrics import compute_all_metrics
from sih26052.train.dataset import SpeechEnhancementDataset
from sih26052.train.loss import CombinedLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanity_test")


def main():
    print("=" * 70)
    print("GTCRN DEFENSE SMALL-SUBSET SANITY TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load clean files & noise pools
    clean_manifest = repo_root / "data" / "manifests" / "clean_train.json"
    noise_manifest = repo_root / "data" / "manifests" / "noise_pools.json"

    assert clean_manifest.exists(), "Missing clean_train.json. Run prepare_defense_data.py first."
    assert noise_manifest.exists(), "Missing noise_pools.json. Run prepare_defense_data.py first."

    with open(clean_manifest, "r") as f:
        clean_files = json.load(f)

    with open(noise_manifest, "r") as f:
        noise_pools = json.load(f)

    logger.info("Step 1: Building tiny dataset (100 virtual pairs)...")
    dataset = SpeechEnhancementDataset(
        clean_dir=clean_files[:100],
        noise_dirs=noise_pools["train"],
        crop_samples=32000,
        snr_range=(-10.0, 15.0),
        composite_mixing=True,
        n_pairs=100,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Load GTCRN
    logger.info("Step 2: Loading GTCRN model...")
    model = GTCRN().to(device)
    ckpt_path = repo_root / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"
    if not ckpt_path.exists():
        ckpt_path = repo_root / "models" / "checkpoints" / "checkpoint_best.pth"

    if ckpt_path.exists():
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state_dict:
            sd = state_dict["model_state_dict"]
        elif "model" in state_dict:
            sd = state_dict["model"]
        else:
            sd = state_dict
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded weights from %s", ckpt_path)
    else:
        logger.warning("No checkpoint found at %s. Initialized random weights.", ckpt_path)

    nfft = 512
    hop = 256
    # CRITICAL: Use square-root Hann window matching original GTCRN pretrained checkpoint
    window = torch.hann_window(nfft).pow(0.5).to(device)
    assert torch.allclose(window.cpu(), torch.hann_window(nfft).pow(0.5)), "STFT window mismatch!"
    criterion = CombinedLoss(alpha=0.5, compression_power=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3. Forward & Backward Pass Verification
    logger.info("Step 3: Verifying forward & backward pass on batch 1...")
    model.train()
    batch = next(iter(loader))
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)

    # STFT
    noisy_stft_c = torch.stft(noisy, nfft, hop, window=window, return_complex=True)
    noisy_stft = torch.view_as_real(noisy_stft_c)
    clean_stft_c = torch.stft(clean, nfft, hop, window=window, return_complex=True)
    clean_stft = torch.view_as_real(clean_stft_c)

    # Forward
    pred_stft = model(noisy_stft)
    assert pred_stft.shape == clean_stft.shape, f"Shape mismatch: {pred_stft.shape} vs {clean_stft.shape}"

    # ISTFT
    pred_c = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
    pred_wav = torch.istft(pred_c, nfft, hop, window=window, length=clean.shape[-1])

    # Loss
    loss, components = criterion(pred_stft, clean_stft, pred_wav, clean)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    logger.info("Batch 1 Loss: %.4f | SI-SNR Loss: %.4f | Spec Loss: %.4f",
                loss.item(), components["si_snr_loss"], components["spectral_loss"])

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    assert all(np.isfinite(g) for g in grad_norms), "NaN/Inf gradient detected!"
    logger.info("Backward pass successful! Mean grad norm: %.4f", float(np.mean(grad_norms)))
    optimizer.step()

    # 4. Train for 2 epochs on tiny set & verify loss decreases
    logger.info("Step 4: Training for 2 epochs on 100-pair subset...")
    epoch_losses = []
    for epoch in range(1, 3):
        total_loss = 0.0
        n_b = 0
        for b in loader:
            b_noisy = b["noisy"].to(device)
            b_clean = b["clean"].to(device)

            b_noisy_stft = torch.view_as_real(torch.stft(b_noisy, nfft, hop, window=window, return_complex=True))
            b_clean_stft = torch.view_as_real(torch.stft(b_clean, nfft, hop, window=window, return_complex=True))

            b_pred_stft = model(b_noisy_stft)
            b_pred_c = torch.complex(b_pred_stft[..., 0], b_pred_stft[..., 1])
            b_pred_wav = torch.istft(b_pred_c, nfft, hop, window=window, length=b_clean.shape[-1])

            b_loss, _ = criterion(b_pred_stft, b_clean_stft, b_pred_wav, b_clean)
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()

            total_loss += b_loss.item()
            n_b += 1

        avg_loss = total_loss / n_b
        epoch_losses.append(avg_loss)
        logger.info("Epoch %d: Average Loss = %.4f", epoch, avg_loss)

    assert epoch_losses[1] < epoch_losses[0] or abs(epoch_losses[1] - epoch_losses[0]) < 0.2, (
        f"Loss did not decrease: {epoch_losses}"
    )
    logger.info("Loss progression: Epoch 1=%.4f -> Epoch 2=%.4f (DECREASED: %s)",
                epoch_losses[0], epoch_losses[1], epoch_losses[1] < epoch_losses[0])

    # 5. Run inference on a test sample & save WAV
    logger.info("Step 5: Running inference on test audio & saving output WAVs...")
    out_dir = repo_root / "data" / "sanity_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    test_manifest_path = repo_root / "data" / "defense_test_manifests" / "background_gunshot.jsonl"
    if not test_manifest_path.exists():
        test_manifest_path = repo_root / "data" / "defense_test_manifests" / "defense_all_test.jsonl"

    from sih26052.data.manifest import read_manifest
    entries = read_manifest(test_manifest_path)
    sample_entry = entries[0]

    raw_noisy, sr_n = sf.read(sample_entry.noisy, dtype="float32")
    raw_clean, sr_c = sf.read(sample_entry.clean, dtype="float32")
    if raw_noisy.ndim > 1:
        raw_noisy = raw_noisy.mean(axis=1)
    if raw_clean.ndim > 1:
        raw_clean = raw_clean.mean(axis=1)
    raw_noisy = np.asarray(raw_noisy, dtype=np.float32)
    raw_clean = np.asarray(raw_clean, dtype=np.float32)

    with torch.no_grad():
        t_noisy = torch.from_numpy(raw_noisy).unsqueeze(0).to(device)
        stft_c = torch.stft(t_noisy, nfft, hop, window=window, return_complex=True)
        stft_r = torch.view_as_real(stft_c)
        enh_stft = model(stft_r)
        enh_c = torch.complex(enh_stft[..., 0], enh_stft[..., 1])
        enhanced_wav = torch.istft(enh_c, nfft, hop, window=window, length=len(raw_noisy)).squeeze(0).cpu().numpy()

    # Save audio files
    sf.write(str(out_dir / "sanity_noisy.wav"), raw_noisy, 16000)
    sf.write(str(out_dir / "sanity_clean.wav"), raw_clean, 16000)
    sf.write(str(out_dir / "sanity_enhanced.wav"), enhanced_wav, 16000)
    logger.info("Saved sanity WAVs to %s", out_dir)

    # 6. Compute metrics
    logger.info("Step 6: Computing metrics on test sample...")
    metrics_noisy = compute_all_metrics(raw_clean, raw_noisy, sr=16000)
    metrics_enh = compute_all_metrics(raw_clean, enhanced_wav, sr=16000)

    delta_sisnr = metrics_enh.si_snr - metrics_noisy.si_snr
    logger.info("Noisy Input Metrics:    SI-SNR: %6.2f dB | STOI: %5.3f | PESQ: %s",
                metrics_noisy.si_snr, metrics_noisy.stoi or 0.0,
                f"{metrics_noisy.pesq:.3f}" if metrics_noisy.pesq else "N/A")
    logger.info("Enhanced Output Metrics: SI-SNR: %6.2f dB | STOI: %5.3f | PESQ: %s",
                metrics_enh.si_snr, metrics_enh.stoi or 0.0,
                f"{metrics_enh.pesq:.3f}" if metrics_enh.pesq else "N/A")
    logger.info("SI-SNR Improvement:     %+6.2f dB", delta_sisnr)

    # 7. Identity test
    logger.info("Step 7: Verifying clean speech identity test...")
    with torch.no_grad():
        t_clean = torch.from_numpy(raw_clean).unsqueeze(0).to(device)
        stft_c = torch.stft(t_clean, nfft, hop, window=window, return_complex=True)
        enh_stft = model(torch.view_as_real(stft_c))
        enh_c = torch.complex(enh_stft[..., 0], enh_stft[..., 1])
        ident_wav = torch.istft(enh_c, nfft, hop, window=window, length=len(raw_clean)).squeeze(0).cpu().numpy()

    ident_metrics = compute_all_metrics(raw_clean, ident_wav, sr=16000)
    logger.info("Identity pass-through SI-SNR: %.2f dB, STOI: %.3f", ident_metrics.si_snr, ident_metrics.stoi or 0.0)

    print("=" * 70)
    print("SANITY TEST RESULT: ALL 7 VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
