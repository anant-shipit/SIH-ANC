import sys
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sih26052.train.train_gtcrn import (
    GTCRNHybridLoss,
    GTCRNSISNRLoss,
    get_window,
    forward_gtcrn,
    evaluate,
    parse_args,
    NFFT,
    HOP,
    WIN_LEN,
)


class MockGTCRN(nn.Module):
    """A minimal mock GTCRN model that accepts (B, F, T, 2) and returns same shape."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, F, T, 2) -> permute to (B, 2, F, T) for Conv2d
        b, f, t, c = x.shape
        x_perm = x.permute(0, 3, 1, 2)
        out = self.conv(x_perm)
        return out.permute(0, 2, 3, 1)


class TestGTCRNTrainingComponents:
    def test_window_is_sqrt_hann(self):
        device = torch.device("cpu")
        w = get_window(device)
        assert w.shape == (WIN_LEN,)
        # Check that it is sqrt-hann: w == hann_window(512)**0.5
        expected = torch.hann_window(WIN_LEN).pow(0.5)
        assert torch.allclose(w, expected, atol=1e-6)

    def test_gtcrn_sisnr_loss(self):
        criterion = GTCRNSISNRLoss()
        clean = torch.randn(2, 16000)
        # Identical signal -> very negative loss (high SI-SNR)
        loss_ident = criterion(clean, clean)
        assert loss_ident.item() < -50.0

        # Noisy signal -> higher loss
        noisy = clean + 5.0 * torch.randn(2, 16000)
        loss_noisy = criterion(noisy, clean)
        assert loss_noisy.item() > -5.0

        # Gradient flow
        noisy_param = nn.Parameter(noisy.clone())
        loss = criterion(noisy_param, clean)
        loss.backward()
        assert noisy_param.grad is not None

    def test_gtcrn_hybrid_loss(self):
        criterion = GTCRNHybridLoss()
        device = torch.device("cpu")
        window = get_window(device)

        pred_stft = torch.randn(2, 257, 63, 2, requires_grad=True)
        true_stft = torch.randn(2, 257, 63, 2)

        loss = criterion(pred_stft, true_stft, window)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

        loss.backward()
        assert pred_stft.grad is not None
        assert not torch.isnan(pred_stft.grad).any()

    def test_forward_gtcrn_waveform_and_stft(self):
        device = torch.device("cpu")
        model = MockGTCRN().to(device)
        noisy = torch.randn(2, 16000, device=device)

        # Waveform only
        enh_wav = forward_gtcrn(model, noisy, device, return_stft=False)
        assert isinstance(enh_wav, torch.Tensor)
        assert enh_wav.shape == (2, 16000)

        # Waveform + STFT
        enh_wav, enh_stft = forward_gtcrn(model, noisy, device, return_stft=True)
        assert isinstance(enh_wav, torch.Tensor)
        assert enh_wav.shape == (2, 16000)
        assert enh_stft.ndim == 4
        assert enh_stft.shape[0] == 2
        assert enh_stft.shape[-1] == 2

    def test_evaluate_smoke(self):
        device = torch.device("cpu")
        model = MockGTCRN().to(device)
        criterion = GTCRNHybridLoss()

        # Create dummy dataloader with (noisy, clean, name)
        noisy_data = torch.randn(4, 16000)
        clean_data = torch.randn(4, 16000)
        names = ["1.wav", "2.wav", "3.wav", "4.wav"]

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return noisy_data[idx], clean_data[idx], names[idx]

        val_loader = DataLoader(DummyDataset(), batch_size=2, shuffle=False)

        metrics = evaluate(model, val_loader, device, criterion, compute_cm=True)
        assert "loss" in metrics
        assert "si_snr" in metrics
        assert "pesq" in metrics
        assert "stoi" in metrics
        assert "cm" in metrics
        assert metrics["cm"] is not None
        assert "TP" in metrics["cm"]
