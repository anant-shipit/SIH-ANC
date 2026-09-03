import pytest
torch = pytest.importorskip("torch")

from sih26052.train.loss import SISNRLoss, CompressedSpectralLoss, CombinedLoss

class TestSISNRLoss:
    def test_identity_is_zero(self):
        loss_fn = SISNRLoss()
        x = torch.randn(2, 32000)
        
        # SI-SNR is scale-invariant, so exact match or scaled match should both be very high SNR (low loss)
        loss = loss_fn(x, x)
        assert loss.item() < -100.0  # Perfect match means SNR goes to infinity, loss goes to -infinity
        
        loss_scaled = loss_fn(x * 0.5, x)
        assert loss_scaled.item() < -100.0

    def test_noise_is_high_loss(self):
        loss_fn = SISNRLoss()
        clean = torch.randn(2, 32000)
        noisy = clean + torch.randn(2, 32000) * 10.0
        
        loss = loss_fn(noisy, clean)
        assert loss.item() > -10.0  # SNR is low, so -SNR is high

class TestCompressedSpectralLoss:
    def test_identity_is_zero(self):
        loss_fn = CompressedSpectralLoss(compression_power=0.3)
        # shape: (batch, freq, time, 2)
        x = torch.randn(2, 257, 100, 2)
        
        loss = loss_fn(x, x)
        assert loss.item() < 1e-6
        
    def test_gradient_flows(self):
        loss_fn = CompressedSpectralLoss(compression_power=0.3)
        pred = torch.randn(2, 257, 10, 2, requires_grad=True)
        target = torch.randn(2, 257, 10, 2)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

class TestCombinedLoss:
    def test_combined(self):
        loss_fn = CombinedLoss(alpha=0.5, compression_power=0.3)
        
        pred_wav = torch.randn(2, 32000, requires_grad=True)
        true_wav = torch.randn(2, 32000)
        
        pred_stft = torch.randn(2, 257, 100, 2, requires_grad=True)
        true_stft = torch.randn(2, 257, 100, 2)
        
        loss, components = loss_fn(pred_stft, true_stft, pred_wav, true_wav)
        loss.backward()
        
        assert "si_snr_loss" in components
        assert "spectral_loss" in components
        assert "total_loss" in components
        
        assert pred_wav.grad is not None
        assert pred_stft.grad is not None
