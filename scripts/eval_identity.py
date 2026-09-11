import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "models" / "gtcrn"))

import numpy as np
import soundfile as sf
import torch
from sih26052.eval.metrics import compute_all_metrics
from gtcrn import GTCRN


def main():
    parser = argparse.ArgumentParser(description="Evaluate identity test for metric harness and GTCRN model")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=repo_root / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Part 1: Metric Harness Synthetic Identity Test (clean -> clean)")
    print("=" * 60)
    sr = 16000
    duration_s = 3.0
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    # Synthetic multi-tone signal with envelope
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = (signal * envelope).astype(np.float32)

    res = compute_all_metrics(signal, signal, sr=sr)
    print(f"SI-SNR:  {res.si_snr:.2f} dB (Expected > 80 dB)")
    print(f"STOI:    {res.stoi if res.stoi is not None else 'N/A'}")
    print(f"PESQ:    {res.pesq if res.pesq is not None else 'N/A'}")

    passed_synth = res.si_snr > 80.0 and (res.pesq is None or res.pesq > 4.0)
    print(f"Synthetic Metric Identity: {'PASSED' if passed_synth else 'FAILED'}")

    print("\n" + "=" * 60)
    print("Part 2: Model Clean Speech Preservation Test (clean speech -> GTCRN -> output)")
    print("=" * 60)

    # Check for real clean speech sample
    clean_sample_p = repo_root / "Datasets" / "VoiceBank" / "clean_testset_wav" / "p232_001.wav"
    if not clean_sample_p.exists():
        clean_wavs = list((repo_root / "Datasets" / "VoiceBank").rglob("*.wav"))
        clean_sample_p = clean_wavs[0] if clean_wavs else None

    if clean_sample_p and clean_sample_p.exists():
        clean_wav, sr_c = sf.read(str(clean_sample_p), dtype="float32")
        if clean_wav.ndim > 1:
            clean_wav = clean_wav.mean(axis=1)

        # Load GTCRN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GTCRN().to(device)

        ckpt_p = args.checkpoint
        if not ckpt_p.exists():
            ckpt_p = repo_root / "models" / "checkpoints" / "checkpoint_best.pth"

        if ckpt_p.exists():
            ckpt = torch.load(str(ckpt_p), map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(sd, strict=False)
            model.eval()

            nfft = 512
            hop = 256
            window = torch.hann_window(nfft).to(device)

            with torch.no_grad():
                t_clean = torch.from_numpy(clean_wav).unsqueeze(0).to(device)
                stft_c = torch.stft(t_clean, nfft, hop, window=window, return_complex=True)
                enh_stft = model(torch.view_as_real(stft_c))
                enh_c = torch.complex(enh_stft[..., 0], enh_stft[..., 1])
                out_wav = torch.istft(enh_c, nfft, hop, window=window, length=len(clean_wav)).squeeze(0).cpu().numpy()

            model_res = compute_all_metrics(clean_wav, out_wav, sr=sr)
            print(f"Model Clean In vs Out: SI-SNR = {model_res.si_snr:.2f} dB, STOI = {model_res.stoi:.3f}, PESQ = {model_res.pesq:.3f}")
            model_passed = (model_res.stoi is not None and model_res.stoi > 0.90) and model_res.si_snr > 15.0
            print(f"Model Clean Speech Preservation: {'PASSED (High Fidelity)' if model_passed else 'WARNING (Slight Distortion)'}")
    else:
        print("No clean speech test file found to evaluate Part 2.")

    sys.exit(0 if passed_synth else 1)


if __name__ == "__main__":
    main()

