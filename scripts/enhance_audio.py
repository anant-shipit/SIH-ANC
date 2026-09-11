#!/usr/bin/env python3
"""
enhance_audio.py — Enhance any audio file using the fine-tuned GTCRN model (PyTorch or ONNX).
"""
import argparse
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


def get_default_model() -> Path:
    # Prefer best fine-tuned streaming ONNX or PyTorch checkpoint
    onnx_path = repo_root / "models" / "gtcrn_stream.onnx"
    if onnx_path.exists():
        return onnx_path
    stage2_ckpt = repo_root / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best.pth"
    if stage2_ckpt.exists():
        return stage2_ckpt
    return repo_root / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"


def enhance_with_pytorch(model_path: Path, noisy_wav: np.ndarray, sr: int, device: torch.device, spectral_floor: float = 0.025) -> np.ndarray:
    from gtcrn import GTCRN

    model = GTCRN().to(device)
    ckpt = torch.load(str(model_path), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).pow(0.5).to(device)

    noisy_t = torch.from_numpy(noisy_wav.astype(np.float32)).unsqueeze(0).to(device)
    stft = torch.view_as_real(torch.stft(noisy_t, nfft, hop, window=window, return_complex=True))

    with torch.no_grad():
        pred_stft = model(stft)
        if spectral_floor > 0.0:
            pred_stft = pred_stft + stft * spectral_floor

    pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
    enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=len(noisy_wav))
    return enhanced.squeeze(0).cpu().numpy()


def enhance_with_onnx(onnx_path: Path, noisy_wav: np.ndarray, sr: int) -> np.ndarray:
    from sih26052.runtime.ola import OverlapAdd
    from sih26052.runtime.enhancer import StreamingEnhancer

    nfft = 512
    hop = 256
    ola = OverlapAdd(nfft=nfft, hop=hop)
    enhancer = StreamingEnhancer(onnx_path, n_freq=nfft // 2 + 1)

    n_samples = len(noisy_wav)
    # Pad to integer number of hops plus extra hop to flush the pipeline
    noisy_padded = np.pad(noisy_wav, (0, hop * 2))
    output_chunks = []

    for i in range(0, len(noisy_padded), hop):
        chunk = noisy_padded[i : i + hop]
        if len(chunk) < hop:
            break
        spec_frame = ola.analyze(chunk)
        enhanced_spec = enhancer.process_frame(spec_frame)
        out_chunk = ola.synthesize(enhanced_spec)
        output_chunks.append(out_chunk)

    full_out = np.concatenate(output_chunks, axis=0)
    # Compensate 1-hop algorithmic delay (first hop is ring-buffer fill)
    return full_out[hop : n_samples + hop]


def main():
    default_model = get_default_model()
    parser = argparse.ArgumentParser(description="Enhance audio with fine-tuned GTCRN")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to input noisy WAV")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Path to output enhanced WAV")
    parser.add_argument("--model", "-m", type=Path, default=default_model, help=f"Model path (.pth, .tar, or .onnx) (default: {default_model.name})")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} does not exist.")
        sys.exit(1)

    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_enhanced.wav"

    noisy_wav, sr = sf.read(str(args.input), dtype="float32")
    if noisy_wav.ndim > 1:
        noisy_wav = noisy_wav.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        print(f"Resampling from {sr} Hz to 16000 Hz...")
        import scipy.signal
        n_resampled = int(len(noisy_wav) * 16000 / sr)
        resampled = scipy.signal.resample(noisy_wav, n_resampled)
        if isinstance(resampled, tuple):
            resampled = resampled[0]
        noisy_wav = np.asarray(resampled, dtype=np.float32)
        sr = 16000

    assert isinstance(noisy_wav, np.ndarray)

    print(f"Processing '{args.input}' ({len(noisy_wav)/sr:.2f}s) with {args.model.name}...")

    if args.model.suffix == ".onnx":
        enhanced = enhance_with_onnx(args.model, noisy_wav, sr)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enhanced = enhance_with_pytorch(args.model, noisy_wav, sr, device)

    sf.write(str(args.output), enhanced, sr)
    print(f"Enhanced audio saved to '{args.output}'!")


if __name__ == "__main__":
    main()
