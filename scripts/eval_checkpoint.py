#!/usr/bin/env python3
"""
eval_checkpoint.py — Evaluate a GTCRN PyTorch checkpoint on a test manifest.
"""
import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "models" / "gtcrn"))

# Auto-reexec with project virtualenv if running under system python without torch/numpy
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import torch
from gtcrn import GTCRN
from sih26052.eval.harness import run_eval_harness


def get_default_checkpoint() -> Path:
    ckpt_dir = repo_root / "models" / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            return checkpoints[-1]
    dns3_ckpt = repo_root / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"
    if dns3_ckpt.exists():
        return dns3_ckpt
    return repo_root / "checkpoint.pth"


def main():
    default_ckpt = get_default_checkpoint()
    default_manifest = repo_root / "data" / "test_manifest.jsonl"
    parser = argparse.ArgumentParser(description="Evaluate GTCRN checkpoint on manifest")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help=f"Path to .tar or .pth checkpoint (default: {default_ckpt})",
    )
    parser.add_argument("--spectral-floor", type=float, default=0.0, help="Anti-gating spectral floor (e.g. 0.025)")
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = GTCRN().to(device)
    
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).pow(0.5).to(device)

    def enhance_fn(noisy_wav: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy_wav.astype(np.float32)).unsqueeze(0).to(device)
            noisy_stft_complex = torch.stft(noisy_t, nfft, hop, window=window, return_complex=True)
            noisy_stft = torch.view_as_real(noisy_stft_complex)

            pred_stft = model(noisy_stft)
            if args.spectral_floor > 0.0:
                pred_stft = pred_stft + noisy_stft * args.spectral_floor

            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=len(noisy_wav))
            return enhanced.squeeze(0).cpu().numpy()

    print(f"Evaluating {args.checkpoint} on {args.manifest}...")
    results = run_eval_harness(args.manifest, enhance_fn=enhance_fn, sr=16000, align=True)
    print("\nResults Table:")
    print(results.format_table())


if __name__ == "__main__":
    main()
