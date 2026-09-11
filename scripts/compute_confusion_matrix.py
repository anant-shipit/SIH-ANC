#!/usr/bin/env python3
"""
compute_confusion_matrix.py — Compute Voice Activity Detection (VAD) & Signal Confusion Matrix
for baseline noisy speech vs GTCRN enhanced speech.
"""
import argparse
import json
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
        try:
            import numpy
            import torch
        except ImportError:
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import soundfile as sf
import torch
from gtcrn import GTCRN
from sih26052.data.manifest import read_manifest


def compute_vad_frames(wav: np.ndarray, frame_len: int = 512, hop_len: int = 256, threshold_db: float = -35.0) -> np.ndarray:
    """Compute frame-level energy and threshold to determine binary activity."""
    n_frames = 1 + (len(wav) - frame_len) // hop_len
    if n_frames <= 0:
        return np.array([False])
    
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n_frames, frame_len),
        strides=(wav.strides[0] * hop_len, wav.strides[0]),
    )
    frame_energy = np.mean(frames ** 2, axis=1) + 1e-12
    peak_energy = np.max(frame_energy)
    energy_db = 10.0 * np.log10(frame_energy / (peak_energy + 1e-12))
    return energy_db > threshold_db


def evaluate_confusion_matrix(manifest_path: Path, checkpoint_path: Path, device_str: str = "cuda"):
    entries = read_manifest(manifest_path)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load model
    model = GTCRN().to(device)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).to(device)

    # Accumulators for confusion matrices
    # [ [TN, FP],
    #   [FN, TP] ]
    cm_noisy = np.zeros((2, 2), dtype=np.int64)
    cm_enhanced = np.zeros((2, 2), dtype=np.int64)

    for entry in entries:
        c_path = Path(entry.clean)
        if not c_path.exists():
            c_path = repo_root / c_path
        n_path = Path(entry.noisy)
        if not n_path.exists():
            n_path = repo_root / n_path

        clean, sr = sf.read(str(c_path), dtype="float32")
        noisy, _ = sf.read(str(n_path), dtype="float32")

        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)

        # Enhance with model
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
            noisy_stft = torch.stft(noisy_t, nfft, hop, window=window, return_complex=True)
            noisy_stft_real = torch.view_as_real(noisy_stft)

            pred_stft = model(noisy_stft_real)
            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=len(noisy)).squeeze(0).cpu().numpy()

        # Ground truth VAD on clean speech
        y_true = compute_vad_frames(clean, frame_len=nfft, hop_len=hop, threshold_db=-30.0).astype(int)

        # Predicted VAD on noisy signal
        y_noisy = compute_vad_frames(noisy, frame_len=nfft, hop_len=hop, threshold_db=-30.0).astype(int)

        # Predicted VAD on enhanced signal
        y_enhanced = compute_vad_frames(enhanced, frame_len=nfft, hop_len=hop, threshold_db=-30.0).astype(int)

        min_len = min(len(y_true), len(y_noisy), len(y_enhanced))
        y_true = y_true[:min_len]
        y_noisy = y_noisy[:min_len]
        y_enhanced = y_enhanced[:min_len]

        for yt, yn, ye in zip(y_true, y_noisy, y_enhanced):
            cm_noisy[yt, yn] += 1
            cm_enhanced[yt, ye] += 1

    def calc_metrics(cm):
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        total = tn + fp + fn + tp
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Total": int(total),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "Specificity": float(spec),
            "F1_Score": float(f1),
        }

    res_noisy = calc_metrics(cm_noisy)
    res_enhanced = calc_metrics(cm_enhanced)

    output = {
        "baseline_noisy": res_noisy,
        "gtcrn_enhanced": res_enhanced,
    }
    return output


def print_confusion_matrix_table(title: str, m: dict):
    print(f"\n==================== {title} ====================")
    print("                       Predicted Inactive (0) | Predicted Active (1)")
    print(f"Actual Inactive (0):   TN = {m['TN']:<18} | FP = {m['FP']:<18}")
    print(f"Actual Active (1):     FN = {m['FN']:<18} | TP = {m['TP']:<18}")
    print("----------------------------------------------------------------------")
    print(f"Accuracy:    {m['Accuracy'] * 100:.2f}%")
    print(f"Precision:   {m['Precision'] * 100:.2f}%")
    print(f"Recall:      {m['Recall'] * 100:.2f}%")
    print(f"Specificity: {m['Specificity'] * 100:.2f}%")
    print(f"F1 Score:    {m['F1_Score'] * 100:.2f}%")


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
    default_out_json = repo_root / "data" / "confusion_matrix.json"

    parser = argparse.ArgumentParser(description="Compute Signal/VAD Confusion Matrix")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help=f"Path to checkpoint (default: {default_ckpt})",
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--out-json", type=Path, default=default_out_json)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest {args.manifest} not found. Generating test set automatically...")
        from scripts.create_test_set import main as gen_test_set
        gen_test_set()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Using test manifest:   {args.manifest}")
    results = evaluate_confusion_matrix(args.manifest, args.checkpoint)

    print_confusion_matrix_table("RAW NOISY SPEECH (BASELINE)", results["baseline_noisy"])
    print_confusion_matrix_table("GTCRN ENHANCED SPEECH (FINE-TUNED)", results["gtcrn_enhanced"])

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved confusion matrix data to {args.out_json}")


if __name__ == "__main__":
    main()
