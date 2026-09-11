#!/usr/bin/env python3
"""
show_model_report.py — Inspect best model training report and evaluate confusion matrix.

Usage:
    python scripts/show_model_report.py
    python scripts/show_model_report.py --report experiments/gtcrn_stage2_finetuned/training_report.json
    python scripts/show_model_report.py --checkpoint experiments/gtcrn_stage2_finetuned/checkpoint_best.pth --manifest data/test_manifest.jsonl --save-plot
"""
import argparse
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "models" / "gtcrn"))

# Auto-reexec with project virtualenv if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import soundfile as sf
import torch
from gtcrn import GTCRN
from sih26052.eval.metrics import compute_classification_metrics


def print_header(title: str):
    width = 78
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width)


def display_training_report(report_path: Path):
    if not report_path.exists():
        print(f"[!] Report not found at: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_header("GTCRN Best Model Training Report")
    print(f"  Model Architecture  : {data.get('model', 'GTCRN')}")
    print(f"  Model Parameters    : {data.get('model_params', '48,245 (23.67K weights)')}")
    print(f"  Computational Cost  : {data.get('training_config', {}).get('model_mmacs', '33.0')} MMACs / frame (~0.15ms latency)")
    print(f"  Best Epoch          : Epoch {data.get('best_epoch', 'N/A')} (Score: {data.get('best_score', 0):.4f})")
    print(f"  Total Epochs Trained: {data.get('total_epochs_trained', 'N/A')}")

    init_m = data.get("initial_metrics", {})
    final_m = data.get("final_metrics", {})
    if init_m and final_m:
        print("\n" + "-" * 78)
        print("  METRIC EVOLUTION (Initial -> Best Checkpoint)")
        print("-" * 78)
        print(f"  {'Metric':<25} | {'Initial (Stage-1)':>18} | {'Best (Stage-2)':>18} | {'Improvement':>10}")
        print("  " + "-" * 74)
        metrics_keys = [
            ("Speech Recall (VAD)", "recall", "{:.1%}", True),
            ("Accuracy", "accuracy", "{:.2f}%", False),
            ("Precision", "precision", "{:.1%}", True),
            ("F1-Score", "f1_score", "{:.3f}", True),
            ("STOI (Intelligibility)", "stoi", "{:.4f}", True),
            ("SI-SNR Output", "si_snr", "{:.2f} dB", False),
            ("SI-SNR Improvement (Δ)", "si_snr_improvement", "{:+.2f} dB", False),
            ("PESQ (Perceptual Quality)", "pesq", "{:.4f}", True),
        ]
        for label, key, fmt, is_ratio in metrics_keys:
            v_init = init_m.get(key, 0.0)
            v_final = final_m.get(key, 0.0)
            diff = v_final - v_init
            if is_ratio:
                diff_str = f"{diff:+.1%}"
            elif "%" in fmt:
                diff_str = f"{diff:+.2f}%"
            else:
                diff_str = f"{diff:+.3f}"
            print(f"  {label:<25} | {fmt.format(v_init):>18} | {fmt.format(v_final):>18} | {diff_str:>10}")

    per_snr = data.get("per_snr", {})
    if per_snr:
        print("\n" + "-" * 78)
        print("  PERFORMANCE ACROSS SNR LEVELS (-10 dB to +15 dB)")
        print("-" * 78)
        print(f"  {'Input SNR':<12} | {'SI-SDR Δ':>12} | {'PESQ Out':>10} | {'STOI Out':>10}")
        print("  " + "-" * 52)
        for snr_val in sorted(per_snr.keys(), key=lambda x: float(x)):
            m = per_snr[snr_val]
            print(
                f"  {float(snr_val):+5.1f} dB    | "
                f"{m.get('si_sdr_improvement', 0.0):+10.2f} dB | "
                f"{m.get('pesq_output', 0.0):10.3f} | "
                f"{m.get('stoi_output', 0.0):10.3f}"
            )

    per_noise = data.get("per_noise_type", {})
    if per_noise:
        print("\n" + "-" * 78)
        print("  TACTICAL NOISE ROBUSTNESS BREAKDOWN")
        print("-" * 78)
        print(f"  {'Noise Category':<22} | {'Files':>6} | {'SI-SDR Δ':>12} | {'PESQ Out':>10} | {'STOI Out':>10}")
        print("  " + "-" * 68)
        for ntype, m in sorted(per_noise.items()):
            print(
                f"  {ntype:<22} | {m.get('n_files', 0):>6} | "
                f"{m.get('si_sdr_improvement', 0.0):+10.2f} dB | "
                f"{m.get('pesq_output', 0.0):10.3f} | "
                f"{m.get('stoi_output', 0.0):10.3f}"
            )


def compute_confusion_matrix(checkpoint_path: Path, manifest_path: Path, spectral_floor: float = 0.025, save_plot_path: Path = None):
    if not checkpoint_path.exists():
        print(f"[!] Checkpoint not found at: {checkpoint_path}")
        return
    if not manifest_path.exists():
        print(f"[!] Manifest not found at: {manifest_path}")
        return

    print_header("Evaluating Confusion Matrix on Test Manifest")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Manifest  : {manifest_path}")
    print(f"  Comfort Noise Floor: {spectral_floor}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTCRN().to(device)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).pow(0.5).to(device)

    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    m_dir = manifest_path.resolve().parent
    for entry in entries:
        noisy_p = Path(entry["noisy"])
        if not noisy_p.exists():
            noisy_p = m_dir / noisy_p if (m_dir / noisy_p).exists() else m_dir.parent / noisy_p
        clean_p = Path(entry["clean"])
        if not clean_p.exists():
            clean_p = m_dir / clean_p if (m_dir / clean_p).exists() else m_dir.parent / clean_p

        noisy, sr_n = sf.read(str(noisy_p), dtype="float32")
        clean, sr_c = sf.read(str(clean_p), dtype="float32")

        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
            noisy_stft = torch.view_as_real(torch.stft(noisy_t, nfft, hop, window=window, return_complex=True))
            pred_stft = model(noisy_stft)
            if spectral_floor > 0.0:
                pred_stft = pred_stft + noisy_stft * spectral_floor

            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=min_len).squeeze(0).cpu().numpy()

        clf = compute_classification_metrics(clean, enhanced, frame_len=nfft, hop_len=hop, threshold_db=-35.0)
        total_tp += clf.get("tp", 0.0)
        total_fp += clf.get("fp", 0.0)
        total_tn += clf.get("tn", 0.0)
        total_fn += clf.get("fn", 0.0)

    total_frames = total_tp + total_fp + total_tn + total_fn
    actual_speech = total_tp + total_fn
    actual_noise = total_tn + total_fp
    pred_speech = total_tp + total_fp
    pred_noise = total_tn + total_fn

    accuracy = (total_tp + total_tn) / (total_frames + 1e-12) * 100.0
    precision = total_tp / (pred_speech + 1e-12)
    recall = total_tp / (actual_speech + 1e-12)
    specificity = total_tn / (actual_noise + 1e-12) * 100.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    print("=" * 78)
    print("                    CONFUSION MATRIX (FRAME-LEVEL VAD)                   ")
    print("=" * 78)
    print(f" Total Frames Evaluated : {int(total_frames):,} (Time: {total_frames * hop / 16000:.1f}s audio)")
    print(f" Actual Speech Frames   : {int(actual_speech):,} ({actual_speech / total_frames:.1%})")
    print(f" Actual Noise/Silence   : {int(actual_noise):,} ({actual_noise / total_frames:.1%})")
    print("-" * 78)
    print(f"{'':22} | {'PREDICTED SPEECH':^24} | {'PREDICTED NOISE / SIL':^24}")
    print("-" * 78)
    print(
        f"{'ACTUAL SPEECH':<22} | "
        f"TP = {int(total_tp):<6} ({total_tp / (actual_speech + 1e-12):>6.1%})       | "
        f"FN = {int(total_fn):<6} ({total_fn / (actual_speech + 1e-12):>6.1%})"
    )
    print(
        f"{'ACTUAL NOISE':<22} | "
        f"FP = {int(total_fp):<6} ({total_fp / (actual_noise + 1e-12):>6.1%})       | "
        f"TN = {int(total_tn):<6} ({total_tn / (actual_noise + 1e-12):>6.1%})"
    )
    print("-" * 78)
    print("  KEY PERFORMANCE METRICS:")
    print(f"  • Accuracy (Overall Correctness) : {accuracy:.2f}%")
    print(f"  • Recall (Speech Preservation)   : {recall:.1%}  <-- Measures zero speech-cutting (Target: 90%+)")
    print(f"  • Precision (Speech Purity)      : {precision:.1%}")
    print(f"  • Specificity (Noise Suppression): {specificity:.1f}%")
    print(f"  • F1-Score                       : {f1:.4f}")
    print("=" * 78)

    if save_plot_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            cm = np.array([[int(total_tp), int(total_fn)], [int(total_fp), int(total_tn)]])
            cm_pct = np.array([
                [total_tp / (actual_speech + 1e-12), total_fn / (actual_speech + 1e-12)],
                [total_fp / (actual_noise + 1e-12), total_tn / (actual_noise + 1e-12)],
            ])

            fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
            cax = ax.matshow(cm_pct, cmap="Blues", vmin=0, vmax=1.0)
            fig.colorbar(cax)

            labels = [
                [f"True Positive (TP)\n{int(total_tp):,}\n({cm_pct[0,0]:.1%})", f"False Negative (FN)\n{int(total_fn):,}\n({cm_pct[0,1]:.1%})"],
                [f"False Positive (FP)\n{int(total_fp):,}\n({cm_pct[1,0]:.1%})", f"True Negative (TN)\n{int(total_tn):,}\n({cm_pct[1,1]:.1%})"],
            ]

            for i in range(2):
                for j in range(2):
                    color = "white" if cm_pct[i, j] > 0.5 else "black"
                    ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Speech (1)", "Noise (0)"], fontsize=11, fontweight="bold")
            ax.set_yticklabels(["Speech (1)", "Noise (0)"], fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label (Enhanced Audio)", fontsize=12, labelpad=10, fontweight="bold")
            ax.set_ylabel("Actual Label (Clean Ground Truth)", fontsize=12, labelpad=10, fontweight="bold")
            ax.set_title(f"GTCRN Confusion Matrix\nAccuracy: {accuracy:.1f}% | Recall: {recall:.1%} | F1: {f1:.3f}", fontsize=13, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(str(save_plot_path), bbox_inches="tight")
            plt.close()
            print(f"\n[+] Visual confusion matrix saved to: {save_plot_path}")
        except Exception as e:
            print(f"[!] Warning: Could not generate confusion matrix plot: {e}")


def main():
    default_report = repo_root / "experiments" / "gtcrn_stage2_finetuned" / "training_report.json"
    if not default_report.exists():
        default_report = repo_root / "experiments" / "gtcrn_optimized_v1" / "training_report.json"

    default_ckpt = repo_root / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best.pth"
    if not default_ckpt.exists():
        default_ckpt = repo_root / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best.pth"

    default_manifest = repo_root / "data" / "test_manifest.jsonl"
    default_plot = repo_root / "experiments" / "gtcrn_stage2_finetuned" / "confusion_matrix.png"

    parser = argparse.ArgumentParser(description="Inspect best model report and confusion matrix")
    parser.add_argument("--report", type=Path, default=default_report, help="Path to training_report.json")
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt, help="Path to checkpoint (.pth)")
    parser.add_argument("--manifest", type=Path, default=default_manifest, help="Path to test_manifest.jsonl")
    parser.add_argument("--spectral-floor", type=float, default=0.025, help="Comfort noise floor")
    parser.add_argument("--save-plot", type=Path, default=default_plot, help="Path to save confusion matrix image")
    parser.add_argument("--skip-eval", action="store_true", help="Only display JSON report, skip test manifest evaluation")
    args = parser.parse_args()

    display_training_report(args.report)

    if not args.skip_eval:
        compute_confusion_matrix(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            spectral_floor=args.spectral_floor,
            save_plot_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
